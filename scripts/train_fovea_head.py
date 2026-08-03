#!/usr/bin/env python3
"""
Train a fovea coordinate head on top of the frozen v2 disc model.

Why
---
Fovea localisation is currently a hand-written heuristic (green scan-line anchor,
then darkest-pixel search) whose starting estimate is derived from the detected
disc. It carries a mean error of 76 px against the clinician's marking, versus
16 px for the learned disc head, and its dependence on the disc propagates disc
error into every fovea. It is also the largest single contributor to the
fovea-to-GA measurement.

No new annotation is needed: the fovea is already marked in `raw_marked/`, and
`scripts/build_fovea_labels.py` extracts it into en-face crop coordinates.

Why a coordinate head rather than a second heatmap channel
----------------------------------------------------------
A 2-channel heatmap was tried first and does not work at this data size. With
~31 training images, MSE against a mostly-zero target is minimised by hedging:
the decoder emitted a diffuse blob (peak/mean 5.4 against the target's 28.6),
argmax inside it was close to arbitrary, and validation error swung between 33
and 500 px across consecutive epochs. Sharing the decoder also degraded disc
localisation, which was the one part of the pipeline already working.

Regressing two numbers off the frozen bottleneck is a far smaller problem, and
freezing everything upstream means disc detection is unchanged *by construction*
rather than by hope. This mirrors how `height_head` was added to this model.

The trade-off is real and worth stating: the head pools globally, so it cannot
resolve fine spatial detail. It is a coarse anatomical estimate — which is also
what the 76 px heuristic it replaces is.

Usage:
  python scripts/train_fovea_head.py
  python scripts/train_fovea_head.py --val-fold 1 --epochs 60
  python scripts/train_fovea_head.py --all-folds        # cross-validate
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.retfound_unet import RETFound_UNet  # noqa: E402
from scripts.make_splits import split_dataframe  # noqa: E402

CONFIG = {
    'img_size': 224,
    'batch_size': 4,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 60,
    'early_stopping_patience': 15,
    'seed': 42,
    'data_dir': 'data/training/en_face',
    'disc_csv': 'data/training/disc_labels_v2.csv',
    'fovea_csv': 'data/training/fovea_labels_v1.csv',
    'split_file': 'data/splits/splits_v1.json',
    'checkpoint_path': 'weights/best_disc_model_v2.pth',
    'save_path': 'weights/best_fovea_head.pth',
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FoveaDataset(Dataset):
    """En-face crops with the fovea as a fraction of image width/height.

    Normalising by image size rather than working in pixels keeps both axes on
    the same scale despite the crops being non-square, and makes the target
    invariant to the resize.
    """

    def __init__(self, dataframe, img_dir, img_size=224, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(os.path.join(self.img_dir, row['filename']))
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['filename']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        keypoints = [(float(row['fovea_x']), float(row['fovea_y']))]

        if self.transform:
            t = self.transform(image=image, keypoints=keypoints)
            image = t['image']
            kps = t['keypoints']
            if kps:
                fx, fy = kps[0][0] / self.img_size, kps[0][1] / self.img_size
            else:
                fx, fy = keypoints[0][0] / w_orig, keypoints[0][1] / h_orig
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            fx, fy = keypoints[0][0] / w_orig, keypoints[0][1] / h_orig

        return {
            'image': image,
            'fovea': torch.tensor([fx, fy], dtype=torch.float32),
            'orig_size': torch.tensor([w_orig, h_orig], dtype=torch.float32),
            'filename': row['filename'],
        }


def build_transforms(img_size: int):
    kp = A.KeypointParams(format='xy', remove_invisible=False)
    train = A.Compose([
        A.Resize(img_size, img_size),
        # No horizontal flip: it swaps OD/OS handedness, and where the fovea sits
        # relative to the disc is exactly what the head has to learn.
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=10, p=0.7),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=kp)
    val = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=kp)
    return train, val


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, mean error px, median error px) in original image pixels."""
    model.eval()
    total_loss = 0.0
    errors = []
    for batch in loader:
        images = batch['image'].to(device)
        target = batch['fovea'].to(device)
        _, pred = model(images, predict_fovea=True)
        total_loss += criterion(pred, target).item()

        p = pred.cpu().numpy()
        t = batch['fovea'].numpy()
        sizes = batch['orig_size'].numpy()
        for i in range(len(p)):
            w, h = float(sizes[i][0]), float(sizes[i][1])
            errors.append(float(np.hypot((p[i][0] - t[i][0]) * w, (p[i][1] - t[i][1]) * h)))

    errors = np.array(errors)
    return total_loss / max(len(loader), 1), float(errors.mean()), float(np.median(errors))


def load_labels(cfg) -> pd.DataFrame:
    disc = pd.read_csv(cfg['disc_csv'])
    fovea = pd.read_csv(cfg['fovea_csv'])
    return disc.merge(fovea[['filename', 'fovea_x', 'fovea_y']], on='filename', how='inner')


def build_model(cfg, device):
    """Load the v2 disc model and freeze everything except the fovea head."""
    model = RETFound_UNet.load_pretrained_and_add_height_head(
        cfg['checkpoint_path'], img_size=cfg['img_size'], freeze_encoder=False)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fovea_head.parameters():
        p.requires_grad = True
    return model.to(device)


def train_fold(cfg, df, val_fold, device, epochs, verbose=True):
    train_df, val_df = split_dataframe(df, cfg['split_file'], val_fold)
    train_tf, val_tf = build_transforms(cfg['img_size'])

    train_loader = DataLoader(
        FoveaDataset(train_df, cfg['data_dir'], cfg['img_size'], train_tf),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(
        FoveaDataset(val_df, cfg['data_dir'], cfg['img_size'], val_tf),
        batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    model = build_model(cfg, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Train {len(train_df)}  Val {len(val_df)}  | trainable params: {trainable:,}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.fovea_head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_err = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()          # frozen backbone: keep norm stats fixed
        model.decoder_blocks.eval()
        model.height_head.eval()

        running = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            target = batch['fovea'].to(device)
            optimizer.zero_grad()
            _, pred = model(images, predict_fovea=True)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val_loss, mean_err, median_err = evaluate(model, val_loader, criterion, device)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  ep {epoch:3d}/{epochs} | train {running / max(len(train_loader),1):.5f} "
                  f"| val {val_loss:.5f} | fovea mean {mean_err:6.1f}px  median {median_err:6.1f}px")

        if mean_err < best_err:
            best_err = mean_err
            best_state = {k: v.detach().cpu().clone() for k, v in model.fovea_head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg['early_stopping_patience']:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    return best_err, best_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the fovea coordinate head.")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG['epochs'])
    parser.add_argument("--seed", type=int, default=CONFIG['seed'])
    parser.add_argument("--all-folds", action="store_true",
                        help="Cross-validate over every fold and report the spread")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    set_seed(args.seed)
    device = get_device()

    for key in ('disc_csv', 'fovea_csv', 'checkpoint_path'):
        if not os.path.exists(cfg[key]):
            print(f"[ERROR] Missing {key}: {cfg[key]}")
            return 1

    df = load_labels(cfg)
    print(f"=== Fovea head | device {device} | {len(df)} labelled images ===\n")

    if args.all_folds:
        import json
        n_folds = len(json.loads(open(cfg['split_file']).read())['folds'])
        results = []
        for fold in range(n_folds):
            print(f"--- fold {fold} ---")
            err, _ = train_fold(cfg, df, fold, device, args.epochs, verbose=False)
            print(f"    best val fovea error: {err:.1f} px")
            results.append(err)
        arr = np.array(results)
        print(f"\nCross-validated fovea error: {arr.mean():.1f} +- {arr.std(ddof=1):.1f} px "
              f"(per fold: {', '.join(f'{r:.0f}' for r in arr)})")
        print(f"Heuristic baseline: 76 px")
        return 0

    print(f"--- val fold {args.val_fold} ---")
    best_err, best_state = train_fold(cfg, df, args.val_fold, device, args.epochs)

    torch.save({'fovea_head_state_dict': best_state,
                'val_fovea_error_px': best_err,
                'val_fold': args.val_fold,
                'base_checkpoint': cfg['checkpoint_path']}, cfg['save_path'])
    print(f"\nSaved {cfg['save_path']}")
    print(f"  best val fovea error: {best_err:.1f} px   (heuristic baseline: 76 px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
