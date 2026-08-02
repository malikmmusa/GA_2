"""
Train RETFound U-Net for optic disc detection with simultaneous height regression.

Two-phase training:
  Phase 1: Freeze encoder + decoder + final_conv; train height_head only.
  Phase 2: Unfreeze everything with differential learning rates.
"""

import os
import sys
import csv

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.retfound_unet import RETFound_UNet, HeatmapGenerator, get_coordinates_from_heatmap
from scripts.make_splits import split_dataframe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    'img_size': 224,
    'batch_size': 4,
    'lambda_height': 10.0,
    'sigma': 20,
    'phase1_epochs': 30,
    'phase2_epochs': 30,
    'early_stopping_patience': 10,
    'phase1_lr': 1e-3,
    'phase2_lr_encoder': 1e-5,
    'phase2_lr_head': 1e-4,
    'seed': 42,
    'data_dir': 'data/training/en_face',
    'csv_file': 'data/training/disc_labels_v2.csv',
    # Frozen eye-grouped split (scripts/make_splits.py). Replaces the old random
    # 80/20 filename split, which put both timepoints of the same eye on opposite
    # sides — 9 of 10 val images had their partner in train.
    'split_file': 'data/splits/splits_v1.json',
    'val_fold': 0,
    'checkpoint_path': 'weights/best_disc_model.pth',
    'save_path': 'weights/best_disc_model_v2.pth',
    'phase1_save_path': 'weights/disc_model_phase1.pth',
}

# ---------------------------------------------------------------------------
# Device selection: MPS > CUDA > CPU
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DiscHeightDataset(Dataset):
    """
    Loads en-face OCT images and disc labels (center + height).

    CSV columns: filename, disc_x, disc_y, disc_height
    disc_height is in pixels of the *original* image; we normalize it by h_orig
    before any resize so the target is always in [0, 1] regardless of augmentation.
    """

    def __init__(self, dataframe, img_dir, img_size=224, sigma=20, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.sigma = sigma
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']

        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = image.shape[:2]
        x_orig = float(row['disc_x'])
        y_orig = float(row['disc_y'])
        disc_height_orig = float(row['disc_height'])

        # Normalize height by original image height BEFORE any resize/augment.
        # ShiftScaleRotate only perturbs the crop window; the final output is
        # always img_size x img_size, so the ratio is preserved relative to the
        # full-resolution image.
        height_normalized = disc_height_orig / h_orig

        if self.transform:
            transformed = self.transform(image=image, keypoints=[(x_orig, y_orig)])
            image = transformed['image']       # (C, H, W) tensor
            keypoints = transformed['keypoints']

            if keypoints:
                x_trans, y_trans = keypoints[0][0], keypoints[0][1]
            else:
                # Keypoint fell outside the augmented crop; scale naively
                x_trans = x_orig * (self.img_size / w_orig)
                y_trans = y_orig * (self.img_size / h_orig)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            x_trans = x_orig * (self.img_size / w_orig)
            y_trans = y_orig * (self.img_size / h_orig)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        heatmap = HeatmapGenerator(x_trans, y_trans, self.img_size, self.img_size,
                                   sigma=self.sigma)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()  # (1, H, W)

        return {
            'image': image,
            'heatmap': heatmap,
            'height_normalized': torch.tensor([height_normalized], dtype=torch.float32),
            'h_orig': torch.tensor([h_orig], dtype=torch.float32),
            'coords': torch.tensor([x_trans, y_trans], dtype=torch.float32),
            'filename': filename,
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_transforms(img_size: int):
    kp_params = A.KeypointParams(format='xy', remove_invisible=False)

    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=kp_params)

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=kp_params)

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class DiscLoss(nn.Module):
    def __init__(self, lambda_height: float = 10.0):
        super().__init__()
        self.lambda_height = lambda_height
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, heatmap_pred, heatmap_gt, height_pred, height_gt):
        heatmap_loss = self.mse(heatmap_pred, heatmap_gt)
        height_loss = self.smooth_l1(height_pred, height_gt)
        total = heatmap_loss + self.lambda_height * height_loss
        return total, heatmap_loss, height_loss


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, phase1_frozen_backbone=False):
    model.train()
    if phase1_frozen_backbone:
        model.encoder.eval()
        model.decoder_blocks.eval()
        model.final_conv.eval()
        model.height_head.train()
    total_loss = hmap_loss_sum = height_loss_sum = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        heatmap_gt = batch['heatmap'].to(device)
        height_gt = batch['height_normalized'].to(device)

        optimizer.zero_grad()
        heatmap_pred, height_pred = model(images, predict_height=True)
        loss, hmap_l, height_l = criterion(heatmap_pred, heatmap_gt, height_pred, height_gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        hmap_loss_sum += hmap_l.item()
        height_loss_sum += height_l.item()

    n = len(loader)
    return total_loss / n, hmap_loss_sum / n, height_loss_sum / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = hmap_loss_sum = 0.0
    height_mae_px = 0.0
    center_error_px = 0.0
    count = 0

    for batch in loader:
        images = batch['image'].to(device)
        heatmap_gt = batch['heatmap'].to(device)
        height_gt = batch['height_normalized'].to(device)
        h_orig = batch['h_orig']            # (B, 1) CPU
        true_coords = batch['coords']       # (B, 2) CPU

        heatmap_pred, height_pred = model(images, predict_height=True)
        loss, hmap_l, _ = criterion(heatmap_pred, heatmap_gt, height_pred, height_gt)

        total_loss += loss.item()
        hmap_loss_sum += hmap_l.item()

        # Height MAE in original pixels
        height_pred_np = height_pred.cpu().numpy().squeeze(-1)   # (B,)
        height_gt_np = height_gt.cpu().numpy().squeeze(-1)       # (B,)
        h_orig_np = h_orig.numpy().squeeze(-1)                   # (B,)
        height_mae_px += float(np.abs(
            (height_pred_np - height_gt_np) * h_orig_np
        ).sum())

        # Center localization error
        preds_np = heatmap_pred.cpu().numpy().squeeze(1)         # (B, H, W)
        B = preds_np.shape[0]
        for i in range(B):
            pred_x, pred_y = get_coordinates_from_heatmap(preds_np[i])
            tx, ty = true_coords[i].numpy()
            center_error_px += float(np.sqrt((pred_x - tx) ** 2 + (pred_y - ty) ** 2))

        count += B

    n = len(loader)
    return (
        total_loss / n,
        hmap_loss_sum / n,
        height_mae_px / count,
        center_error_px / count,
    )


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float('inf')
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"=== Training disc detector with height head | device: {device} ===\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not os.path.exists(CONFIG['csv_file']):
        print(f"[ERROR] CSV not found: {CONFIG['csv_file']}")
        sys.exit(1)

    df = pd.read_csv(CONFIG['csv_file'])
    required_cols = {'filename', 'disc_x', 'disc_y', 'disc_height'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"[ERROR] CSV missing columns: {missing}")
        sys.exit(1)

    train_df, val_df = split_dataframe(df, CONFIG['split_file'], CONFIG['val_fold'])
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  (val = fold {CONFIG['val_fold']})")

    train_transform, val_transform = build_transforms(CONFIG['img_size'])

    train_ds = DiscHeightDataset(train_df, CONFIG['data_dir'],
                                 CONFIG['img_size'], CONFIG['sigma'], train_transform)
    val_ds = DiscHeightDataset(val_df, CONFIG['data_dir'],
                               CONFIG['img_size'], CONFIG['sigma'], val_transform)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"[ERROR] Checkpoint not found: {CONFIG['checkpoint_path']}")
        sys.exit(1)

    model = RETFound_UNet.load_pretrained_and_add_height_head(
        CONFIG['checkpoint_path'],
        img_size=CONFIG['img_size'],
        freeze_encoder=False,
    )

    criterion = DiscLoss(lambda_height=CONFIG['lambda_height'])
    os.makedirs('weights', exist_ok=True)

    # ==================================================================
    # Phase 1: Freeze encoder + decoder + final_conv; train height_head
    # ==================================================================
    print("\n--- Phase 1: height_head only ---")

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder_blocks.parameters():
        param.requires_grad = False
    for param in model.final_conv.parameters():
        param.requires_grad = False
    for param in model.height_head.parameters():
        param.requires_grad = True

    model.encoder.eval()
    model.decoder_blocks.eval()
    model.final_conv.eval()
    model.height_head.train()

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Phase 1): {trainable_p1:,}")

    model.to(device)

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['phase1_lr'],
    )

    early_stop = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    best_p1_loss = float('inf')

    for epoch in range(1, CONFIG['phase1_epochs'] + 1):
        tr_loss, tr_hmap, tr_h = train_one_epoch(
            model, train_loader, optimizer_p1, criterion, device,
            phase1_frozen_backbone=True,
        )
        val_loss, val_hmap, val_h_mae, val_center = validate(model, val_loader,
                                                              criterion, device)

        print(
            f"P1 Ep {epoch:3d}/{CONFIG['phase1_epochs']} | "
            f"tr_loss={tr_loss:.5f}  hmap={tr_hmap:.5f}  "
            f"| val_loss={val_loss:.5f}  hmap={val_hmap:.5f}  "
            f"h_mae={val_h_mae:.2f}px  center={val_center:.2f}px"
        )

        if val_loss < best_p1_loss:
            best_p1_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, CONFIG['phase1_save_path'])
            print(f"  --> Phase 1 best saved  ({CONFIG['phase1_save_path']})")

        if early_stop.step(val_loss):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    print(f"Phase 1 complete. Best val loss: {best_p1_loss:.5f}\n")

    print("Loading best Phase 1 weights before Phase 2...")
    phase1_ckpt = torch.load(CONFIG['phase1_save_path'], map_location=device)
    model.load_state_dict(phase1_ckpt['model_state_dict'])

    # ==================================================================
    # Phase 2: Unfreeze everything, differential LRs
    # ==================================================================
    print("--- Phase 2: full fine-tune ---")

    for param in model.parameters():
        param.requires_grad = True

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Phase 2): {trainable_p2:,}")

    encoder_decoder_params = (
        list(model.encoder.parameters()) +
        list(model.decoder_blocks.parameters()) +
        list(model.final_conv.parameters())
    )
    head_params = list(model.height_head.parameters())

    optimizer_p2 = torch.optim.AdamW([
        {'params': encoder_decoder_params, 'lr': CONFIG['phase2_lr_encoder']},
        {'params': head_params,            'lr': CONFIG['phase2_lr_head']},
    ])

    early_stop_p2 = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    best_total_loss = float('inf')

    for epoch in range(1, CONFIG['phase2_epochs'] + 1):
        tr_loss, tr_hmap, tr_h = train_one_epoch(model, train_loader, optimizer_p2,
                                                  criterion, device)
        val_loss, val_hmap, val_h_mae, val_center = validate(model, val_loader,
                                                              criterion, device)

        print(
            f"P2 Ep {epoch:3d}/{CONFIG['phase2_epochs']} | "
            f"tr_loss={tr_loss:.5f}  hmap={tr_hmap:.5f}  "
            f"| val_loss={val_loss:.5f}  hmap={val_hmap:.5f}  "
            f"h_mae={val_h_mae:.2f}px  center={val_center:.2f}px"
        )

        if val_loss < best_total_loss:
            best_total_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_center_error_px': val_center,
                'val_height_mae_px': val_h_mae,
            }, CONFIG['save_path'])
            print(f"  --> Best model saved  ({CONFIG['save_path']})")

        if early_stop_p2.step(val_loss):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best val loss: {best_total_loss:.5f}")
    print(f"Final model: {CONFIG['save_path']}")


if __name__ == '__main__':
    main()
