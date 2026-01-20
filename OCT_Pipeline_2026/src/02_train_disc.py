"""
Stage 1: Optic Disc Detection Training (Optimized)

Features:
- 5-Fold Cross Validation
- Predicts (x, y, height) for precise line drawing
- SmoothL1Loss
- ResNet18 backbone
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
import os
import shutil
import sys

# Add parent directory to path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))
from src.models.resnet import DiscDetectorCNN

class CoordinateAwareDataset(Dataset):
    """
    Dataset that handles image-coordinate-height tuples with augmentation.
    """
    def __init__(self, image_dir, labels_df, size=(224, 224), augment=False):
        self.image_dir = Path(image_dir)
        self.labels = labels_df
        self.size = size
        self.augment = augment
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = self.image_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        
        orig_w, orig_h = image.size
        
        # The labels (from src/00_extract_disc_labels.py) are the centroids of the manually drawn markers.
        # These markers are drawn ON the optic disc.
        # Therefore, we use the coordinates directly.
        x_px = float(row['disc_x'])
        
        # Debug sanity check (print once per epoch or for first item)
        # if idx == 0:
        #    print(f"DEBUG: Loading {row['filename']} - Target x: {x_px}, y: {row['disc_y']}")
        
        # Normalized coordinates [0, 1]
        x = x_px / orig_w
        y = row['disc_y'] / orig_h
        h = row['disc_height'] / orig_h
        
        # Resize image
        image = TF.resize(image, self.size)
        
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                x = 1.0 - x
            
            # Random Rotation (subtle)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
            
            # Color Jitter
            if random.random() > 0.5:
                image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Target: [x, y, height]
        targets = torch.tensor([x, y, h], dtype=torch.float32)
        return image, targets, row['filename']

def calculate_metrics(pred, target, size=(224, 224)):
    """
    Calculate Euclidean distance error for center (x,y) and absolute error for height.
    """
    # Scale back to pixels (approximate, using resize dimensions)
    pred_px = pred * torch.tensor([size[0], size[1], size[1]], device=pred.device)
    target_px = target * torch.tensor([size[0], size[1], size[1]], device=target.device)
    
    # Center Error (x, y)
    center_error = torch.sqrt(torch.sum((pred_px[:, :2] - target_px[:, :2])**2, dim=1))
    
    # Height Error (h)
    height_error = torch.abs(pred_px[:, 2] - target_px[:, 2])
    
    return center_error.mean().item(), height_error.mean().item()

def train_one_fold(fold, train_loader, val_loader, device, num_epochs=50):
    model = DiscDetectorCNN(weights='IMAGENET1K_V1').to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    fold_model_path = f"OCT_Pipeline_2026/models/disc_detector_fold{fold}.pth"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
    
    # Progress bar for epochs
    pbar = tqdm(range(num_epochs), desc=f"Fold {fold}", unit="epoch")
    for epoch in pbar:
        model.train()
        train_loss = 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        val_center_error = 0
        val_height_error = 0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                c_err, h_err = calculate_metrics(outputs, targets)
                val_center_error += c_err
                val_height_error += h_err
        
        val_loss /= len(val_loader)
        val_center_error /= len(val_loader)
        val_height_error /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), fold_model_path)
            
        pbar.set_postfix({
            "Loss": f"{val_loss:.4f}", 
            "Err(xy)": f"{val_center_error:.1f}px",
            "Err(h)": f"{val_height_error:.1f}px"
        })
            
    print(f"Fold {fold} Complete. Best Loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50) # Increased default
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Labels
    labels_path = "OCT_Pipeline_2026/data/csv/disc_labels_v2.csv"
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found. Run realign_and_split.py first.")
        return
    
    df = pd.read_csv(labels_path)
    # Use the clean images now that labels are aligned
    image_dir = "OCT_Pipeline_2026/data/processed/en_face"
    
    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"Starting 5-Fold Cross Validation on {len(df)} images for {args.epochs} epochs...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ds = CoordinateAwareDataset(image_dir, train_df, augment=True)
        val_ds = CoordinateAwareDataset(image_dir, val_df, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        val_loss = train_one_fold(fold + 1, train_loader, val_loader, device, num_epochs=args.epochs)
        fold_results.append(val_loss)
        
    print(f"\nFinal K-Fold Mean Validation Loss: {np.mean(fold_results):.4f}")
    
    # Save final ensemble model (picking the best fold)
    best_fold = np.argmin(fold_results) + 1
    src_model = f"OCT_Pipeline_2026/models/disc_detector_fold{best_fold}.pth"
    dst_model = "OCT_Pipeline_2026/models/disc_detector.pth"
    shutil.copy(src_model, dst_model)
    print(f"Final model selected from Fold {best_fold} and saved to {dst_model}")

if __name__ == "__main__":
    main()
