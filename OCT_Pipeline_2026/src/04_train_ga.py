"""
Stage 3: Geographic Atrophy (GA) Segmentation (The Pathology)

Input: En Face images
Model: U-Net (Semantic Segmentation)
Challenge: Labels are sparse points, but target is a region
Strategy: Weak supervision using Gaussian blobs (σ ≈ 50px) as proxy masks
Output: Binary mask or probability map of GA lesion
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from models.unet import UNetForSegmentation
from utils.gaussian_utils import generate_gaussian_mask


class GASegmentationDataset(Dataset):
    """
    Dataset for GA segmentation using weak supervision.
    Converts point annotations to Gaussian blob masks.
    """
    def __init__(self, image_dir, labels_csv, transform=None, sigma=50, output_size=(256, 256)):
        """
        Args:
            image_dir: Directory containing en face images
            labels_csv: CSV with columns: filename, ga_x, ga_y
            transform: Image transforms
            sigma: Standard deviation for Gaussian blob (default: 50px, larger than fovea)
            output_size: Size to resize images and masks
        """
        self.image_dir = Path(image_dir)
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform
        self.sigma = sigma
        self.output_size = output_size
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_name = row['filename']
        ga_x = float(row['ga_x'])
        ga_y = float(row['ga_y'])
        
        # Load image
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Resize image
        image = image.resize(self.output_size, Image.BILINEAR)
        
        # Scale coordinates to resized image
        scale_x = self.output_size[0] / orig_width
        scale_y = self.output_size[1] / orig_height
        scaled_x = ga_x * scale_x
        scaled_y = ga_y * scale_y
        
        # Generate Gaussian blob as proxy mask (weak supervision)
        mask = generate_gaussian_mask(
            scaled_x, scaled_y,
            self.output_size[1], self.output_size[0],
            sigma=self.sigma,
            threshold=0.3  # Lower threshold for broader mask
        )
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask, img_name


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        dice: Dice coefficient
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        return 1 - dice_coefficient(pred, target, self.smooth)


class CombinedLoss(nn.Module):
    """
    Combination of BCE and Dice loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    for images, masks, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate Dice coefficient
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            running_dice += dice.item()
    
    return running_loss / len(dataloader), running_dice / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate Dice coefficient
            dice = dice_coefficient(outputs, masks)
            running_dice += dice.item()
    
    return running_loss / len(dataloader), running_dice / len(dataloader)


def train_ga_segmenter(
    train_csv,
    val_csv,
    en_face_dir,
    output_model_path,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    sigma=50
):
    """
    Train the GA segmentation model using weak supervision.
    
    Args:
        train_csv: Path to training labels CSV
        val_csv: Path to validation labels CSV
        en_face_dir: Directory containing en face images
        output_model_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        sigma: Standard deviation for Gaussian blobs (proxy masks)
    """
    # Setup device - Optimized for Apple Silicon M4 Max
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU (No GPU acceleration)")
    print(f"Device details: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = GASegmentationDataset(
        en_face_dir, train_csv, train_transform, sigma=sigma
    )
    val_dataset = GASegmentationDataset(
        en_face_dir, val_csv, val_transform, sigma=sigma
    )
    
    # Optimized for M4 Max's 16 CPU cores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = UNetForSegmentation(in_channels=1, out_channels=1, apply_sigmoid=True).to(device)
    
    # Loss and optimizer (Combined BCE + Dice loss)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_dice = 0.0
    
    print(f"\n{'='*60}")
    print("Starting Training - GA Segmentation (Weak Supervision)")
    print(f"Gaussian sigma: {sigma}px (proxy masks)")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dice = val_dice
            torch.save(model.state_dict(), output_model_path)
            print(f"✓ Model saved (Best Val Loss: {best_val_loss:.6f}, Dice: {best_dice:.4f})")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best Dice coefficient: {best_dice:.4f}")
    print(f"Model saved to: {output_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    
    # Paths
    EN_FACE_DIR = BASE_DIR / "data" / "processed" / "en_face"
    TRAIN_CSV = BASE_DIR / "data" / "csv" / "train_ga_labels.csv"
    VAL_CSV = BASE_DIR / "data" / "csv" / "val_ga_labels.csv"
    OUTPUT_MODEL = BASE_DIR / "models" / "ga_segmenter.pth"
    
    # Create models directory
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    train_ga_segmenter(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        en_face_dir=EN_FACE_DIR,
        output_model_path=OUTPUT_MODEL,
        num_epochs=20,  # Quick training for initial model
        batch_size=8,
        learning_rate=1e-4,
        sigma=50  # 50px Gaussian for GA (larger than fovea)
    )
