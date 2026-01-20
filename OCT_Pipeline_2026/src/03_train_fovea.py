"""
Stage 2: Fovea Localization (The Landmark)

Input: B-Scan images (or dual-stream with En Face)
Model: U-Net with Gaussian Heatmap Regression
Output: (x, y) coordinates of the fovea
Training: Point labels → 2D Gaussian heatmaps (σ ≈ 15px)
Loss: Mean Squared Error (MSE)
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

from models.unet import UNetWithSigmoid
from utils.gaussian_utils import generate_gaussian_heatmap, heatmap_to_coordinates


class FoveaHeatmapDataset(Dataset):
    """
    Dataset for fovea detection using heatmap regression.
    Converts point annotations to Gaussian heatmaps.
    """
    def __init__(self, image_dir, labels_csv, transform=None, sigma=15, output_size=(256, 256)):
        """
        Args:
            image_dir: Directory containing B-scan images
            labels_csv: CSV with columns: filename, fovea_x, fovea_y
            transform: Image transforms
            sigma: Standard deviation for Gaussian heatmap (default: 15px)
            output_size: Size to resize images and heatmaps
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
        fovea_x = float(row['fovea_x'])
        fovea_y = float(row['fovea_y'])
        
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
        scaled_x = fovea_x * scale_x
        scaled_y = fovea_y * scale_y
        
        # Generate Gaussian heatmap target
        heatmap = generate_gaussian_heatmap(
            scaled_x, scaled_y,
            self.output_size[1], self.output_size[0],
            sigma=self.sigma
        )
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
        
        return image, heatmap, img_name


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    for images, heatmaps, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # MSE loss between predicted and target heatmaps
        loss = criterion(outputs, heatmaps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model and calculate coordinate error.
    """
    model.eval()
    running_loss = 0.0
    total_pixel_error = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, heatmaps, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            running_loss += loss.item()
            
            # Calculate pixel-wise localization error
            batch_size = images.size(0)
            for i in range(batch_size):
                pred_heatmap = outputs[i, 0].cpu().numpy()
                true_heatmap = heatmaps[i, 0].cpu().numpy()
                
                pred_x, pred_y = heatmap_to_coordinates(pred_heatmap)
                true_x, true_y = heatmap_to_coordinates(true_heatmap)
                
                error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                total_pixel_error += error
                num_samples += 1
    
    avg_loss = running_loss / len(dataloader)
    avg_pixel_error = total_pixel_error / num_samples if num_samples > 0 else 0
    
    return avg_loss, avg_pixel_error


def train_fovea_detector(
    train_csv,
    val_csv,
    b_scan_dir,
    output_model_path,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    sigma=15
):
    """
    Train the fovea detector with heatmap regression.
    
    Args:
        train_csv: Path to training labels CSV
        val_csv: Path to validation labels CSV
        b_scan_dir: Directory containing B-scan images
        output_model_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        sigma: Standard deviation for Gaussian heatmaps
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
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = FoveaHeatmapDataset(
        b_scan_dir, train_csv, train_transform, sigma=sigma
    )
    val_dataset = FoveaHeatmapDataset(
        b_scan_dir, val_csv, val_transform, sigma=sigma
    )
    
    # Optimized for M4 Max's 16 CPU cores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = UNetWithSigmoid(in_channels=1, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    
    print(f"\n{'='*60}")
    print("Starting Training - Fovea Localization (Heatmap Regression)")
    print(f"Gaussian sigma: {sigma}px")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pixel_error = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Pixel Error: {val_pixel_error:.2f}px")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = val_pixel_error
            torch.save(model.state_dict(), output_model_path)
            print(f"✓ Model saved (Best Val Loss: {best_val_loss:.6f}, Pixel Error: {best_pixel_error:.2f}px)")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best pixel error: {best_pixel_error:.2f}px")
    print(f"Model saved to: {output_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    
    # Paths
    B_SCAN_DIR = BASE_DIR / "data" / "processed" / "b_scans"
    TRAIN_CSV = BASE_DIR / "data" / "csv" / "train_fovea_labels.csv"
    VAL_CSV = BASE_DIR / "data" / "csv" / "val_fovea_labels.csv"
    OUTPUT_MODEL = BASE_DIR / "models" / "fovea_detector.pth"
    
    # Create models directory
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    train_fovea_detector(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        b_scan_dir=B_SCAN_DIR,
        output_model_path=OUTPUT_MODEL,
        num_epochs=20,  # Quick training for initial model
        batch_size=8,
        learning_rate=1e-4,
        sigma=15  # 15px Gaussian for fovea
    )
