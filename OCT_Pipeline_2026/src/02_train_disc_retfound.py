import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import model
from models.retfound_unet import RETFound_UNet, HeatmapGenerator, get_coordinates_from_heatmap

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 224,
    'batch_size': 8,
    'learning_rate_frozen': 1e-4,
    'learning_rate_unfrozen': 1e-5,
    'epochs_frozen': 50,
    'epochs_unfrozen': 20,
    'sigma': 20,  # Heatmap spread
    'data_dir': 'OCT_Pipeline_2026/data/processed/en_face',
    'csv_file': 'OCT_Pipeline_2026/data/csv/disc_labels.csv',
    'weights_path': 'OCT_Pipeline_2026/weights/retfound_cfp_weights.pth',
    'save_dir': 'OCT_Pipeline_2026/models',
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
}

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class OCTDiscDataset(Dataset):
    def __init__(self, dataframe, img_dir, img_size=224, sigma=20, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.img_size = img_size
        self.sigma = sigma
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Load Image
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        
        # Original Coordinates
        x_orig, y_orig = row['disc_x'], row['disc_y']
        
        # Augmentation (which handles resizing and keypoint transformation)
        if self.transform:
            transformed = self.transform(image=image, keypoints=[(x_orig, y_orig)])
            image = transformed['image']
            keypoints = transformed['keypoints']
            
            # Check if keypoint was removed (outside bounds)
            if not keypoints:
                # Fallback: if point is outside, put it on edge or mask it?
                # For now, let's keep it at 0,0 (masked) or center
                # Better: use original scaled
                x_trans, y_trans = x_orig * (self.img_size/w_orig), y_orig * (self.img_size/h_orig)
            else:
                x_trans, y_trans = keypoints[0]
        else:
            # Manual resize if no transform provided
            image = cv2.resize(image, (self.img_size, self.img_size))
            x_trans = x_orig * (self.img_size / w_orig)
            y_trans = y_orig * (self.img_size / h_orig)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        # Generate Heatmap Target
        heatmap = HeatmapGenerator(x_trans, y_trans, self.img_size, self.img_size, sigma=self.sigma)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float() # (1, H, W)
        
        return {
            'image': image,
            'heatmap': heatmap,
            'coords': torch.tensor([x_trans, y_trans], dtype=torch.float32),
            'orig_coords': torch.tensor([x_orig, y_orig], dtype=torch.float32),
            'filename': filename,
            'scale': torch.tensor([w_orig/self.img_size, h_orig/self.img_size], dtype=torch.float32)
        }

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in loader:
        images = batch['image'].to(device)      # (B, 3, H, W)
        targets = batch['heatmap'].to(device)   # (B, 1, H, W)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_error = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['heatmap'].to(device)
            true_coords = batch['coords'].numpy() # (B, 2)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Calculate pixel error
            preds_np = outputs.cpu().numpy().squeeze(1) # (B, H, W)
            for i in range(preds_np.shape[0]):
                pred_x, pred_y = get_coordinates_from_heatmap(preds_np[i])
                tx, ty = true_coords[i]
                dist = np.sqrt((pred_x - tx)**2 + (pred_y - ty)**2)
                total_error += dist
                count += 1
                
    return running_loss / len(loader), total_error / count

def save_debug_images(model, loader, device, epoch, save_dir):
    """Saves a visualization of the first batch"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'debug'), exist_ok=True)
    
    batch = next(iter(loader))
    images = batch['image'].to(device)
    targets = batch['heatmap'].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        
    # Plot first image in batch
    img = images[0].cpu().permute(1, 2, 0).numpy()
    tgt = targets[0, 0].cpu().numpy()
    pred = outputs[0, 0].cpu().numpy()
    
    # Normalize img for display
    img = (img - img.min()) / (img.max() - img.min())
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title(f"Input (Epoch {epoch})")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Target Heatmap")
    plt.imshow(tgt, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Heatmap")
    plt.imshow(pred, cmap='jet')
    plt.axis('off')
    
    plt.savefig(os.path.join(save_dir, 'debug', f'epoch_{epoch:03d}.png'))
    plt.close()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print(f"=== Starting RETFound Disc Detection Training on {CONFIG['device']} ===")
    
    # 1. Prepare Data
    if not os.path.exists(CONFIG['csv_file']):
        print(f"Error: Label file not found at {CONFIG['csv_file']}")
        return
        
    df = pd.read_csv(CONFIG['csv_file'])
    
    # Simple train/val split (80/20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Augmentations
    train_transform = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    val_transform = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    train_dataset = OCTDiscDataset(train_df, CONFIG['data_dir'], CONFIG['img_size'], CONFIG['sigma'], train_transform)
    val_dataset = OCTDiscDataset(val_df, CONFIG['data_dir'], CONFIG['img_size'], CONFIG['sigma'], val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. Initialize Model
    print("Initializing RETFound U-Net...")
    # Check for weights
    weights = CONFIG['weights_path'] if os.path.exists(CONFIG['weights_path']) else None
    if not weights:
        print("Warning: RETFound weights not found. Using random initialization for encoder.")
    
    model = RETFound_UNet(img_size=CONFIG['img_size'], weights_path=weights, freeze_encoder=True)
    model.to(CONFIG['device'])
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['learning_rate_frozen'])
    
    # 3. Training Loop (Stage 1: Frozen Encoder)
    print("\n--- Stage 1: Training Decoder (Encoder Frozen) ---")
    best_error = float('inf')
    
    for epoch in range(1, CONFIG['epochs_frozen'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_error = validate(model, val_loader, criterion, CONFIG['device'])
        
        print(f"Epoch {epoch}/{CONFIG['epochs_frozen']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Error: {val_error:.2f} px")
        
        if val_error < best_error:
            best_error = val_error
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_disc_model.pth'))
            print("  --> New Best Model Saved!")
            
        if epoch % 10 == 0:
            save_debug_images(model, val_loader, CONFIG['device'], epoch, CONFIG['save_dir'])

    # 4. Training Loop (Stage 2: Unfrozen Encoder)
    print("\n--- Stage 2: Fine-Tuning Encoder (Unfrozen) ---")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate_unfrozen']) # New optimizer for all params
    
    total_epochs = CONFIG['epochs_frozen'] + CONFIG['epochs_unfrozen']
    for epoch in range(CONFIG['epochs_frozen'] + 1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_error = validate(model, val_loader, criterion, CONFIG['device'])
        
        print(f"Epoch {epoch}/{total_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Error: {val_error:.2f} px")
        
        if val_error < best_error:
            best_error = val_error
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_disc_model.pth'))
            print("  --> New Best Model Saved!")
            
        if epoch % 10 == 0:
            save_debug_images(model, val_loader, CONFIG['device'], epoch, CONFIG['save_dir'])
            
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
