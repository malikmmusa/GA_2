import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path if running directly
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.retfound_unet import RETFound_UNet, get_coordinates_from_heatmap

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 224,
    'model_path': 'OCT_Pipeline_2026/models/best_disc_model.pth',
    'data_dir': 'OCT_Pipeline_2026/data/processed/en_face',
    'csv_file': 'OCT_Pipeline_2026/data/csv/disc_labels.csv',
    'output_dir': 'OCT_Pipeline_2026/data/inference_results/disc_test',
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
}

def load_model(path, device):
    print(f"Loading model from {path}...")
    model = RETFound_UNet(img_size=CONFIG['img_size'], weights_path=None, freeze_encoder=False)
    
    # Load state dict
    try:
        state_dict = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image.shape[:2]
    
    transform = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0) # (1, 3, H, W) 
    
    return tensor, image, (h_orig, w_orig)

def visualize_result(filename, orig_img, heatmap, pred_xy, true_xy, save_dir):
    """
    Saves a composite image:
    1. Original Image with Pred (Green) and GT (Red) markers
    2. Predicted Heatmap
    """
    pred_x, pred_y = pred_xy
    true_x, true_y = true_xy
    
    # Resize heatmap to original image size for display
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Draw markers on original image
    vis_img = orig_img.copy()
    
    # Ground Truth: Red Cross
    cv2.drawMarker(vis_img, (int(true_x), int(true_y)), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    
    # Prediction: Green Circle
    cv2.circle(vis_img, (int(pred_x), int(pred_y)), 10, (0, 255, 0), 2)
    cv2.circle(vis_img, (int(pred_x), int(pred_y)), 2, (0, 255, 0), -1)
    
    # Text
    err = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
    cv2.putText(vis_img, f"Err: {err:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine
    combined = np.hstack([vis_img, heatmap_colored])
    
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"res_{filename}"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def main():
    # 1. Setup
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 2. Load Data (Validation Set only)
    df = pd.read_csv(CONFIG['csv_file'])
    # Re-create the same split logic as training to ensure we test on validation data
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    print(f"Testing on {len(val_df)} validation images.")
    
    # 3. Load Model
    model = load_model(CONFIG['model_path'], device)
    if model is None:
        return

    errors = []
    
    print("\nRunning Inference...")
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        filename = row['filename']
        true_x, true_y = row['disc_x'], row['disc_y']
        
        # Load
        img_path = os.path.join(CONFIG['data_dir'], filename)
        input_tensor, orig_img, (h_orig, w_orig) = preprocess_image(img_path)
        
        if input_tensor is None:
            print(f"Skipping {filename} (not found)")
            continue
            
        # Predict
        with torch.no_grad():
            output = model(input_tensor.to(device))
            heatmap = output.cpu().squeeze().numpy()
            
        # Extract Coords (in 224x224 space)
        px, py = get_coordinates_from_heatmap(heatmap)
        
        # Scale back to original size
        scale_x = w_orig / CONFIG['img_size']
        scale_y = h_orig / CONFIG['img_size']
        
        pred_x_orig = px * scale_x
        pred_y_orig = py * scale_y
        
        # Metric
        dist = np.sqrt((pred_x_orig - true_x)**2 + (pred_y_orig - true_y)**2)
        errors.append(dist)
        
        # Visualize
        visualize_result(filename, orig_img, heatmap, (pred_x_orig, pred_y_orig), (true_x, true_y), CONFIG['output_dir'])
        
    # Summary
    errors = np.array(errors)
    print("\n=== Test Results ===")
    print(f"Images Tested: {len(errors)}")
    print(f"Mean Error:   {np.mean(errors):.2f} px")
    print(f"Median Error: {np.median(errors):.2f} px")
    print(f"Std Dev:      {np.std(errors):.2f} px")
    print(f"Min Error:    {np.min(errors):.2f} px")
    print(f"Max Error:    {np.max(errors):.2f} px")
    print(f"\nVisual results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
