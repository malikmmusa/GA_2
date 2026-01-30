"""
Stage 1: Optic Disc Inference with Heatmap (RETFound)
Generates a heatmap for the optic disc and draws a vertical reference line
based on the "deepest red" (highest activation) core.

Usage:
    python src/05_inference_heatmap.py <path_to_composite_image>
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.models.retfound_unet import RETFound_UNet
from src.utils.image_utils import get_split_indices_and_images

def preprocess_for_heatmap(image, output_size=(224, 224)):
    """
    Prepares image for RETFound U-Net.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, output_size)
    
    # Normalize (standard ImageNet/RETFound mean/std)
    # Using the same normalization as in training (albumentations)
    # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    # To Tensor (C, H, W)
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor

def process_heatmap_logic(heatmap, original_h, original_w, offset_x):
    """
    Applies the "Deepest Red" logic to the heatmap.
    Returns: (start_point, end_point, center_point) in absolute coordinates.
    """
    # Resize heatmap to original en-face dimensions
    heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
    
    # 1. Threshold: Top 10% of values (Deepest Red)
    max_val = np.max(heatmap_resized)
    threshold_val = 0.9 * max_val
    _, mask = cv2.threshold(heatmap_resized, threshold_val, 1.0, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)
    
    # 2. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No deep red core detected.")
        # Fallback to simple max point
        y, x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        return (offset_x + x, y - 50), (offset_x + x, y + 50), (offset_x + x, y)

    # 3. Largest contour is the disc core
    c = max(contours, key=cv2.contourArea)
    
    # --- Refined logic: Maximum Vertical Feret Diameter ---
    # Create a clean mask for just this contour to scan column-by-column
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [c], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(c)
    
    max_span = -1
    best_x = x + w // 2 # Default to center if logic fails
    best_top = y
    best_bottom = y + h
    
    # Scan the x-coordinate where the column has the tallest vertical span
    for col_x in range(x, x + w):
        column = contour_mask[:, col_x]
        pos_pixels = np.where(column > 0)[0]
        
        if len(pos_pixels) > 0:
            top_y = pos_pixels[0]
            bottom_y = pos_pixels[-1]
            span = bottom_y - top_y
            
            if span > max_span:
                max_span = span
                best_x = col_x
                best_top = top_y
                best_bottom = bottom_y
    
    # Map to Absolute Coordinates
    cx_abs = int(best_x + offset_x)
    top_pt = (cx_abs, int(best_top))
    bottom_pt = (cx_abs, int(best_bottom))
    center_pt = (cx_abs, int((best_top + best_bottom) // 2))
    
    return top_pt, bottom_pt, center_pt

def run_inference(image_path, model_path, output_dir, device):
    img = cv2.imread(str(image_path))
    if img is None: raise ValueError(f"Load error: {image_path}")
    
    print(f"Processing (Heatmap Mode) for: {image_path}")
    
    # 1. Split Image
    b_scan, en_face, metadata = get_split_indices_and_images(img, divider_safety_margin=10)
    start_col = metadata['final_split_column']
    
    # 2. Load Model
    model = RETFound_UNet(img_size=224, weights_path=None)
    # Load state dict
    try:
        # Weights might be saved as 'model' or 'state_dict' or directly
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device).eval()
    
    # 3. Preprocess and Run
    tensor = preprocess_for_heatmap(en_face)
    with torch.no_grad():
        heatmap_tensor = model(tensor.to(device))
        heatmap = heatmap_tensor.squeeze().cpu().numpy() # (224, 224)
        
    # 4. Process Heatmap to get Lines
    h_orig, w_orig = en_face.shape[:2]
    top_pt, bottom_pt, center_pt = process_heatmap_logic(heatmap, h_orig, w_orig, start_col)
    
    print(f"  Line: {top_pt} to {bottom_pt}")
    
    # 5. Visualization
    vis = img.copy()
    
    # Draw Line (Red)
    cv2.line(vis, top_pt, bottom_pt, (0, 0, 255), 8)
    
    # Draw Center Dot (Red)
    cv2.circle(vis, center_pt, 12, (0, 0, 255), -1)
    
    # Optional: Overlay Heatmap for debugging
    # Resize heatmap to full en-face size
    hm_full = cv2.resize(heatmap, (w_orig, h_orig))
    # Normalize to 0-255
    hm_norm = (hm_full * 255).astype(np.uint8)
    # Colorize
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    
    # Blend into the En-Face part of the visualization
    roi = vis[:, start_col:start_col+w_orig]
    # Ensure sizes match (sometimes off by 1 due to rounding)
    h_roi, w_roi = roi.shape[:2]
    hm_color = cv2.resize(hm_color, (w_roi, h_roi))
    
    # Add weighted overlay
    blended = cv2.addWeighted(roi, 0.7, hm_color, 0.3, 0)
    vis[:, start_col:start_col+w_orig] = blended
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{Path(image_path).stem}_result.png"
    cv2.imwrite(str(out_path), vis)
    print(f"  Result saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model', default='OCT_Pipeline_2026/models/best_disc_model.pth')
    
    args = parser.parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    run_inference(args.image_path, args.model, 'OCT_Pipeline_2026/data/inference_results', device)
