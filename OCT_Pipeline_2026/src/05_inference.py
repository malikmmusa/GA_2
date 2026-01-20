"""
Stage 1 Only: Optic Disc Inference Script (Full Composite Visualization)
Detects the optic disc and draws a vertical reference line on the full composite image.
Now supports height prediction to limit the line to the disc boundaries.
Includes intelligent handling of single En-Face images (square aspect ratio).

Usage:
    python src/05_inference.py <path_to_composite_image>
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.models.resnet import DiscDetectorCNN
from src.utils.image_utils import get_split_indices_and_images

# --- Inference Logic ---

def preprocess_for_disc(image, output_size=(224, 224)):
    original_size = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0), original_size

def run_stage1(image_path, model_path, output_dir, device):
    img = cv2.imread(str(image_path))
    if img is None: raise ValueError(f"Load error: {image_path}")
    
    print(f"Processing Stage 1 for: {image_path}")
    
    # Use the centralized splitting logic with the SAME margin as training (margin=10)
    # The 'realign_and_split.py' script used margin=10 for the new dataset.
    b_scan, en_face, metadata = get_split_indices_and_images(img, divider_safety_margin=10)
    
    # The training logic uses the 'en_face' image directly.
    # The 'metadata' tells us where 'en_face' starts relative to the original image.
    # final_split_column is the column index in 'img' corresponding to column 0 of 'en_face'.
    start_col = metadata['final_split_column']
    
    print(f"  En-face starts at original column {start_col}")
    
    # Debug: Save the image being fed to the model
    debug_dir = Path(output_dir) / "debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"debug_input_{Path(image_path).name}"), en_face)

    model = DiscDetectorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    tensor, (orig_h, orig_w) = preprocess_for_disc(en_face)
    with torch.no_grad():
        coords = model(tensor.to(device)).squeeze().cpu().numpy()
    
    # Unpack (x, y, height)
    dx_rel = int(coords[0] * orig_w)
    dy_rel = int(coords[1] * orig_h)
    h_px = int(coords[2] * orig_h)
    
    dx_abs = start_col + dx_rel
    dy_abs = dy_rel 
    
    print(f"  Optic Disc detected at ABSOLUTE x={dx_abs}, y={dy_abs}, height={h_px}")

    # Visualization on FULL COMPOSITE
    vis = img.copy()
    h, w = vis.shape[:2]
    
    # Calculate top and bottom of the line based on predicted height
    y_top = max(0, dy_abs - h_px // 2)
    y_bottom = min(h, dy_abs + h_px // 2)
    
    # Draw thick RED vertical line restricted to disc height
    cv2.line(vis, (dx_abs, y_top), (dx_abs, y_bottom), (0, 0, 255), 10)
    
    # Draw a circle at the center
    cv2.circle(vis, (dx_abs, dy_abs), 15, (255, 0, 0), -1) # Blue center
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{Path(image_path).stem}_result.png"
    cv2.imwrite(str(out_path), vis)
    print(f"  Result saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model', default='OCT_Pipeline_2026/models/disc_detector.pth')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    run_stage1(args.image_path, args.model, 'OCT_Pipeline_2026/data/inference_results', device)
