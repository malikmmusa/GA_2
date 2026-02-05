import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.retfound_unet import RETFound_UNet, get_coordinates_from_heatmap
from utils.image_utils import get_split_indices_and_images, find_fovea_robust

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 224,
    'model_path': 'weights/best_disc_model.pth',
    'output_dir': 'output_results',
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
}

def load_model(path, device):
    print(f"Loading model from {path}...")
    model = RETFound_UNet(img_size=CONFIG['img_size'], weights_path=None, freeze_encoder=False)
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_on_image(image_path, model, device):
    # 1. Load and Split
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Split using the standard logic
    _, en_face, metadata = get_split_indices_and_images(img, divider_safety_margin=10)
    
    en_face_rgb = cv2.cvtColor(en_face, cv2.COLOR_BGR2RGB)
    h_ef, w_ef = en_face_rgb.shape[:2]
    
    # 2. Preprocess
    transform = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=en_face_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        heatmap = output.cpu().squeeze().numpy()
    
    # 4. Extract Coordinates
    # We will compute coordinates directly from the raw heatmap to avoid resizing artifacts
    
    # -------------------------------------------------------------------------
    # New Algorithm: Vertical Cup/Disc Span (Raw Heatmap Global Cluster)
    # -------------------------------------------------------------------------
    # 1. Analyze Raw Heatmap (224x224)
    hm_h, hm_w = heatmap.shape
    raw_max = heatmap.max()
    
    # 2. Global Strict Threshold (99%)
    # Use the entire high-intensity cluster, not just one column
    threshold_span = 0.99 * raw_max
    y_indices, x_indices = np.where(heatmap > threshold_span)
    
    if len(y_indices) > 0:
        # Find global Y-extents of the peak cluster
        min_y_raw = y_indices.min()
        max_y_raw = y_indices.max()
        
        # Find center X of the cluster
        cx_raw = np.mean(x_indices)
    else:
        # Fallback to argmax point if no pixels > 0.99 (rare)
        py_raw, px_raw = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        min_y_raw = py_raw
        max_y_raw = py_raw
        cx_raw = px_raw

    # 3. Project to En Face Dimensions with Sub-Pixel Correction
    # Add 0.5 to center the coordinate within the raw pixel grid
    # This shifts the top point DOWN by 0.5 raw pixels (approx 2-3 orig pixels),
    # which addresses the "top passes a little bit too much" feedback.
    
    scale_x = w_ef / CONFIG['img_size']
    scale_y = h_ef / CONFIG['img_size']
    
    pred_x_ef = (cx_raw + 0.5) * scale_x
    pred_min_y_ef = (min_y_raw + 0.5) * scale_y
    pred_max_y_ef = (max_y_raw + 0.5) * scale_y
    
    # Also calculate the single point for logging
    pred_y_ef = (pred_min_y_ef + pred_max_y_ef) / 2
    
    # 4. Project to Original Composite Size
    start_x_offset = metadata['final_split_column']
    
    orig_cx = start_x_offset + pred_x_ef
    orig_min_y = pred_min_y_ef
    orig_max_y = pred_max_y_ef
    
    # For logging purposes
    pred_x_orig = orig_cx
    pred_y_orig = (orig_min_y + orig_max_y) / 2
    
    print(f"Prediction for {Path(image_path).name}:")
    print(f"  En Face Coordinates: ({pred_x_ef:.1f}, {pred_y_ef:.1f})")
    print(f"  Original Coordinates: ({pred_x_orig:.1f}, {pred_y_orig:.1f})")
    
    # -------------------------------------------------------------------------
    # Fovea Detection (Robust)
    # -------------------------------------------------------------------------
    red_line_coords = ((orig_cx, orig_min_y), (orig_cx, orig_max_y))
    # actual_split_column from metadata is the raw split point for en-face ROI
    fovea_coords = find_fovea_robust(img, red_line_coords, metadata['actual_split_column'], image_name=Path(image_path).name)
    
    if fovea_coords:
        print(f"  Fovea Detected at: {fovea_coords}")
    else:
        print("  Fovea not detected or rejected by user.")

    # 5. Visualize
    # Create heatmap overlay for the en_face part (just for the separate heatmap file)
    heatmap_resized = cv2.resize(heatmap, (w_ef, h_ef))
    
    # -------------------------------------------------------------------------
    # Visualization: Red Line on Original Image (No Heatmap Blend)
    # -------------------------------------------------------------------------
    vis_overlay = img.copy()
    vis_overlay = cv2.cvtColor(vis_overlay, cv2.COLOR_BGR2RGB)
    
    # Draw Red Vertical Line (RGB: 255, 0, 0)
    cv2.line(vis_overlay, (int(orig_cx), int(orig_min_y)), (int(orig_cx), int(orig_max_y)), (255, 0, 0), 2)
    
    # Draw Fovea (Green Circle) if detected
    if fovea_coords:
        cv2.circle(vis_overlay, (int(fovea_coords[0]), int(fovea_coords[1])), 10, (0, 255, 0), -1)
    
    # Save results as separate images
    filename = Path(image_path).stem
    
    # 1. Save Composite with Overlay and Marker
    save_path_overlay = os.path.join(CONFIG['output_dir'], f"prediction_{filename}_overlay.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_overlay)
    plt.axis('off')
    plt.savefig(save_path_overlay, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2. Save Raw Heatmap
    save_path_hm = os.path.join(CONFIG['output_dir'], f"prediction_{filename}_heatmap.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')
    plt.savefig(save_path_hm, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"  Overlay saved to: {save_path_overlay}")
    print(f"  Heatmap saved to: {save_path_hm}")

def main():
    test_files = [
        'input_images/test_1.png',
        'input_images/test_2.png',
        'input_images/test_3.png'
    ]
    
    device = torch.device(CONFIG['device'])
    model = load_model(CONFIG['model_path'], device)
    
    for f in test_files:
        if os.path.exists(f):
            predict_on_image(f, model, device)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()