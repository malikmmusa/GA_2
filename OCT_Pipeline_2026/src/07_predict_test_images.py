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
from utils.image_utils import get_split_indices_and_images

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 224,
    'model_path': 'OCT_Pipeline_2026/models/best_disc_model.pth',
    'output_dir': 'OCT_Pipeline_2026/data/inference_results/specific_tests',
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
    px, py = get_coordinates_from_heatmap(heatmap)
    
    # Scale back to en_face size
    pred_x_ef = px * (w_ef / CONFIG['img_size'])
    pred_y_ef = py * (h_ef / CONFIG['img_size'])
    
    # Scale to original composite size
    pred_x_orig = metadata['final_split_column'] + pred_x_ef
    pred_y_orig = pred_y_ef
    
    print(f"Prediction for {Path(image_path).name}:")
    print(f"  En Face Coordinates: ({pred_x_ef:.1f}, {pred_y_ef:.1f})")
    print(f"  Original Coordinates: ({pred_x_orig:.1f}, {pred_y_orig:.1f})")
    
    # 5. Visualize
    # Create heatmap overlay for the en_face part
    heatmap_resized = cv2.resize(heatmap, (w_ef, h_ef))
    # Normalize heatmap to 0-255 for visualization
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend heatmap with en_face
    alpha = 0.5
    overlay_ef = cv2.addWeighted(en_face_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Create a blank overlay for the B-scan part (same size as B-scan)
    # We need to reconstruct the full composite logic
    # The 'metadata' tells us where en_face starts.
    # We can just overlay the heatmap onto the full original image.
    
    vis_orig = img.copy()
    vis_orig = cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = vis_orig.shape[:2]
    
    # Create a full-size heatmap canvas
    full_heatmap = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    
    # Place the colored heatmap in the correct location
    # en_face starts at: metadata['final_split_column'] - metadata['trim_left_offset'] - metadata['scrub_offset']?
    # Actually, let's look at how we split:
    # b_scan, en_face, metadata = get_split_indices_and_images(img, divider_safety_margin=10)
    # The 'en_face' image returned is processed (trimmed/scrubbed).
    # We know 'pred_x_orig' is the coordinate in the original image.
    # To visualize the heatmap overlay on the original, we need to map the en_face heatmap back to the original grid.
    
    # Start X in original image for the en_face content:
    start_x = metadata['final_split_column']
    # The en_face image width is w_ef.
    # So we place heatmap_colored at [0:h_ef, start_x : start_x + w_ef]
    
    # Handle bounds (in case of rounding/trimming edge cases)
    end_x = min(start_x + w_ef, w_orig)
    actual_w = end_x - start_x
    
    if actual_w > 0:
        full_heatmap[0:h_ef, start_x:end_x] = heatmap_colored[:, 0:actual_w]
        
    # Now blend this full_heatmap with the original image, but only where we have heatmap data
    # Create a mask where full_heatmap is not black
    mask = (full_heatmap > 0).any(axis=2).astype(np.uint8) * 255
    # Expand mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Blend only in the masked region
    # This is a bit tricky with addWeighted for just a region, so let's do pixel-wise
    # But a simpler way is to just blend the whole thing if we treat black as transparency?
    # No, jet colormap has colors everywhere.
    # Let's just assume we want to overlay on the en-face part.
    
    vis_overlay = vis_orig.copy()
    roi = vis_overlay[0:h_ef, start_x:end_x]
    hm_roi = full_heatmap[0:h_ef, start_x:end_x]
    
    blended_roi = cv2.addWeighted(roi, 1 - alpha, hm_roi, alpha, 0)
    vis_overlay[0:h_ef, start_x:end_x] = blended_roi
    
    # Draw Marker
    cv2.circle(vis_overlay, (int(pred_x_orig), int(pred_y_orig)), 15, (0, 255, 0), 2)
    cv2.circle(vis_overlay, (int(pred_x_orig), int(pred_y_orig)), 3, (0, 255, 0), -1)
    
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
        'OCT_Pipeline_2026/data/test/test_1.png',
        'OCT_Pipeline_2026/data/test/test_2.png'
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