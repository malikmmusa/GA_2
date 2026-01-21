
import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.image_utils import get_split_indices_and_images

def process_heatmap_visualization(original_path, heatmap_path, output_path):
    # 1. Load Images
    original = cv2.imread(original_path)
    heatmap = cv2.imread(heatmap_path)
    
    if original is None:
        raise ValueError(f"Could not load original image: {original_path}")
    if heatmap is None:
        raise ValueError(f"Could not load heatmap image: {heatmap_path}")

    # 2. Get En-Face Offset
    # We assume the heatmap corresponds to the En-Face portion of the image
    b_scan, en_face, metadata = get_split_indices_and_images(original, divider_safety_margin=10)
    offset_x = metadata['final_split_column']
    
    # 3. Resize heatmap to match En-Face portion
    if en_face.shape[:2] != heatmap.shape[:2]:
        print(f"Resizing heatmap {heatmap.shape} to match En-Face {en_face.shape}...")
        heatmap = cv2.resize(heatmap, (en_face.shape[1], en_face.shape[0]))

    # 4. Detect "Deepest Red" Core
    # Calculate Redness Score: R - (G + B)/2
    # This emphasizes pure red and suppresses white/pink/orange
    heatmap_float = heatmap.astype(np.float32)
    B, G, R = cv2.split(heatmap_float)
    redness_score = R - (G + B) / 2.0
    redness_score = np.maximum(redness_score, 0) # Clip negatives
    
    # Find max redness
    max_score = np.max(redness_score)
    print(f"Max Redness Score: {max_score}")
    
    # Adaptive Threshold: Top 10% of redness (Deepest Red)
    # Adjust this factor (0.9) if it's still too long or too short
    threshold_value = 0.9 * max_score
    _, mask = cv2.threshold(redness_score, threshold_value, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    # Find contours on the tight mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No deep red core detected in {heatmap_path}")
        # Fallback to broader red check if deep red fails?
        # For now, just save original
        cv2.imwrite(output_path, original)
        return

    # Assume largest deep red contour is the disc core
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Calculate Centroid relative to En-Face
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx_rel = int(M["m10"] / M["m00"])
        cy_rel = int(M["m01"] / M["m00"])
    else:
        cx_rel = x + w // 2
        cy_rel = y + h // 2
        
    # 5. Map to Global Coordinates
    cx_abs = cx_rel + offset_x
    cy_abs = cy_rel
    
    print(f"Detected Red Blob (Rel): ({cx_rel}, {cy_rel}), H={h}")
    print(f"Global Coordinates: X={cx_abs}, Y={cy_abs}")

    # 6. Draw Line on Original Image
    vis = original.copy()
    
    # Shrink line by 15% (7.5% from each end) to stay inside the disc
    shrink_factor = 0.15
    shrink_px = int(h * shrink_factor / 2)
    
    # Top and Bottom of the bounding box (modified)
    top_y = y + shrink_px
    bottom_y = y + h - shrink_px
    
    top_pt = (cx_abs, top_y)
    bottom_pt = (cx_abs, bottom_y)
    
    # Draw RED line (BGR: 0, 0, 255)
    cv2.line(vis, top_pt, bottom_pt, (0, 0, 255), 8)
    
    # Draw RED center dot
    cv2.circle(vis, (cx_abs, cy_abs), 12, (0, 0, 255), -1)

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('original_image', help='Path to original composite image')
    parser.add_argument('heatmap_image', help='Path to corresponding heatmap image')
    parser.add_argument('output_image', help='Path to save the result')
    
    args = parser.parse_args()
    
    process_heatmap_visualization(args.original_image, args.heatmap_image, args.output_image)
