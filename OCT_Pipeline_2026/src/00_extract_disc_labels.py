"""
Extract Optic Disc Coordinates from Processed Marked En-Face Images

This script processes marked en-face OCT images to extract the optic disc location
indicated by a vertical line (approx color #F5C4AB).
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

def get_marker_mask(image_hsv):
    # Peach/Orange range
    lower_peach = np.array([0, 30, 180])   
    upper_peach = np.array([25, 130, 255])
    mask = cv2.inRange(image_hsv, lower_peach, upper_peach)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def process_image(image_path, debug_dir=None):
    img = cv2.imread(str(image_path))
    if img is None: return None, "Load Error"
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_marker_mask(hsv)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, "No Marker Found"
    
    # Find vertical contours (h > w)
    vertical_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        # Disc marker should be vertical (h > w) and at least 50px tall
        if aspect_ratio > 1.5 and h > 50:
            vertical_candidates.append(cnt)
            
    if not vertical_candidates:
        # Fallback to largest if no strong vertical candidates found
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        # Take the largest vertical one
        largest_contour = max(vertical_candidates, key=cv2.contourArea)
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w//2, y + h//2
        
    # Inpaint the marker to create a clean training image
    # Dilate mask slightly to ensure full coverage of the marker edges
    dilate_kernel = np.ones((5,5), np.uint8)
    inpainting_mask = cv2.dilate(mask, dilate_kernel, iterations=2)
    inpainted_img = cv2.inpaint(img, inpainting_mask, 3, cv2.INPAINT_TELEA)
    
    # Save inpainted image
    inpainted_dir = Path("OCT_Pipeline_2026/data/processed_inpainted/en_face")
    inpainted_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(inpainted_dir / image_path.name), inpainted_img)
    
    # FIX: Force a standard anatomical height (150px) instead of using the marker line's height
    # The marker line is often an arbitrary stroke by the clinician.
    # We want the model to learn the disc location, not the stroke length.
    STANDARD_DISC_HEIGHT = 150
    result = {'filename': image_path.name, 'disc_x': cx, 'disc_y': cy, 'disc_height': STANDARD_DISC_HEIGHT}
    
    if debug_dir:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (cx, cy), 10, (255, 0, 0), -1)
        out_path = debug_dir / f"debug_{image_path.name}"
        cv2.imwrite(str(out_path), debug_img)
        
    return result, None

def main():
    base_dir = Path("OCT_Pipeline_2026")
    input_dir = base_dir / "data/processed_marked/en_face"
    output_csv = base_dir / "data/csv/disc_labels.csv"
    debug_dir = base_dir / "data/debug_disc_labels"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure inpainted directory exists
    (base_dir / "data/processed_inpainted/en_face").mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob("*.png"))
    results, failed = [], []
    for img_path in tqdm(image_files):
        res, err = process_image(img_path, debug_dir)
        if res: results.append(res)
        else: failed.append((img_path.name, err))
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved {len(results)} labels to {output_csv}")
    if failed:
        print(f"\n⚠️  {len(failed)} images failed")

if __name__ == "__main__":
    main()