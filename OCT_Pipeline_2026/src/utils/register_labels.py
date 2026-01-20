"""
Register Inpainted Images to Clean Processed Images
to transfer labels from 'processed_marked' space to 'processed' space.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

def register_images(inpainted_path, clean_path):
    img_inpainted = cv2.imread(str(inpainted_path))
    img_clean = cv2.imread(str(clean_path))
    
    if img_inpainted is None or img_clean is None:
        return None, None
        
    # Convert to grayscale
    gray_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    
    # We assume one is a sub-region of the other or they overlap significantly.
    # Since we don't know which is larger, we try both ways or pad.
    # But typically 'clean' (from split_data) might be wider.
    
    # Template Matching
    # We try to find 'inpainted' inside 'clean'
    # If inpainted is larger in any dimension, this fails.
    
    h_i, w_i = gray_inpainted.shape
    h_c, w_c = gray_clean.shape
    
    if h_i > h_c or w_i > w_c:
        # If inpainted is larger, we swap, but then the offset is negative
        # But for now let's assume clean is larger or similar.
        # If inpainted is larger, we might need to crop it or search clean inside inpainted.
        # Let's search clean inside inpainted
        res = cv2.matchTemplate(gray_inpainted, gray_clean, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        # clean is at top_left inside inpainted.
        # So Clean(0,0) = Inpainted(x,y).
        # We want Inpainted(cx, cy) -> Clean(cx', cy').
        # cx' = cx - top_left_x
        # cy' = cy - top_left_y
        offset_x = -top_left[0]
        offset_y = -top_left[1]
    else:
        # Search inpainted inside clean
        res = cv2.matchTemplate(gray_clean, gray_inpainted, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        # Inpainted is at top_left inside clean.
        # Inpainted(0,0) = Clean(x, y).
        # cx' = cx + top_left_x
        offset_x = top_left[0]
        offset_y = top_left[1]
        
    return offset_x, offset_y

def main():
    base_dir = Path("OCT_Pipeline_2026")
    inpainted_dir = base_dir / "data/processed_inpainted/en_face"
    clean_dir = base_dir / "data/processed/en_face"
    labels_csv = base_dir / "data/csv/disc_labels.csv"
    output_csv = base_dir / "data/csv/disc_labels_aligned.csv"
    
    if not labels_csv.exists():
        print("Labels CSV not found")
        return
        
    df = pd.read_csv(labels_csv)
    new_rows = []
    
    print(f"Aligning {len(df)} labels...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        inpainted_path = inpainted_dir / filename
        clean_path = clean_dir / filename
        
        if not inpainted_path.exists(): 
            print(f"Missing inpainted: {filename}")
            continue
        if not clean_path.exists():
            print(f"Missing clean: {filename}")
            continue
            
        dx, dy = register_images(inpainted_path, clean_path)
        
        if dx is not None:
            new_x = row['disc_x'] + dx
            new_y = row['disc_y'] + dy
            
            # Check if within bounds of clean image
            clean_img = cv2.imread(str(clean_path))
            h, w = clean_img.shape[:2]
            
            if 0 <= new_x < w and 0 <= new_y < h:
                new_row = row.copy()
                new_row['disc_x'] = new_x
                new_row['disc_y'] = new_y
                new_rows.append(new_row)
            else:
                print(f"WARN: {filename} aligned coords ({new_x}, {new_y}) out of bounds ({w}, {h})")
        else:
            print(f"Failed to register {filename}")
            
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    print(f"\nSaved aligned labels to {output_csv}")

if __name__ == "__main__":
    main()
