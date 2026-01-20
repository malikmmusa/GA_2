import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.image_utils import get_split_indices_and_images

def main():
    base_dir = Path("OCT_Pipeline_2026")
    raw_dir = base_dir / "data/raw"
    labels_path = base_dir / "data/csv/disc_labels.csv"
    
    # Output paths
    output_b_scan_dir = base_dir / "data/processed/b_scans"
    output_en_face_dir = base_dir / "data/processed/en_face"
    output_labels_path = base_dir / "data/csv/disc_labels_v2.csv"
    
    output_b_scan_dir.mkdir(parents=True, exist_ok=True)
    output_en_face_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original labels
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labels from {labels_path}")
    
    # Map filenames (stem -> full filename) for easier lookup
    # Labels usually have .png, raw might be .jpg
    label_map = {Path(f).stem: row for idx, row in df.iterrows() for f in [row['filename']]}
    
    new_rows = []
    
    image_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']])
    print(f"Found {len(image_files)} raw images.")
    
    for img_path in tqdm(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        stem = img_path.stem
        
        # 1. Simulate OLD Split (to get the coordinate origin of current labels)
        # The old script used -150 margin
        _, _, meta_old = get_split_indices_and_images(img, divider_safety_margin=-150)
        old_start_col = meta_old['final_split_column']
        
        # 2. Perform NEW Split (Clean, 0 margin)
        # We use a small positive margin to be safe from divider edges? 
        # Actually 0 is fine if the detector is good, or 10 to be safe.
        # Let's use 10 to ensure no divider pixels.
        b_scan, en_face, meta_new = get_split_indices_and_images(img, divider_safety_margin=10)
        new_start_col = meta_new['final_split_column']
        
        # 3. Save New Images
        # Maintain original filename extension or force PNG? 
        # Labels have .png, so let's save as .png to match label filenames if possible,
        # or just use the raw filename and update csv.
        # The existing labels seem to expect .png
        out_filename = f"{stem}.png" # Force PNG for processed data
        
        cv2.imwrite(str(output_b_scan_dir / out_filename), b_scan)
        cv2.imwrite(str(output_en_face_dir / out_filename), en_face)
        
        # 4. Calculate Label Offset
        # Check if we have a label for this image
        if stem in label_map:
            row = label_map[stem]
            original_x = row['disc_x']
            original_y = row['disc_y']
            
            # X_abs = X_old_rel + Old_Start
            abs_x = original_x + old_start_col
            
            # X_new_rel = X_abs - New_Start
            new_x = abs_x - new_start_col
            
            # Y should be unaffected unless we cropped top/bottom (we didn't)
            new_y = original_y 
            
            # Sanity Check
            height, width = en_face.shape[:2]
            if not (0 <= new_x < width):
                print(f"Warning: {stem} coord {new_x} out of bounds (width {width}). Old: {original_x}, Abs: {abs_x}, Start: {new_start_col}")
                # Use a clamp? Or discard?
                # If it's slightly out, clamp.
                new_x = max(0, min(new_x, width - 1))
            
            new_rows.append({
                'filename': out_filename,
                'disc_x': new_x,
                'disc_y': new_y,
                'disc_height': row['disc_height']
            })
            
    # Save new CSV
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_labels_path, index=False)
    print(f"\nSaved {len(new_rows)} realigned labels to {output_labels_path}")
    print("Images regenerated in data/processed/")

if __name__ == "__main__":
    main()
