import cv2
import pandas as pd
from pathlib import Path
import os

def visualize_labels():
    base_dir = Path("OCT_Pipeline_2026")
    image_dir = base_dir / "data/processed/en_face"
    labels_path = base_dir / "data/csv/disc_labels_aligned.csv"
    output_dir = base_dir / "data/debug_check_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not labels_path.exists():
        print("No labels file found.")
        return

    df = pd.read_csv(labels_path)
    
    print(f"Generating debug images for {len(df)} samples...")
    
    for _, row in df.iterrows():
        filename = row['filename']
        img_path = image_dir / filename
        
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        x, y = int(row['disc_x']), int(row['disc_y'])
        
        # Draw Crosshair
        cv2.line(img, (x - 20, y), (x + 20, y), (0, 255, 0), 2)
        cv2.line(img, (x, y - 20), (x, y + 20), (0, 255, 0), 2)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # Save
        cv2.imwrite(str(output_dir / f"check_{filename}"), img)

if __name__ == "__main__":
    visualize_labels()