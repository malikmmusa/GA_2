import cv2
import numpy as np
import os
from pathlib import Path

def debug_masks(image_path, output_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range 1: Green (Hue 60-70)
    lower_green = np.array([60, 50, 50])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Range 2: Blue (Hue 100-110)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([110, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Save original and masks
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    cv2.imwrite(str(output_path / "debug_original.png"), img)
    cv2.imwrite(str(output_path / "debug_mask_green.png"), mask_green)
    cv2.imwrite(str(output_path / "debug_mask_blue.png"), mask_blue)
    
    print(f"Saved debug images to {output_dir}")
    print(f"Green pixels: {np.count_nonzero(mask_green)}")
    print(f"Blue pixels: {np.count_nonzero(mask_blue)}")

if __name__ == "__main__":
    # Pick the first image found
    img_dir = Path("OCT_Pipeline_2026/data/processed_marked/en_face")
    first_img = next(img_dir.glob("*.png"))
    debug_masks(first_img, "OCT_Pipeline_2026/data/debug_markers")
