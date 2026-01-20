import cv2
import numpy as np
import os
from pathlib import Path
from collections import Counter

def analyze_colors(directory, num_images=5):
    input_path = Path(directory)
    image_files = list(input_path.glob('*.png'))[:num_images]
    
    print(f"Analyzing {len(image_files)} images for unique marker colors...")
    
    # Known ranges to exclude (approximate)
    # Red (Fovea)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Peach/Orange/Pink (GA)
    # Broad range covering peach tones
    lower_peach = np.array([10, 50, 100])
    upper_peach = np.array([30, 255, 255]) 
    
    color_histogram = Counter()
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks for known things to ignore them
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_peach = cv2.inRange(hsv, lower_peach, upper_peach)
        
        # Mask for background (low saturation or very dark/very bright)
        # S < 30 usually background/gray
        mask_bg = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 255]))
        
        # Combine ignored masks
        mask_ignore = mask_red1 | mask_red2 | mask_peach | mask_bg
        
        # Invert to find "interesting" pixels
        mask_interesting = cv2.bitwise_not(mask_ignore)
        
        # Get interesting pixels
        interesting_pixels = hsv[mask_interesting > 0]
        
        if len(interesting_pixels) > 0:
            # Count hues
            for pixel in interesting_pixels:
                h, s, v = pixel
                # Bin hue by 10
                hue_bin = (h // 10) * 10
                color_histogram[hue_bin] += 1
                
    print("\nDominant Unique Hues (excluding Red/Peach/Grayscale):")
    for hue, count in color_histogram.most_common(5):
        print(f"Hue Bin {hue}-{hue+10}: {count} pixels")
        
    # Heuristic interpretation
    print("\nInterpretation:")
    for hue, count in color_histogram.most_common(3):
        if 35 <= hue <= 85:
            print(f"- Found Green/Cyan tones (Hue {hue}). Potential Disc marker?")
        elif 85 <= hue <= 130:
            print(f"- Found Blue/Azure tones (Hue {hue}). Potential Disc marker?")
        elif 130 <= hue <= 170:
            print(f"- Found Purple/Violet tones (Hue {hue}). Potential Disc marker?")
        else:
            print(f"- Found Hue {hue}. Unknown marker.")

if __name__ == "__main__":
    analyze_colors("OCT_Pipeline_2026/data/processed_marked/en_face")
