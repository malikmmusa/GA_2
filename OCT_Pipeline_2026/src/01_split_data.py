"""
Stage 0: Data Splitting Script
Split composite OCT images into B-Scans (left) and En Face (right) images.

Input: Composite JPG images in data/raw/
Output: 
    - B-Scans saved to data/processed/b_scans/
    - En Face images saved to data/processed/en_face/

Critical: Maintains filename consistency across both output folders.
"""

import cv2
import os
from pathlib import Path
import numpy as np
import csv
import sys

# Add parent directory to path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.image_utils import get_split_indices_and_images

# SAFETY MARGIN: Buffer zone around dividers
# Set to 10 to ensure we skip the divider edge without losing anatomical features
DIVIDER_SAFETY_MARGIN = 10  # pixels


def is_image_file(filename):
    """
    Check if file is a valid image format.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file is an image
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    return Path(filename).suffix.lower() in valid_extensions


def split_composite_image(image_path, output_b_scan_dir, output_en_face_dir):
    """
    Split a composite OCT image vertically using centralized logic.
    """
    # Read image using OpenCV
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return False, {}

    # Use centralized utility
    b_scan, en_face, metadata = get_split_indices_and_images(img, divider_safety_margin=DIVIDER_SAFETY_MARGIN)
    
    filename = Path(image_path).name

    # Save B-Scan
    b_scan_path = os.path.join(output_b_scan_dir, filename)
    cv2.imwrite(b_scan_path, b_scan)

    # Save En Face
    en_face_path = os.path.join(output_en_face_dir, filename)
    cv2.imwrite(en_face_path, en_face)

    print(f"✓ Processed: {filename} -> B-Scan ({b_scan.shape}) + En Face ({en_face.shape})")
    
    # Add filename to metadata
    metadata['filename'] = filename
    metadata['safety_margin_applied'] = DIVIDER_SAFETY_MARGIN

    return True, metadata



def process_all_images(raw_dir, b_scan_dir, en_face_dir):
    """
    Process all composite images in the raw directory and save split metadata.

    Args:
        raw_dir: Directory containing raw composite images
        b_scan_dir: Output directory for B-Scans
        en_face_dir: Output directory for En Face images
    """
    # Create output directories if they don't exist
    Path(b_scan_dir).mkdir(parents=True, exist_ok=True)
    Path(en_face_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files from raw directory
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"❌ Error: Raw directory does not exist: {raw_dir}")
        return

    # Filter for image files only
    image_files = [f for f in raw_path.iterdir()
                   if f.is_file() and is_image_file(f.name)]

    if len(image_files) == 0:
        print(f"⚠️  Warning: No image files found in {raw_dir}")
        print("   Place your composite OCT images in this directory.")
        return

    print(f"\n{'='*60}")
    print(f"OCT Image Splitter")
    print(f"{ '='*60}")
    print(f"Found {len(image_files)} image(s) to process\n")

    # Process each image and collect metadata
    success_count = 0
    split_metadata = []
    for img_file in sorted(image_files):
        result = split_composite_image(img_file, b_scan_dir, en_face_dir)
        if result[0]:  # Success
            success_count += 1
            split_metadata.append(result[1])  # Store metadata

    # Write metadata CSV for coordinate transformation integrity
    processed_dir = Path(b_scan_dir).parent
    metadata_path = processed_dir / 'split_metadata.csv'

    with open(metadata_path, 'w', newline='') as f:
        fieldnames = [
            'filename', 'split_column', 'b_scan_width', 'en_face_width',
            'original_width', 'detection_method', 'safety_margin_applied',
            'trim_left_offset', 'scrub_offset', 'b_scrub_offset', 'final_split_column'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_metadata)

    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {success_count}/{len(image_files)} images")
    print(f"{ '='*60}")
    print(f"B-Scans saved to: {b_scan_dir}")
    print(f"En Face images saved to: {en_face_dir}")
    print(f"Split metadata saved to: {metadata_path}")
    print(f"{ '='*60}\n")


def verify_split(b_scan_dir, en_face_dir):
    """
    Verify that B-Scan and En Face directories have matching files.
    
    Args:
        b_scan_dir: Directory containing B-Scans
        en_face_dir: Directory containing En Face images
    """
    b_scan_files = set(f.name for f in Path(b_scan_dir).iterdir() if f.is_file())
    en_face_files = set(f.name for f in Path(en_face_dir).iterdir() if f.is_file())
    
    if b_scan_files == en_face_files:
        print(f"✓ Verification passed: {len(b_scan_files)} matching files in both directories")
    else:
        missing_in_b_scan = en_face_files - b_scan_files
        missing_in_en_face = b_scan_files - en_face_files
        
        if missing_in_b_scan:
            print(f"⚠️  Warning: Files in En Face but not in B-Scan: {missing_in_b_scan}")
        if missing_in_en_face:
            print(f"⚠️  Warning: Files in B-Scan but not in En Face: {missing_in_en_face}")


if __name__ == "__main__":
    # Define paths relative to project root
    BASE_DIR = Path(__file__).parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    B_SCAN_DIR = BASE_DIR / "data" / "processed" / "b_scans"
    EN_FACE_DIR = BASE_DIR / "data" / "processed" / "en_face"
    
    # Process all images
    process_all_images(RAW_DIR, B_SCAN_DIR, EN_FACE_DIR)
    
    # Verify the split
    if Path(B_SCAN_DIR).exists() and Path(EN_FACE_DIR).exists():
        verify_split(B_SCAN_DIR, EN_FACE_DIR)