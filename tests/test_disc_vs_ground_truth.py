"""
Ground-Truth Validation for Disc Detection Algorithm

This script validates the disc detector against 50 manually marked images
from input_images_marked/. Each image has a peach/salmon vertical line marking
the ground-truth optic disc position.

Validation Protocol:
1. Load each marked image from input_images_marked/
2. Extract ground-truth disc line (peach/salmon color ~#F4C5AD) via HSV filtering
3. Run DiscDetectorService on the same image
4. Compute error metrics: delta X, delta top_y, delta bottom_y, delta height
5. Generate overlay visualizations (ground-truth vs predicted)
6. Output summary CSV with all metrics

Output:
- debug_output/disc_validation/<image_name>_comparison.png (visual overlays)
- debug_output/disc_validation/summary.csv (all metrics)
- Console: aggregate statistics (mean/median/max errors)
"""

import os
import sys
from pathlib import Path

# Add src to path FIRST
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

from src.api.services.disc_detector import DiscDetectorService
from src.utils.image_utils import get_split_indices_and_images


def extract_ground_truth_disc_line(
    image: np.ndarray,
    en_face_split_x: int
) -> Optional[Dict[str, float]]:
    """
    Extract ground-truth disc line from the marked image.
    
    The ground-truth is a peach/salmon vertical line (#F4C5AD-ish) drawn on
    the en-face region to mark the disc position and 1800um extent.
    
    Args:
        image: Full composite OCT image (BGR)
        en_face_split_x: X offset where en-face region starts
    
    Returns:
        Dictionary with:
            - disc_center_x: X coordinate of line center (original image coords)
            - disc_top_y: Y coordinate of top of line
            - disc_bottom_y: Y coordinate of bottom of line
            - disc_height_pixels: Height of line in pixels
        Or None if extraction fails
    """
    # Extract en-face region
    en_face = image[:, en_face_split_x:, :]
    h_ef, w_ef = en_face.shape[:2]
    
    # Convert to HSV for color-based extraction
    hsv = cv2.cvtColor(en_face, cv2.COLOR_BGR2HSV)
    
    # Peach/salmon line detection: Hue ~10-30 (orange-red), medium saturation, high value
    # Using a wider range to account for JPEG artifacts and annotation tool variations
    lower_peach = np.array([5, 50, 150])   # Light salmon
    upper_peach = np.array([35, 200, 255])  # Peach/orange
    
    mask = cv2.inRange(hsv, lower_peach, upper_peach)
    
    # Morphological cleanup to connect broken line segments
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find line pixels
    line_points = cv2.findNonZero(mask)
    
    if line_points is None or len(line_points) < 10:
        # Try alternate color range (darker salmon)
        lower_peach_alt = np.array([0, 40, 120])
        upper_peach_alt = np.array([40, 220, 255])
        mask_alt = cv2.inRange(hsv, lower_peach_alt, upper_peach_alt)
        mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_CLOSE, kernel)
        line_points = cv2.findNonZero(mask_alt)
        
        if line_points is None or len(line_points) < 10:
            return None
    
    # Extract line coordinates (in en-face space)
    points = line_points.reshape(-1, 2)  # [N, 2] where each row is [x, y]
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Vertical line characteristics: X should be roughly constant, Y spans a range
    x_center_local = np.median(x_coords)
    y_min_local = np.min(y_coords)
    y_max_local = np.max(y_coords)
    
    # Filter outliers: keep only points within 20px of median X (handles slight curvature)
    x_tolerance = 20
    valid_mask = np.abs(x_coords - x_center_local) < x_tolerance
    x_coords_filtered = x_coords[valid_mask]
    y_coords_filtered = y_coords[valid_mask]
    
    if len(x_coords_filtered) < 10:
        return None
    
    # Recalculate with filtered points
    x_center_local = np.mean(x_coords_filtered)
    y_min_local = np.min(y_coords_filtered)
    y_max_local = np.max(y_coords_filtered)
    
    # Convert to original image coordinates
    disc_center_x = en_face_split_x + x_center_local
    disc_top_y = y_min_local
    disc_bottom_y = y_max_local
    disc_height_pixels = disc_bottom_y - disc_top_y
    
    return {
        'disc_center_x': float(disc_center_x),
        'disc_center_y': float((disc_top_y + disc_bottom_y) / 2),
        'disc_top_y': float(disc_top_y),
        'disc_bottom_y': float(disc_bottom_y),
        'disc_height_pixels': float(disc_height_pixels),
        'pixel_to_micron_ratio': 1800.0 / disc_height_pixels if disc_height_pixels > 0 else 0.0
    }


def compute_error_metrics(
    ground_truth: Dict[str, float],
    predicted: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute error metrics between ground-truth and predicted disc coordinates.
    
    Returns:
        Dictionary with:
            - delta_center_x: Absolute error in center X (pixels)
            - delta_center_y: Absolute error in center Y (pixels)
            - delta_top_y: Absolute error in top Y (pixels)
            - delta_bottom_y: Absolute error in bottom Y (pixels)
            - delta_height_pixels: Absolute error in height (pixels)
            - delta_height_percent: Percent error in height (%)
            - delta_ratio_percent: Percent error in pixel-to-micron ratio (%)
    """
    delta_center_x = abs(predicted['disc_center_x'] - ground_truth['disc_center_x'])
    delta_center_y = abs(predicted['disc_center_y'] - ground_truth['disc_center_y'])
    delta_top_y = abs(predicted['disc_top_y'] - ground_truth['disc_top_y'])
    delta_bottom_y = abs(predicted['disc_bottom_y'] - ground_truth['disc_bottom_y'])
    delta_height = predicted['disc_height_pixels'] - ground_truth['disc_height_pixels']
    
    delta_height_percent = (delta_height / ground_truth['disc_height_pixels']) * 100.0
    
    ratio_error = predicted['pixel_to_micron_ratio'] - ground_truth['pixel_to_micron_ratio']
    delta_ratio_percent = (ratio_error / ground_truth['pixel_to_micron_ratio']) * 100.0
    
    return {
        'delta_center_x': delta_center_x,
        'delta_center_y': delta_center_y,
        'delta_top_y': delta_top_y,
        'delta_bottom_y': delta_bottom_y,
        'delta_height_pixels': delta_height,
        'delta_height_percent': delta_height_percent,
        'delta_ratio_percent': delta_ratio_percent,
    }


def create_comparison_visualization(
    image: np.ndarray,
    ground_truth: Dict[str, float],
    predicted: Dict[str, float],
    en_face_split_x: int,
    output_path: str
) -> None:
    """
    Create side-by-side visualization of ground-truth vs predicted disc lines.
    
    Args:
        image: Full composite OCT image (BGR)
        ground_truth: Ground-truth disc coordinates
        predicted: Predicted disc coordinates
        en_face_split_x: X offset where en-face starts
        output_path: Path to save visualization
    """
    # Extract en-face region for visualization
    en_face = image[:, en_face_split_x:, :].copy()
    h, w = en_face.shape[:2]
    
    # Create two copies: one for ground-truth, one for predicted
    vis_gt = en_face.copy()
    vis_pred = en_face.copy()
    
    # Ground-truth line (GREEN) -- coordinates are already in en-face-local space
    gt_x = int(ground_truth['disc_center_x'] - en_face_split_x)
    gt_top = int(ground_truth['disc_top_y'])
    gt_bottom = int(ground_truth['disc_bottom_y'])
    
    cv2.line(vis_gt, (gt_x, gt_top), (gt_x, gt_bottom), (0, 255, 0), 3)
    cv2.circle(vis_gt, (gt_x, gt_top), 8, (0, 255, 0), -1)
    cv2.circle(vis_gt, (gt_x, gt_bottom), 8, (0, 255, 0), -1)
    cv2.putText(vis_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_gt, f"H: {ground_truth['disc_height_pixels']:.1f} px", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Predicted line (RED) -- coordinates are in original image space, convert to en-face-local
    pred_x = int(predicted['disc_center_x'] - en_face_split_x)
    pred_top = int(predicted['disc_top_y'])
    pred_bottom = int(predicted['disc_bottom_y'])
    
    cv2.line(vis_pred, (pred_x, pred_top), (pred_x, pred_bottom), (0, 0, 255), 3)
    cv2.circle(vis_pred, (pred_x, pred_top), 8, (0, 0, 255), -1)
    cv2.circle(vis_pred, (pred_x, pred_bottom), 8, (0, 0, 255), -1)
    cv2.putText(vis_pred, "Predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis_pred, f"H: {predicted['disc_height_pixels']:.1f} px", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Overlay visualization (both lines on one image)
    vis_overlay = en_face.copy()
    cv2.line(vis_overlay, (gt_x, gt_top), (gt_x, gt_bottom), (0, 255, 0), 3)
    cv2.line(vis_overlay, (pred_x, pred_top), (pred_x, pred_bottom), (0, 0, 255), 2)
    cv2.putText(vis_overlay, "Overlay (Green=GT, Red=Pred)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine all three
    composite = np.hstack([vis_gt, vis_pred, vis_overlay])
    
    cv2.imwrite(output_path, composite)


def process_single_image(
    image_path: str,
    detector: DiscDetectorService,
    output_dir: Path
) -> Optional[Dict]:
    """
    Process a single marked image and compute validation metrics.
    
    Returns:
        Dictionary with all metrics, or None if ground-truth extraction failed
    """
    image_name = Path(image_path).stem
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: Could not load image")
        return None
    
    # Split composite image to get en_face_split_x
    _, _, metadata = get_split_indices_and_images(img, divider_safety_margin=10)
    en_face_split_x = metadata['final_split_column']
    
    # Extract ground-truth disc line
    print(f"  Extracting ground-truth disc line...")
    ground_truth = extract_ground_truth_disc_line(img, en_face_split_x)
    
    if ground_truth is None:
        print(f"  ERROR: Could not extract ground-truth disc line (peach line not found)")
        return None
    
    print(f"  Ground-truth: center_x={ground_truth['disc_center_x']:.1f}, "
          f"top_y={ground_truth['disc_top_y']:.1f}, "
          f"bottom_y={ground_truth['disc_bottom_y']:.1f}, "
          f"height={ground_truth['disc_height_pixels']:.1f} px")
    
    # Run disc detector
    print(f"  Running disc detector...")
    predicted = detector.detect_from_image(img)
    
    print(f"  Predicted: center_x={predicted['disc_center_x']:.1f}, "
          f"top_y={predicted['disc_top_y']:.1f}, "
          f"bottom_y={predicted['disc_bottom_y']:.1f}, "
          f"height={predicted['disc_height_pixels']:.1f} px")
    
    # Compute error metrics
    errors = compute_error_metrics(ground_truth, predicted)
    
    print(f"\n  ERRORS:")
    print(f"    Delta center X: {errors['delta_center_x']:.1f} px")
    print(f"    Delta center Y: {errors['delta_center_y']:.1f} px")
    print(f"    Delta top Y: {errors['delta_top_y']:.1f} px")
    print(f"    Delta bottom Y: {errors['delta_bottom_y']:.1f} px")
    print(f"    Delta height: {errors['delta_height_pixels']:.1f} px ({errors['delta_height_percent']:.1f}%)")
    print(f"    Delta ratio: {errors['delta_ratio_percent']:.2f}%")
    
    # Create comparison visualization
    vis_path = output_dir / f"{image_name}_comparison.png"
    create_comparison_visualization(img, ground_truth, predicted, en_face_split_x, str(vis_path))
    print(f"  Saved visualization: {vis_path}")
    
    # Return all data for summary
    return {
        'image_name': image_name,
        'gt_center_x': ground_truth['disc_center_x'],
        'gt_center_y': ground_truth['disc_center_y'],
        'gt_top_y': ground_truth['disc_top_y'],
        'gt_bottom_y': ground_truth['disc_bottom_y'],
        'gt_height_pixels': ground_truth['disc_height_pixels'],
        'gt_pixel_to_micron_ratio': ground_truth['pixel_to_micron_ratio'],
        'pred_center_x': predicted['disc_center_x'],
        'pred_center_y': predicted['disc_center_y'],
        'pred_top_y': predicted['disc_top_y'],
        'pred_bottom_y': predicted['disc_bottom_y'],
        'pred_height_pixels': predicted['disc_height_pixels'],
        'pred_pixel_to_micron_ratio': predicted['pixel_to_micron_ratio'],
        **errors
    }


def generate_summary_report(results: List[Dict], output_dir: Path):
    """Generate summary CSV and aggregate statistics."""
    if not results:
        print("\nNo results to summarize")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"{'='*70}")
    
    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nTotal images processed: {len(results)}")
    
    print(f"\nDelta Center X (pixels):")
    print(f"  Mean:   {df['delta_center_x'].mean():.2f}")
    print(f"  Median: {df['delta_center_x'].median():.2f}")
    print(f"  Std:    {df['delta_center_x'].std():.2f}")
    print(f"  Max:    {df['delta_center_x'].max():.2f}")
    
    print(f"\nDelta Center Y (pixels):")
    print(f"  Mean:   {df['delta_center_y'].mean():.2f}")
    print(f"  Median: {df['delta_center_y'].median():.2f}")
    print(f"  Std:    {df['delta_center_y'].std():.2f}")
    print(f"  Max:    {df['delta_center_y'].max():.2f}")
    
    print(f"\nDelta Height (percent):")
    print(f"  Mean:   {df['delta_height_percent'].mean():.2f}%")
    print(f"  Median: {df['delta_height_percent'].median():.2f}%")
    print(f"  Std:    {df['delta_height_percent'].std():.2f}%")
    print(f"  Max:    {df['delta_height_percent'].abs().max():.2f}%")
    
    print(f"\nDelta Pixel-to-Micron Ratio (percent):")
    print(f"  Mean:   {df['delta_ratio_percent'].mean():.2f}%")
    print(f"  Median: {df['delta_ratio_percent'].median():.2f}%")
    print(f"  Std:    {df['delta_ratio_percent'].std():.2f}%")
    print(f"  Max:    {df['delta_ratio_percent'].abs().max():.2f}%")
    
    # Flag critical errors (> 20% height error)
    critical_errors = df[df['delta_height_percent'].abs() > 20.0]
    if len(critical_errors) > 0:
        print(f"\n⚠️  WARNING: {len(critical_errors)} images have > 20% height error:")
        for _, row in critical_errors.iterrows():
            print(f"    - {row['image_name']}: {row['delta_height_percent']:.1f}%")
    
    # Check for systematic bias
    mean_height_error = df['delta_height_pixels'].mean()
    if abs(mean_height_error) > 10:
        direction = "taller" if mean_height_error > 0 else "shorter"
        print(f"\n⚠️  SYSTEMATIC BIAS DETECTED:")
        print(f"    Algorithm consistently detects disc {abs(mean_height_error):.1f} px {direction} than ground-truth")


def main():
    """Main execution function."""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_dir = project_root / "input_images_marked"
    output_dir = project_root / "debug_output" / "disc_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find marked images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.bmp"))
    
    # Filter out README
    image_files = [f for f in image_files if f.stem != "README"]
    
    if not image_files:
        print(f"\nERROR: No marked images found in {input_dir}")
        return 1
    
    print(f"\nFound {len(image_files)} marked images in {input_dir}")
    
    # Initialize detector
    print(f"\nInitializing disc detector...")
    model_path = project_root / "weights" / "best_disc_model.pth"
    if not model_path.exists():
        print(f"ERROR: Model weights not found at {model_path}")
        return 1
    
    detector = DiscDetectorService(model_path=str(model_path))
    
    # Process all images
    results = []
    failed_images = []
    
    for img_path in sorted(image_files):
        try:
            result = process_single_image(str(img_path), detector, output_dir)
            if result:
                results.append(result)
            else:
                failed_images.append(img_path.name)
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            failed_images.append(img_path.name)
    
    # Generate summary
    if results:
        generate_summary_report(results, output_dir)
        print(f"\n✓ Validation complete! Processed {len(results)}/{len(image_files)} images.")
    else:
        print(f"\nNo images processed successfully.")
    
    if failed_images:
        print(f"\n⚠️  Failed to process {len(failed_images)} images:")
        for name in failed_images:
            print(f"    - {name}")
    
    return 0 if results else 1


if __name__ == "__main__":
    exit(main())
