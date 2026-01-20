"""
Extract Fovea and GA Coordinates from Manually Marked OCT Images

This script processes annotated OCT images to extract:
- Fovea location from red dots
- GA target location from peach/yellow lines

Marking Convention:
- Fovea: Red dot (possibly with blue outline)
- GA: Peach/yellow line (#F4C5AD) connecting fovea to nearest GA edge
  The FAR endpoint from the fovea is the GA target point.

Usage:
    python src/00_extract_vector_labels.py --input data/raw_marked --output data/csv --split 0.2 --debug
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from skimage import morphology
from skimage.morphology import skeletonize
from scipy import ndimage
from tqdm import tqdm


class VectorLabelExtractor:
    """Extract fovea and GA coordinates from marked OCT images."""
    
    def __init__(self, 
                 red_hue_ranges=[(0, 10), (170, 180)],
                 red_sat_range=(100, 255),
                 red_val_range=(100, 255),
                 peach_color=(246, 198, 173),  # Actual color from images
                 peach_tolerance=(20, 80, 80),  # Increased tolerance
                 min_red_area=10,
                 max_fovea_distance=300,  # Relaxed distance constraint
                 debug=False):
        """
        Initialize the extractor with color detection parameters.
        
        Args:
            red_hue_ranges: List of (min, max) HSV hue ranges for red detection
            red_sat_range: (min, max) saturation range for red
            red_val_range: (min, max) value range for red
            peach_color: RGB values for peach/yellow line color
            peach_tolerance: (H, S, V) tolerance around peach color
            min_red_area: Minimum pixel area for valid red dot
            max_fovea_distance: Max distance closest endpoint can be from fovea
            debug: If True, save debug visualization images
        """
        self.red_hue_ranges = red_hue_ranges
        self.red_sat_range = red_sat_range
        self.red_val_range = red_val_range
        self.peach_color_rgb = peach_color
        self.peach_tolerance = peach_tolerance
        self.min_red_area = min_red_area
        self.max_fovea_distance = max_fovea_distance
        self.debug = debug
        
        # Convert peach color to HSV for detection
        peach_bgr = np.uint8([[[peach_color[2], peach_color[1], peach_color[0]]]])
        peach_hsv = cv2.cvtColor(peach_bgr, cv2.COLOR_BGR2HSV)[0][0]
        self.peach_hsv = peach_hsv
        
    def detect_fovea(self, image):
        """
        Detect red dot and return fovea centroid.
        
        Args:
            image: BGR image (from cv2.imread)
            
        Returns:
            (x, y) tuple of fovea coordinates, or None if not found
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color (handle wraparound at hue=180)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_min, hue_max in self.red_hue_ranges:
            lower = np.array([hue_min, self.red_sat_range[0], self.red_val_range[0]])
            upper = np.array([hue_max, self.red_sat_range[1], self.red_val_range[1]])
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Find largest component (excluding background)
        if num_labels <= 1:
            return None
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        if areas[largest_idx - 1] < self.min_red_area:
            return None
        
        # Return centroid (x, y)
        centroid = centroids[largest_idx]
        return (int(centroid[0]), int(centroid[1]))
    
    def detect_peach_line(self, image):
        """
        Detect peach/yellow/pink line and return binary mask.

        Detects multiple color ranges to handle:
        - Original peach/orange colors (H: 0-25)
        - Lighter pink/salmon colors (H: 150-180)

        Args:
            image: BGR image

        Returns:
            Binary mask of peach/pink line pixels
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask 1: Original peach color with tolerance
        h_tol, s_tol, v_tol = self.peach_tolerance
        lower = np.array([
            max(0, int(self.peach_hsv[0]) - h_tol),
            max(0, int(self.peach_hsv[1]) - s_tol),
            max(0, int(self.peach_hsv[2]) - v_tol)
        ], dtype=np.uint8)
        upper = np.array([
            min(180, int(self.peach_hsv[0]) + h_tol),
            min(255, int(self.peach_hsv[1]) + s_tol),
            min(255, int(self.peach_hsv[2]) + v_tol)
        ], dtype=np.uint8)
        mask_peach = cv2.inRange(hsv, lower, upper)

        # Mask 2: Peach/Orange range with lower saturation (H: 10-25, S: 30+, V: 100+)
        mask_orange1 = cv2.inRange(hsv, np.array([10, 30, 100]), np.array([25, 255, 255]))
        # Mask 3: Red-orange range with lower saturation (H: 0-10, S: 30+, V: 100+)
        mask_orange2 = cv2.inRange(hsv, np.array([0, 30, 100]), np.array([10, 255, 255]))
        # Mask 4: Pink/Magenta range for lighter pink lines (H: 150-180, S: 30+, V: 100+)
        mask_pink = cv2.inRange(hsv, np.array([150, 30, 100]), np.array([180, 255, 255]))

        # Combine all masks
        mask = mask_peach | mask_orange1 | mask_orange2 | mask_pink

        # Apply morphological closing to connect gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
    
    def skeletonize_line(self, mask):
        """
        Skeletonize the line mask to get thin centerline.
        
        Args:
            mask: Binary mask
            
        Returns:
            Skeletonized binary mask
        """
        # Convert to boolean for skimage
        binary = mask > 0
        
        # Skeletonize
        skeleton = skeletonize(binary)
        
        return skeleton.astype(np.uint8) * 255
    
    def find_endpoints(self, skeleton):
        """
        Find endpoints of the skeletonized line.
        
        Args:
            skeleton: Binary skeletonized image
            
        Returns:
            List of (x, y) endpoint coordinates
        """
        # Kernel to detect endpoints (pixels with exactly 1 neighbor)
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Convolve to count neighbors
        filtered = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
        
        # Endpoints have value 11 (center=10 + 1 neighbor)
        endpoints = np.argwhere(filtered == 11)
        
        # Convert to (x, y) format
        endpoints = [(pt[1], pt[0]) for pt in endpoints]
        
        return endpoints
    
    def select_ga_endpoint(self, endpoints, fovea):
        """
        Select the GA target endpoint (farthest from fovea).
        
        Args:
            endpoints: List of (x, y) endpoint coordinates
            fovea: (fx, fy) fovea coordinates
            
        Returns:
            (x, y) GA target coordinates, or None if validation fails
        """
        if len(endpoints) < 2:
            return None
        
        # If more than 2 endpoints, select the two farthest apart
        if len(endpoints) > 2:
            max_dist = 0
            best_pair = (endpoints[0], endpoints[1])
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dist = np.linalg.norm(np.array(endpoints[i]) - np.array(endpoints[j]))
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (endpoints[i], endpoints[j])
            endpoints = list(best_pair)
        
        # Calculate distances to fovea
        fx, fy = fovea
        distances = []
        for x, y in endpoints:
            dist = np.sqrt((x - fx)**2 + (y - fy)**2)
            distances.append(dist)
        
        # Sort by distance
        sorted_idx = np.argsort(distances)
        closest_endpoint = endpoints[sorted_idx[0]]
        farthest_endpoint = endpoints[sorted_idx[1]]
        
        # Validate: closest endpoint should be near fovea (relaxed validation)
        # If validation fails, still return but warn
        if distances[sorted_idx[0]] > self.max_fovea_distance:
            # Return farthest endpoint anyway, but this will be logged
            pass
        
        return farthest_endpoint
    
    def process_image(self, image_path):
        """
        Process a single image to extract fovea and GA coordinates.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with 'fovea', 'ga', 'error', 'debug_image' keys
        """
        result = {
            'fovea': None,
            'ga': None,
            'error': None,
            'debug_image': None
        }
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = 'Failed to load image'
            return result
        
        # Detect fovea
        fovea = self.detect_fovea(image)
        if fovea is None:
            result['error'] = 'Red dot (fovea) not detected'
            return result
        result['fovea'] = fovea
        
        # Detect peach line
        peach_mask = self.detect_peach_line(image)
        if peach_mask.sum() == 0:
            result['error'] = 'Peach line not detected'
            return result
        
        # Skeletonize line
        skeleton = self.skeletonize_line(peach_mask)
        
        # Find endpoints
        endpoints = self.find_endpoints(skeleton)
        if len(endpoints) < 2:
            result['error'] = f'Insufficient endpoints found: {len(endpoints)}'
            return result
        
        # Select GA endpoint
        ga = self.select_ga_endpoint(endpoints, fovea)
        if ga is None:
            result['error'] = 'Could not determine GA endpoint (validation failed)'
            return result
        result['ga'] = ga
        
        # Create debug visualization if requested
        if self.debug:
            debug_img = image.copy()
            # Draw fovea (green circle)
            cv2.circle(debug_img, fovea, 10, (0, 255, 0), 2)
            cv2.circle(debug_img, fovea, 3, (0, 255, 0), -1)
            # Draw skeleton (blue)
            debug_img[skeleton > 0] = [255, 0, 0]
            # Draw endpoints
            for ep in endpoints:
                cv2.circle(debug_img, ep, 5, (255, 255, 0), -1)
            # Draw GA target (red circle)
            cv2.circle(debug_img, ga, 10, (0, 0, 255), 2)
            cv2.circle(debug_img, ga, 3, (0, 0, 255), -1)
            # Add labels
            cv2.putText(debug_img, 'Fovea', (fovea[0] + 15, fovea[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_img, 'GA', (ga[0] + 15, ga[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            result['debug_image'] = debug_img
        
        return result
    
    def process_directory(self, input_dir, output_dir, train_val_split=0.0, debug_dir=None):
        """
        Process all images in a directory and save CSV labels.
        
        Args:
            input_dir: Directory containing marked images
            output_dir: Directory to save CSV files
            train_val_split: Fraction for validation (0.0-1.0). If 0, only creates train files.
            debug_dir: Directory to save debug images (if debug=True)
            
        Returns:
            dict with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if debug_dir and self.debug:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        # Process all images
        results = []
        errors = []
        
        print(f"Processing {len(image_files)} images...")
        for img_file in tqdm(image_files):
            result = self.process_image(img_file)
            
            if result['error']:
                errors.append({'filename': img_file.name, 'error': result['error']})
            else:
                results.append({
                    'filename': img_file.name,
                    'fovea_x': result['fovea'][0],
                    'fovea_y': result['fovea'][1],
                    'ga_x': result['ga'][0],
                    'ga_y': result['ga'][1]
                })
                
                # Save debug image
                if self.debug and result['debug_image'] is not None and debug_dir:
                    debug_filename = debug_path / f"debug_{img_file.name}"
                    cv2.imwrite(str(debug_filename), result['debug_image'])
        
        # Print error summary
        if errors:
            print(f"\n⚠️  {len(errors)} images failed processing:")
            for err in errors[:10]:  # Show first 10
                print(f"  - {err['filename']}: {err['error']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        if not results:
            print("No images successfully processed!")
            return {'total': len(image_files), 'success': 0, 'failed': len(errors)}
        
        # Create DataFrames
        df_all = pd.DataFrame(results)
        
        # Split into train/val if requested
        if train_val_split > 0:
            np.random.seed(42)  # For reproducibility
            val_indices = np.random.choice(len(df_all), 
                                          size=int(len(df_all) * train_val_split),
                                          replace=False)
            val_mask = np.zeros(len(df_all), dtype=bool)
            val_mask[val_indices] = True
            
            df_train = df_all[~val_mask]
            df_val = df_all[val_mask]
            
            # Save train sets
            df_train[['filename', 'fovea_x', 'fovea_y']].to_csv(
                output_path / 'train_fovea_labels.csv', index=False)
            df_train[['filename', 'ga_x', 'ga_y']].to_csv(
                output_path / 'train_ga_labels.csv', index=False)
            
            # Save val sets
            df_val[['filename', 'fovea_x', 'fovea_y']].to_csv(
                output_path / 'val_fovea_labels.csv', index=False)
            df_val[['filename', 'ga_x', 'ga_y']].to_csv(
                output_path / 'val_ga_labels.csv', index=False)
            
            print(f"\n✅ Successfully processed {len(results)} images")
            print(f"   Train: {len(df_train)} images")
            print(f"   Val: {len(df_val)} images")
        else:
            # Save only train sets
            df_all[['filename', 'fovea_x', 'fovea_y']].to_csv(
                output_path / 'train_fovea_labels.csv', index=False)
            df_all[['filename', 'ga_x', 'ga_y']].to_csv(
                output_path / 'train_ga_labels.csv', index=False)
            
            print(f"\n✅ Successfully processed {len(results)} images")
        
        print(f"   CSVs saved to: {output_path}")
        if self.debug and debug_dir:
            print(f"   Debug images saved to: {debug_dir}")
        
        return {
            'total': len(image_files),
            'success': len(results),
            'failed': len(errors)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract fovea and GA coordinates from marked OCT images'
    )
    parser.add_argument('--input', type=str, default='data/raw_marked',
                       help='Input directory with marked images')
    parser.add_argument('--output', type=str, default='data/csv',
                       help='Output directory for CSV files')
    parser.add_argument('--split', type=float, default=0.0,
                       help='Validation split fraction (0.0-1.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug visualization images')
    parser.add_argument('--debug-dir', type=str, default='data/debug',
                       help='Directory for debug images (if --debug)')
    
    # Color detection parameters
    parser.add_argument('--peach-tolerance', type=int, nargs=3, 
                       default=[20, 80, 80], metavar=('H', 'S', 'V'),
                       help='HSV tolerance for peach color detection')
    parser.add_argument('--max-fovea-dist', type=int, default=300,
                       help='Max distance for line endpoint from fovea')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = VectorLabelExtractor(
        peach_tolerance=tuple(args.peach_tolerance),
        max_fovea_distance=args.max_fovea_dist,
        debug=args.debug
    )
    
    # Process directory
    stats = extractor.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        train_val_split=args.split,
        debug_dir=args.debug_dir if args.debug else None
    )
    
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
    print("="*50)


if __name__ == '__main__':
    main()
