"""GA Segmentation Service - Refactored from run_analysis.py"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

# Import legacy utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class GASegmenterService:
    """
    Service for Geographic Atrophy (GA) segmentation using K-means clustering.
    Preserves the exact K-means logic from run_analysis.py.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        min_area: int = 500,
        max_circularity: float = 0.8,
        relative_area_threshold: float = 0.2,
        max_regions: int = 3
    ):
        """
        Initialize GA segmentation service.
        
        Args:
            n_clusters: Number of K-means clusters (default: 3)
            min_area: Minimum contour area in pixels (default: 500)
            max_circularity: Maximum circularity to filter out circular objects (default: 0.8)
            relative_area_threshold: Keep regions >= this fraction of largest (default: 0.2)
            max_regions: Maximum number of regions to return (default: 3)
        """
        self.n_clusters = n_clusters
        self.min_area = min_area
        self.max_circularity = max_circularity
        self.relative_area_threshold = relative_area_threshold
        self.max_regions = max_regions
    
    def segment_ga_regions(
        self,
        image: np.ndarray,
        disc_center_x: Optional[float] = None,
        disc_center_y: Optional[float] = None,
        disc_height_pixels: Optional[float] = None,
        en_face_split_x: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Segment GA regions using K-means clustering.
        
        This method preserves the exact logic from run_analysis.py:
        segment_macular_ga_kmeans() function (lines 77-149).
        
        Args:
            image: Full composite OCT image (BGR)
            disc_center_x: Optional disc center X for masking
            disc_center_y: Optional disc center Y for masking
            disc_height_pixels: Optional disc height for creating mask
            en_face_split_x: Optional split point to extract en-face region
        
        Returns:
            List of contours (numpy arrays) representing GA regions
        """
        # Extract en-face region if split point provided
        if en_face_split_x is not None:
            en_face = image[:, en_face_split_x:, :]
            # Adjust disc coordinates to en-face space
            if disc_center_x is not None:
                disc_center_x_local = disc_center_x - en_face_split_x
            else:
                disc_center_x_local = None
        else:
            en_face = image
            disc_center_x_local = disc_center_x
        
        # Convert to grayscale
        if len(en_face.shape) == 3:
            gray = cv2.cvtColor(en_face, cv2.COLOR_BGR2GRAY)
        else:
            gray = en_face
        
        h, w = gray.shape
        
        # Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Mask out Optic Disc if coordinates provided
        if disc_center_x_local is not None and disc_center_y is not None and disc_height_pixels is not None:
            # Create circular mask around disc
            disc_radius = int(disc_height_pixels * 0.6)  # Slightly larger than disc
            mask = np.ones_like(enhanced)
            cv2.circle(
                mask,
                (int(disc_center_x_local), int(disc_center_y)),
                disc_radius,
                0,
                -1
            )
            enhanced = cv2.bitwise_and(enhanced, mask)
        
        # K-Means Clustering
        pixel_values = enhanced.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixel_values,
            self.n_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        lesion_cluster_index = np.argmax(centers)  # Brightest cluster (GA is bright)
        
        labels = labels.flatten()
        lesion_mask = (labels == lesion_cluster_index).astype(np.uint8) * 255
        lesion_mask = lesion_mask.reshape(gray.shape)
        
        # Morphological Cleanup
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find Contours
        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            print("  [GASegmenter] No GA regions detected")
            return []
        
        # Filter contours
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 1. Size Filter
            if area < self.min_area:
                continue
            
            # 2. Circularity Filter (reject circles - likely blood vessels or disc)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > self.max_circularity:
                continue
            
            # 3. Location Filter (reject regions touching image borders)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if x <= 2 or y <= 2 or (x + w_box) >= (w - 2) or (y + h_box) >= (h - 2):
                continue
            
            candidates.append(cnt)
        
        if not candidates:
            print("  [GASegmenter] No valid GA regions after filtering")
            return []
        
        # Sort by Area (largest first)
        candidates.sort(key=cv2.contourArea, reverse=True)
        
        # Big Fish Rule: Keep regions >= 20% of largest
        largest_area = cv2.contourArea(candidates[0])
        threshold_area = self.relative_area_threshold * largest_area
        
        final_contours = [c for c in candidates if cv2.contourArea(c) >= threshold_area]
        
        # Limit to max regions
        final_contours = final_contours[:self.max_regions]
        
        # Adjust contours back to original image coordinates if needed
        if en_face_split_x is not None:
            adjusted_contours = []
            for cnt in final_contours:
                adjusted = cnt.copy()
                adjusted[:, 0, 0] += en_face_split_x  # Shift X coordinates
                adjusted_contours.append(adjusted)
            final_contours = adjusted_contours
        
        print(f"  [GASegmenter] Detected {len(final_contours)} GA regions")
        return final_contours
    
    def contours_to_json(self, contours: List[np.ndarray]) -> List[List[Tuple[int, int]]]:
        """
        Convert OpenCV contours to JSON-serializable format.
        
        Args:
            contours: List of OpenCV contours
        
        Returns:
            List of regions, each as list of (x, y) tuples
        """
        regions = []
        for cnt in contours:
            points = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]
            regions.append(points)
        return regions
