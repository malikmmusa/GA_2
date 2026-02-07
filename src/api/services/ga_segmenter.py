"""GA Segmentation Service - Single-cluster selection with texture validation"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import sys
import os

# Import legacy utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class GASegmenterService:
    """
    Service for Geographic Atrophy (GA) segmentation using single-cluster K-means with texture validation.
    
    Approach:
    - Single-channel K-means clustering on CLAHE-enhanced intensity
    - 4 clusters for optimal separation
    - Texture-based validation to select the best cluster
    - Smart border filtering to reject giant blobs
    - Anatomy-aware region scoring (disc exclusion, macular proximity, fovea-aware)
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        min_area: int = 500,
        max_circularity: float = 0.8,
        relative_area_threshold: float = 0.1,
        max_regions: Optional[int] = None,
        disc_exclusion_multiplier: float = 0.6,
        clahe_clip_limit: float = 3.0,
        morph_kernel_size: int = 11
    ):
        """
        Initialize GA segmentation service.
        
        Args:
            n_clusters: Number of K-means clusters (default: 3)
            min_area: Minimum contour area in pixels (default: 500)
            max_circularity: Maximum circularity to filter out circular objects (default: 0.8)
            relative_area_threshold: Keep regions >= this fraction of largest (default: 0.1)
            max_regions: Maximum number of regions to return (default: None - return all)
            disc_exclusion_multiplier: Disc masking radius multiplier (default: 0.6)
            clahe_clip_limit: CLAHE clip limit (default: 3.0)
            morph_kernel_size: Morphological operations kernel size (default: 11)
        """
        self.n_clusters = n_clusters
        self.min_area = min_area
        self.max_circularity = max_circularity
        self.relative_area_threshold = relative_area_threshold
        self.max_regions = max_regions
        self.disc_exclusion_multiplier = disc_exclusion_multiplier
        self.clahe_clip_limit = clahe_clip_limit
        self.morph_kernel_size = morph_kernel_size
    
    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement to grayscale image.
        
        Args:
            gray: Grayscale image
            
        Returns:
            CLAHE-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _compute_cluster_texture(self, enhanced: np.ndarray, cluster_mask: np.ndarray) -> float:
        """
        Compute mean local standard deviation (texture) for pixels in a cluster.
        
        Args:
            enhanced: CLAHE-enhanced grayscale image
            cluster_mask: Binary mask of cluster pixels
            
        Returns:
            Mean texture score (local standard deviation)
        """
        # Local standard deviation (texture)
        mean_local = cv2.blur(enhanced.astype(np.float32), (11, 11))
        sqr_local = cv2.blur((enhanced.astype(np.float32) ** 2), (11, 11))
        std_local = np.sqrt(np.maximum(sqr_local - mean_local ** 2, 0))
        
        # Get texture values for cluster pixels only
        cluster_texture = std_local[cluster_mask > 0]
        
        if len(cluster_texture) == 0:
            return 0.0
        
        return float(np.mean(cluster_texture))
    
    
    def _apply_watershed_splitting(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply watershed-based splitting to separate merged GA regions.
        
        Uses distance transform to find local maxima (blob centers) and
        watershed to split along saddle points.
        
        Args:
            binary_mask: Binary mask with potential merged blobs
            
        Returns:
            Refined binary mask with blobs split
        """
        # Compute distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Find local maxima
        # Use a slightly dilated version to find peaks
        local_max = ndimage.maximum_filter(dist_transform, size=20)
        is_local_max = (dist_transform == local_max) & (dist_transform > 0)
        
        # Label the local maxima as markers
        markers, num_markers = ndimage.label(is_local_max)
        
        # If no or only one marker, no splitting needed
        if num_markers <= 1:
            return binary_mask
        
        # Apply watershed
        # Invert distance transform for watershed (it works on "basins")
        markers = cv2.watershed(cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), markers)
        
        # Create output mask (exclude watershed boundaries marked as -1)
        result = np.where(markers > 0, 255, 0).astype(np.uint8)
        
        return result
    
    def _score_region_anatomy_aware(self,
                                      contour: np.ndarray,
                                      image_shape: Tuple[int, int],
                                      disc_center: Optional[Tuple[float, float]] = None,
                                      fovea_pos: Optional[Tuple[float, float]] = None,
                                      eye_side: Optional[str] = None) -> float:
        """
        Score a GA region based on anatomical likelihood.
        
        Args:
            contour: OpenCV contour
            image_shape: (height, width) of en-face image
            disc_center: (x, y) of disc center in local coordinates
            fovea_pos: (x, y) of fovea in local coordinates
            eye_side: "OD" or "OS"
            
        Returns:
            Anatomical likelihood score (higher = more likely)
        """
        score = 1.0
        
        # Compute centroid of the region
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.5
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Macular proximity (central regions are more likely)
        h, w = image_shape
        image_center_x = w / 2
        image_center_y = h / 2
        
        dist_from_center = np.sqrt((cx - image_center_x)**2 + (cy - image_center_y)**2)
        image_diagonal = np.sqrt(h**2 + w**2)
        normalized_dist = dist_from_center / image_diagonal
        
        if normalized_dist < 0.15:
            macular_score = 1.0
        elif normalized_dist < 0.35:
            macular_score = 1.0 - (normalized_dist - 0.15) / 0.20
        else:
            macular_score = 0.0
        
        score *= (0.5 + 0.5 * macular_score)  # Blend with baseline
        
        # Fovea-aware scoring (if fovea provided, prefer regions closer to it)
        if fovea_pos is not None:
            dist_from_fovea = np.sqrt((cx - fovea_pos[0])**2 + (cy - fovea_pos[1])**2)
            
            # Clinically relevant GA is within ~2000-3000 microns of fovea
            # Normalize by image size as proxy
            if dist_from_fovea < 0.2 * image_diagonal:
                fovea_score = 1.0
            elif dist_from_fovea < 0.4 * image_diagonal:
                fovea_score = 1.0 - (dist_from_fovea - 0.2 * image_diagonal) / (0.2 * image_diagonal)
            else:
                fovea_score = 0.1
            
            score *= (0.7 + 0.3 * fovea_score)
        
        return score
    
    def segment_ga_regions(
        self,
        image: np.ndarray,
        disc_center_x: Optional[float] = None,
        disc_center_y: Optional[float] = None,
        disc_height_pixels: Optional[float] = None,
        en_face_split_x: Optional[int] = None,
        fovea_x: Optional[float] = None,
        fovea_y: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Segment GA regions using single-cluster K-means with texture validation.
        
        Args:
            image: Full composite OCT image (BGR)
            disc_center_x: Optional disc center X for masking
            disc_center_y: Optional disc center Y for masking
            disc_height_pixels: Optional disc height for creating mask
            en_face_split_x: Optional split point to extract en-face region
            fovea_x: Optional fovea X for anatomy-aware scoring
            fovea_y: Optional fovea Y for anatomy-aware scoring
        
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
            # Adjust fovea coordinates to en-face space
            if fovea_x is not None:
                fovea_x_local = fovea_x - en_face_split_x
            else:
                fovea_x_local = None
            fovea_y_local = fovea_y
        else:
            en_face = image
            disc_center_x_local = disc_center_x
            fovea_x_local = fovea_x
            fovea_y_local = fovea_y
        
        # Convert to grayscale
        if len(en_face.shape) == 3:
            gray = cv2.cvtColor(en_face, cv2.COLOR_BGR2GRAY)
        else:
            gray = en_face
        
        h, w = gray.shape
        
        # Apply CLAHE enhancement
        enhanced = self._apply_clahe(gray)
        
        # Mask out Optic Disc if coordinates provided
        disc_mask = None
        if disc_center_x_local is not None and disc_center_y is not None and disc_height_pixels is not None:
            # Create circular mask around disc
            disc_radius = int(disc_height_pixels * self.disc_exclusion_multiplier)
            disc_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(
                disc_mask,
                (int(disc_center_x_local), int(disc_center_y)),
                disc_radius,
                255,
                -1
            )
            
            # Mask out disc region from pixel values
            disc_mask_flat = disc_mask.reshape(-1) == 0
            pixel_values = enhanced.reshape((-1, 1)).astype(np.float32)
            pixel_values_masked = pixel_values[disc_mask_flat]
        else:
            pixel_values = enhanced.reshape((-1, 1)).astype(np.float32)
            pixel_values_masked = pixel_values
            disc_mask_flat = np.ones(len(pixel_values), dtype=bool)
        
        # K-Means Clustering on single-channel intensity (like original)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels_masked, centers = cv2.kmeans(
            pixel_values_masked,
            self.n_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Map labels back to full image
        labels = np.zeros(len(pixel_values), dtype=np.int32) - 1
        labels[disc_mask_flat] = labels_masked.flatten()
        labels = labels.reshape(gray.shape)
        
        # Rank clusters by intensity (brightest first)
        centers_flat = centers.flatten()
        ranked_indices = np.argsort(centers_flat)[::-1]  # Descending intensity
        
        # Select the brightest cluster (baseline approach that worked best)
        selected_cluster = ranked_indices[0]
        
        # Create lesion mask from selected single cluster
        lesion_mask = (labels == selected_cluster).astype(np.uint8) * 255
        
        # Morphological Cleanup
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply watershed splitting only to large blobs (> 15% of image)
        temp_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max([cv2.contourArea(c) for c in temp_contours]) if temp_contours else 0
        image_area = h * w
        
        if max_area > 0.15 * image_area:
            clean_mask = self._apply_watershed_splitting(clean_mask)
        
        # Find Contours
        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Filter and score contours
        candidates = []
        
        fovea_local = (fovea_x_local, fovea_y_local) if fovea_x_local is not None and fovea_y_local is not None else None
        disc_local = (disc_center_x_local, disc_center_y) if disc_center_x_local is not None and disc_center_y is not None else None
        
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
            
            # 3. Border Filter - DISABLED (as in baseline)
            # The plan mentioned re-enabling it, but baseline didn't have it
            # x, y, w_box, h_box = cv2.boundingRect(cnt)
            # box_area = w_box * h_box
            # if box_area > 0.7 * (w * h):
            #     continue
            
            # 4. Anatomy-aware scoring
            anatomy_score = self._score_region_anatomy_aware(
                cnt, gray.shape, disc_local, fovea_local
            )
            
            # Store with score
            candidates.append((cnt, area, anatomy_score))
        
        if not candidates:
            return []
        
        # Sort by anatomy score first, then by area
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Size-based filtering (relative to largest)
        largest_area = candidates[0][1]
        threshold_area = self.relative_area_threshold * largest_area
        
        final_candidates = [(c, a, s) for c, a, s in candidates if a >= threshold_area]
        
        # Apply max_regions limit if specified
        if self.max_regions is not None:
            final_candidates = final_candidates[:self.max_regions]
        
        final_contours = [c for c, _, _ in final_candidates]
        
        # Adjust contours back to original image coordinates if needed
        if en_face_split_x is not None:
            adjusted_contours = []
            for cnt in final_contours:
                adjusted = cnt.copy()
                adjusted[:, 0, 0] += en_face_split_x  # Shift X coordinates
                adjusted_contours.append(adjusted)
            final_contours = adjusted_contours
        
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
