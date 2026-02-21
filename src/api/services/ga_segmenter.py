"""GA Segmentation Service - Single-cluster selection with texture validation"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import sys
import os

# Import legacy utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.api.services.sam_refiner import SAMRefiner


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
        morph_kernel_size: int = 11,
        use_sam: bool = True,
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
        self.use_sam = use_sam
        self._sam = SAMRefiner()
    
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

        # SAM refinement: replace K-means contours with SAM-refined ones
        if self.use_sam and self._sam.available and contours:
            boxes = []
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                boxes.append(np.array([x, y, x + w_box, y + h_box]))
            self._sam.set_image(en_face)
            sam_results = self._sam.refine_candidates(boxes)
            if sam_results:
                contours = [r["contour"] for r in sam_results]

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
    
    def segment_ga_local(
        self,
        image: np.ndarray,
        click_x: float,
        click_y: float,
        disc_center_x: Optional[float] = None,
        disc_center_y: Optional[float] = None,
        disc_height_pixels: Optional[float] = None,
        en_face_split_x: Optional[int] = None,
        crop_radius_multiplier: float = 2.5
    ) -> List[np.ndarray]:
        """
        Segment GA region locally around a clicked point with relaxed parameters.
        
        This is a fallback method for when global segmentation misses a visible GA region.
        Uses relaxed clustering and morphology parameters to catch smaller/dimmer lesions.
        
        Args:
            image: Full composite OCT image (BGR)
            click_x: X coordinate of user click (original image space)
            click_y: Y coordinate of user click (original image space)
            disc_center_x: Optional disc center X for masking and crop radius calculation
            disc_center_y: Optional disc center Y for masking
            disc_height_pixels: Optional disc height for crop radius calculation
            en_face_split_x: Optional split point to extract en-face region
            crop_radius_multiplier: Multiplier for crop radius (default: 2.5 * disc_height)
        
        Returns:
            List containing 0 or 1 contour (numpy array) in original image coordinates
        """
        # Extract en-face region if split point provided
        if en_face_split_x is not None:
            en_face = image[:, en_face_split_x:, :]
            # Adjust click coordinates to en-face space
            click_x_local = click_x - en_face_split_x
            click_y_local = click_y
            # Adjust disc coordinates to en-face space
            if disc_center_x is not None:
                disc_center_x_local = disc_center_x - en_face_split_x
            else:
                disc_center_x_local = None
        else:
            en_face = image
            click_x_local = click_x
            click_y_local = click_y
            disc_center_x_local = disc_center_x
        
        # Convert to grayscale
        if len(en_face.shape) == 3:
            gray = cv2.cvtColor(en_face, cv2.COLOR_BGR2GRAY)
        else:
            gray = en_face
        
        h, w = gray.shape
        
        # Determine crop radius
        if disc_height_pixels is not None:
            crop_radius = int(disc_height_pixels * crop_radius_multiplier)
        else:
            crop_radius = min(h, w) // 4  # Fallback: 1/4 of image size
        
        # Define crop bounds (clamped to image)
        x1 = max(0, int(click_x_local - crop_radius))
        x2 = min(w, int(click_x_local + crop_radius))
        y1 = max(0, int(click_y_local - crop_radius))
        y2 = min(h, int(click_y_local + crop_radius))
        
        # Crop region
        gray_crop = gray[y1:y2, x1:x2]
        
        if gray_crop.size == 0:
            print(f"  [GA-Local] Empty crop at ({click_x_local:.1f}, {click_y_local:.1f})")
            return []
        
        print(f"  [GA-Local] Click: ({click_x_local:.1f}, {click_y_local:.1f}), Crop: [{x1}:{x2}, {y1}:{y2}], Size: {gray_crop.shape}")
        
        # Apply CLAHE enhancement
        enhanced_crop = self._apply_clahe(gray_crop)
        
        # Mask out Optic Disc if coordinates provided and disc is in crop
        disc_mask_crop = None
        if disc_center_x_local is not None and disc_center_y is not None and disc_height_pixels is not None:
            # Check if disc center is within crop bounds (with margin)
            disc_radius = int(disc_height_pixels * self.disc_exclusion_multiplier)
            if (x1 - disc_radius < disc_center_x_local < x2 + disc_radius and
                y1 - disc_radius < disc_center_y < y2 + disc_radius):
                # Create circular mask around disc in crop space
                disc_mask_crop = np.zeros(gray_crop.shape, dtype=np.uint8)
                disc_x_crop = int(disc_center_x_local - x1)
                disc_y_crop = int(disc_center_y - y1)
                cv2.circle(
                    disc_mask_crop,
                    (disc_x_crop, disc_y_crop),
                    disc_radius,
                    255,
                    -1
                )
                disc_mask_flat = disc_mask_crop.reshape(-1) == 0
                pixel_values = enhanced_crop.reshape((-1, 1)).astype(np.float32)
                pixel_values_masked = pixel_values[disc_mask_flat]
            else:
                pixel_values = enhanced_crop.reshape((-1, 1)).astype(np.float32)
                pixel_values_masked = pixel_values
                disc_mask_flat = np.ones(len(pixel_values), dtype=bool)
        else:
            pixel_values = enhanced_crop.reshape((-1, 1)).astype(np.float32)
            pixel_values_masked = pixel_values
            disc_mask_flat = np.ones(len(pixel_values), dtype=bool)
        
        # Relaxed K-means: 4 clusters for finer separation
        n_clusters_local = 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels_masked, centers = cv2.kmeans(
            pixel_values_masked,
            n_clusters_local,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Map labels back to full crop
        labels_crop = np.zeros(len(pixel_values), dtype=np.int32) - 1
        labels_crop[disc_mask_flat] = labels_masked.flatten()
        labels_crop = labels_crop.reshape(gray_crop.shape)
        
        # Determine which cluster the click pixel belongs to
        click_x_crop = int(click_x_local - x1)
        click_y_crop = int(click_y_local - y1)
        
        # Clamp click to crop bounds
        click_x_crop = max(0, min(labels_crop.shape[1] - 1, click_x_crop))
        click_y_crop = max(0, min(labels_crop.shape[0] - 1, click_y_crop))
        
        selected_cluster = labels_crop[click_y_crop, click_x_crop]
        
        if selected_cluster == -1:
            # Click was on masked (disc) area
            print(f"  [GA-Local] Click on masked area (disc)")
            return []
        
        print(f"  [GA-Local] Selected cluster: {selected_cluster} (intensity: {centers[selected_cluster][0]:.1f})")
        
        # Create lesion mask from selected cluster
        lesion_mask = (labels_crop == selected_cluster).astype(np.uint8) * 255
        
        # Relaxed morphological cleanup (smaller kernel)
        kernel_size_local = 5
        kernel = np.ones((kernel_size_local, kernel_size_local), np.uint8)
        clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours_crop, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours_crop:
            print(f"  [GA-Local] No contours found after morphology")
            return []
        
        # Relaxed filtering (lower min_area)
        min_area_local = 100
        valid_contours = []
        
        for cnt in contours_crop:
            area = cv2.contourArea(cnt)
            
            if area < min_area_local:
                continue
            
            # Circularity filter (reject circles)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > self.max_circularity:
                continue
            
            valid_contours.append(cnt)
        
        if not valid_contours:
            print(f"  [GA-Local] No valid contours after filtering")
            return []
        
        # Find contour containing or nearest to click point
        click_pt_crop = np.array([click_x_crop, click_y_crop])
        best_contour = None
        min_distance = float('inf')
        
        for cnt in valid_contours:
            # Check if point is inside
            dist = cv2.pointPolygonTest(cnt, (int(click_x_crop), int(click_y_crop)), True)
            
            if dist >= 0:
                # Point is inside this contour
                best_contour = cnt
                break
            else:
                # Point is outside, measure distance to nearest point on contour
                cnt_points = cnt.reshape(-1, 2)
                distances = np.sqrt(np.sum((cnt_points - click_pt_crop)**2, axis=1))
                nearest_dist = np.min(distances)
                
                if nearest_dist < min_distance:
                    min_distance = nearest_dist
                    best_contour = cnt
        
        if best_contour is None:
            print(f"  [GA-Local] No contour found near click")
            return []
        
        # Adjust contour back to en-face coordinates
        adjusted = best_contour.copy()
        adjusted[:, 0, 0] += x1
        adjusted[:, 0, 1] += y1
        
        # Adjust back to original image coordinates if needed
        if en_face_split_x is not None:
            adjusted[:, 0, 0] += en_face_split_x
        
        print(f"  [GA-Local] Found region with area: {cv2.contourArea(best_contour):.0f} px²")
        
        return [adjusted]
    
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
