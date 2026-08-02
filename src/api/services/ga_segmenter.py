"""GA Segmentation Service - Single-cluster selection with texture validation"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from ..utils.logger import get_logger
from .sam_refiner import SAMRefiner

logger = get_logger("services.ga_segmenter")


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
        max_regions: Optional[int] = 8,
        disc_exclusion_multiplier: float = 0.6,
        clahe_clip_limit: float = 3.0,
        morph_kernel_size: int = 11,
        use_sam: bool = True,
        max_click_dist_fraction: float = 0.15,
    ):
        """
        Initialize GA segmentation service.
        
        Args:
            n_clusters: Number of K-means clusters (default: 3)
            min_area: Minimum contour area in pixels (default: 500)
            max_circularity: Maximum circularity to filter out circular objects (default: 0.8)
            relative_area_threshold: Keep regions >= this fraction of largest (default: 0.1)
            max_regions: Maximum number of regions to return (default: 8). Callers
                that take a minimum over all returned regions degrade as this
                grows, so returning everything is not a safe default.
            disc_exclusion_multiplier: Disc masking radius multiplier (default: 0.6)
            clahe_clip_limit: CLAHE clip limit (default: 3.0)
            morph_kernel_size: Morphological operations kernel size (default: 11)
            use_sam: Whether to refine K-means contours with SAM2 (default: True)
            max_click_dist_fraction: In local (click-driven) segmentation, reject
                candidate contours that the click falls outside of by more than
                this fraction of the crop radius. Clicks inside a contour are
                never rejected (default: 0.15)
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
        self.max_click_dist_fraction = max_click_dist_fraction
        self._sam = SAMRefiner() if use_sam else None
        # Keep top-N bright clusters to reduce failure when true GA is not brightest.
        self.top_cluster_count = min(max(1, n_clusters), 2)
        # Guardrails to suppress giant/border-connected false positives.
        self.max_bbox_fraction = 0.50
        self.max_region_fraction = 0.30
        self.border_margin_px = 3
        self.border_reject_area_fraction = 0.025
        # Absolute min_area (500 px) is ~0.015% of a typical en-face, small
        # enough to admit speckle. This floor scales with image size and sits
        # just below the smallest observed true lesion (0.09% of en-face).
        self.min_area_fraction = 0.0008
        # Log-normal size prior over lesion area as a fraction of the en-face,
        # fitted to regions that land on the ground-truth GA edge.
        self.ga_area_fraction_center = 0.014
        self.ga_area_fraction_log_sigma = 1.6
    
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
                                    image_shape: Tuple[int, int]) -> float:
        """
        Score a GA region based on anatomical likelihood.

        Deliberately takes no fovea argument: this score feeds region ranking,
        and ranking must stay independent of the fovea for the fovea-to-GA
        distance to be a measurement rather than a restatement of the prior.

        Args:
            contour: OpenCV contour
            image_shape: (height, width) of en-face image

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

        return score

    def _score_region_size(self, area: float, image_area: float) -> float:
        """Score a region by how closely its size matches a real GA lesion.

        Measured over the validation set: regions that actually sit on the
        ground-truth GA edge occupy a median 1.4% of the en-face (10th-90th
        percentile 0.09%-5.2%), whereas the median returned candidate occupies
        0.06% — i.e. most candidates are speckle, smaller than any true lesion.

        Scored log-normally, since lesion area spans two orders of magnitude.
        """
        if area <= 0 or image_area <= 0:
            return 0.0
        fraction = area / image_area
        log_ratio = np.log(fraction / self.ga_area_fraction_center)
        return float(np.exp(-0.5 * (log_ratio / self.ga_area_fraction_log_sigma) ** 2))

    def _score_region_appearance(
        self,
        enhanced: np.ndarray,
        contour: np.ndarray,
    ) -> float:
        """Score a region on whether it *looks* like atrophy.

        GA on en-face OCT is a window defect: loss of RPE lets more choroidal
        signal through, so the lesion reads brighter than the retina around it
        and is sharply demarcated. Both cues are computed against the region's
        own local surround rather than a global threshold, which keeps the score
        stable across scans with different overall brightness.
        """
        h, w = enhanced.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Surround = a ring just outside the lesion, excluding the lesion itself.
        ring_width = max(5, int(0.02 * np.sqrt(h * w)))
        kernel = np.ones((ring_width, ring_width), np.uint8)
        surround = cv2.dilate(mask, kernel, iterations=2)
        surround = cv2.subtract(surround, mask)

        inside = enhanced[mask > 0]
        outside = enhanced[surround > 0]
        if inside.size == 0 or outside.size == 0:
            return 0.0

        # Contrast: how much brighter the lesion is than its surround, in units
        # of the surround's own spread. ~2 SD is a confident window defect.
        spread = float(np.std(outside)) + 1e-6
        contrast = (float(np.mean(inside)) - float(np.mean(outside))) / spread
        contrast_score = float(np.clip(contrast / 2.0, 0.0, 1.0))

        # Demarcation: mean gradient magnitude along the boundary, normalised
        # against the image's own gradient scale.
        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge_vals = grad_mag[edge > 0]
        if edge_vals.size == 0:
            return contrast_score
        reference = float(np.percentile(grad_mag, 90)) + 1e-6
        edge_score = float(np.clip(float(np.mean(edge_vals)) / reference, 0.0, 1.0))

        return 0.6 * contrast_score + 0.4 * edge_score

    def _extract_cluster_contours(
        self,
        labels: np.ndarray,
        cluster_idx: int
    ) -> List[np.ndarray]:
        """Build cleaned contours for one cluster index."""
        h, w = labels.shape
        lesion_mask = (labels == cluster_idx).astype(np.uint8) * 255

        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        temp_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max([cv2.contourArea(c) for c in temp_contours]) if temp_contours else 0
        image_area = h * w
        if max_area > 0.15 * image_area:
            clean_mask = self._apply_watershed_splitting(clean_mask)

        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
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
        
        # Keep top bright clusters as candidates (mitigates cluster-mismatch failures).
        selected_clusters = ranked_indices[:min(self.top_cluster_count, len(ranked_indices))]
        cluster_rank_map: Dict[int, int] = {
            int(cluster_id): int(rank)
            for rank, cluster_id in enumerate(ranked_indices)
        }

        contour_records: List[Tuple[np.ndarray, int]] = []
        for cluster_idx in selected_clusters:
            contours = self._extract_cluster_contours(labels, int(cluster_idx))
            for cnt in contours:
                contour_records.append((cnt, int(cluster_idx)))

        if not contour_records:
            return []

        if self.use_sam and self._sam is not None and self._sam.available:
            boxes = []
            for cnt, _ in contour_records:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                boxes.append(np.array([x, y, x + w_box, y + h_box]))
            en_face_rgb = cv2.cvtColor(en_face, cv2.COLOR_BGR2RGB) if len(en_face.shape) == 3 else en_face
            self._sam.set_image(en_face_rgb)
            sam_results = self._sam.refine_candidates(boxes)
            if sam_results:
                contour_records = [(r["contour"], -1) for r in sam_results]

        # Filter and score contours
        candidates = []
        image_area = float(h * w)
        image_diagonal = float(np.sqrt(h**2 + w**2))
        fovea_local = (fovea_x_local, fovea_y_local) if fovea_x_local is not None and fovea_y_local is not None else None

        for cnt, cluster_idx in contour_records:
            area = cv2.contourArea(cnt)
            
            # 1. Size Filter
            if area < max(self.min_area, self.min_area_fraction * image_area):
                continue
            
            # 2. Circularity Filter (reject circles - likely blood vessels or disc)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > self.max_circularity:
                continue
            
            # 3. Border/giant-blob filter
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            box_area = float(w_box * h_box)
            if box_area > self.max_bbox_fraction * image_area:
                continue
            if area > self.max_region_fraction * image_area:
                continue

            touch_count = 0
            if x <= self.border_margin_px:
                touch_count += 1
            if y <= self.border_margin_px:
                touch_count += 1
            if (x + w_box) >= (w - self.border_margin_px):
                touch_count += 1
            if (y + h_box) >= (h - self.border_margin_px):
                touch_count += 1
            if touch_count >= 2 and area > self.border_reject_area_fraction * image_area:
                continue
            
            # 4. Scoring.
            #
            # Design rule: no term here may depend on the fovea position.
            # The measurand is the fovea-to-GA distance, so any term favouring
            # regions at a particular distance from the fovea makes the output a
            # function of the prior rather than the image. The previous version
            # scored candidates against `target_dist_norm = 0.10` of the image
            # diagonal, which drove the autonomous measurement to a near-constant
            # value (Bland-Altman difference-vs-mean slope -2.06, r = -0.34,
            # ICC = 0.000 on the holdout).
            #
            # What remains is evidence about the region itself: is it lesion-sized,
            # does it look like a window defect, and is it in the macula.
            anatomy_score = self._score_region_anatomy_aware(cnt, gray.shape)
            size_score = self._score_region_size(area, image_area)
            appearance_score = self._score_region_appearance(enhanced, cnt)

            rank = cluster_rank_map.get(cluster_idx, 0)
            intensity_prior = max(0.70, 1.0 - 0.15 * rank)
            selection_score = (
                (0.40 * size_score)
                + (0.35 * appearance_score)
                + (0.25 * anatomy_score)
            ) * intensity_prior

            # Retained for diagnostics only — never used for ranking.
            if fovea_local is not None:
                boundary_dist = abs(float(cv2.pointPolygonTest(
                    cnt, (float(fovea_local[0]), float(fovea_local[1])), True
                )))
            else:
                boundary_dist = float("nan")

            candidates.append((cnt, area, selection_score, boundary_dist))
        
        if not candidates:
            return []
        
        # Sort by composite score first, then area.
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Size-based filtering, relative to the largest surviving region.
        # Previously this bound was loosened to 2% of image area whenever a fovea
        # was supplied, which made it weaker rather than stronger and let ~98
        # regions through per image on average. Because the caller takes the
        # minimum distance over every returned region, each extra speckle could
        # only pull the measurement toward zero.
        largest_area = candidates[0][1]
        threshold_area = self.relative_area_threshold * largest_area
        final_candidates = [(c, a, s, d) for c, a, s, d in candidates if a >= threshold_area]

        # Drop candidates the evidence does not support at all.
        supported = [(c, a, s, d) for c, a, s, d in final_candidates if s >= 0.15]
        final_candidates = supported or [candidates[0]]


        # Apply max_regions limit if specified
        if self.max_regions is not None:
            final_candidates = final_candidates[:self.max_regions]
        
        final_contours = [c for c, _, _, _ in final_candidates]
        
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
        crop_radius_multiplier: float = 2.5,
        fovea_x: Optional[float] = None,
        fovea_y: Optional[float] = None,
        min_fovea_ga_dist_px: float = 12.0,
    ) -> List[np.ndarray]:
        """
        Segment GA region locally around a clicked point with relaxed parameters.

        Args:
            image: Full composite OCT image (BGR)
            click_x: X coordinate of user click (original image space)
            click_y: Y coordinate of user click (original image space)
            disc_center_x: Optional disc center X for masking and crop radius calculation
            disc_center_y: Optional disc center Y for masking
            disc_height_pixels: Optional disc height for crop radius calculation
            en_face_split_x: Optional split point to extract en-face region
            crop_radius_multiplier: Multiplier for crop radius (default: 2.5 * disc_height)
            fovea_x: Optional fovea X (original image space) for proximity filtering.
            fovea_y: Optional fovea Y (original image space) for proximity filtering.
            min_fovea_ga_dist_px: Minimum acceptable distance from contour to fovea.
                Candidate contours whose nearest vertex is closer than this are
                flagged as fovea-contaminated and skipped in favour of less-
                contaminated alternatives (if any).
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
        
        # SAM point-prompt fast path
        if self.use_sam and self._sam is not None and self._sam.available:
            en_face_rgb = cv2.cvtColor(en_face, cv2.COLOR_BGR2RGB) if len(en_face.shape) == 3 else en_face
            self._sam.set_image(en_face_rgb)
            result = self._sam.refine_point(point=(int(click_x_local), int(click_y_local)))
            if result is not None:
                contour = result["contour"]
                if en_face_split_x is not None:
                    adjusted = contour.copy()
                    adjusted[:, 0, 0] += en_face_split_x
                    return [adjusted]
                return [contour]

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
            logger.debug("GA-local empty crop at (%.1f, %.1f)", click_x_local, click_y_local)
            return []
        
        logger.debug(
            "GA-local click: (%.1f, %.1f), crop: [%s:%s, %s:%s], size: %s",
            click_x_local,
            click_y_local,
            x1,
            x2,
            y1,
            y2,
            gray_crop.shape,
        )
        
        # Apply CLAHE enhancement
        enhanced_crop = self._apply_clahe(gray_crop)
        
        # Mask out Optic Disc if coordinates provided and disc is in crop
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
            logger.debug("GA-local click on masked disc area")
            return []

        # Build candidate cluster order: click cluster first, then others ranked by
        # intensity proximity so we fall back to the most similar brightness class.
        click_intensity = float(centers[selected_cluster][0])
        other_clusters = [i for i in range(n_clusters_local) if i != selected_cluster]
        other_clusters.sort(key=lambda i: abs(float(centers[i][0]) - click_intensity))
        cluster_order = [selected_cluster] + other_clusters

        crop_area = (x2 - x1) * (y2 - y1)
        max_area_local = int(0.65 * crop_area)  # cap at 65% of crop; watershed splits blobs earlier
        # Proximity gate. The cluster fallback below walks every cluster in the
        # crop, so without a bound it can return a contour from a cluster with no
        # intensity relation to the click (e.g. a CLAHE edge artifact hundreds of
        # grey levels away). Candidates that the click falls this far outside are
        # not GA at the click and are dropped outright. The click may legitimately
        # sit just outside the contour — ground truth marks the GA *edge* — so the
        # gate bounds that gap rather than requiring containment.
        max_click_dist_local = self.max_click_dist_fraction * crop_radius
        min_area_local = 100
        kernel_size_local = 5
        kernel = np.ones((kernel_size_local, kernel_size_local), np.uint8)
        click_pt_crop = np.array([click_x_crop, click_y_crop])

        best_contour = None

        for candidate_cluster in cluster_order:
            logger.debug(
                "GA-local trying cluster %s (intensity: %.1f)",
                candidate_cluster,
                centers[candidate_cluster][0],
            )

            # Create lesion mask from candidate cluster
            lesion_mask = (labels_crop == candidate_cluster).astype(np.uint8) * 255

            # Relaxed morphological cleanup (smaller kernel)
            clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

            # Watershed splitting when the mask contains a large merged blob.
            # Use an 8% threshold (vs 15% for global) so that local blobs are split
            # more aggressively; this produces smaller per-region pieces that the
            # click-point selection below can pick from accurately.
            contours_pre, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_pre:
                max_pre_area = max(cv2.contourArea(c) for c in contours_pre)
                if max_pre_area > 0.08 * crop_area:
                    clean_mask = self._apply_watershed_splitting(clean_mask)

            # Find contours after potential watershed split
            contours_crop, _ = cv2.findContours(
                clean_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours_crop:
                logger.debug("GA-local cluster %s: no contours after morphology", candidate_cluster)
                continue

            # Filter: min area, max area cap, circularity, proximity to click
            valid_contours = []
            for cnt in contours_crop:
                area = cv2.contourArea(cnt)
                if area < min_area_local:
                    continue
                if area > max_area_local:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > self.max_circularity:
                    continue
                # Gate on how far *outside* the contour the click falls. A click
                # inside a region scores 0, so large lesions clicked near their
                # centre are never penalised for being far from their own edge.
                signed = cv2.pointPolygonTest(cnt, (int(click_x_crop), int(click_y_crop)), True)
                outside_dist = max(0.0, -float(signed))
                if outside_dist > max_click_dist_local:
                    continue
                pts = cnt.reshape(-1, 2)
                nearest_click_dist = float(
                    np.min(np.sqrt(np.sum((pts - click_pt_crop) ** 2, axis=1)))
                )
                valid_contours.append((cnt, nearest_click_dist, signed))

            if not valid_contours:
                logger.debug("GA-local cluster %s: no valid contours after filtering", candidate_cluster)
                continue

            # Pick the contour containing the click, or nearest to it.
            # Prefer contours that are not contaminated (i.e. not touching the fovea).
            def _nearest_to_fovea(cnt: np.ndarray) -> float:
                """Min distance from contour vertices to fovea (in en-face / crop space)."""
                if fovea_x is None or fovea_y is None:
                    return float("inf")
                # Convert fovea to crop-local coords
                fovea_local_x = fovea_x - (en_face_split_x or 0) - x1
                fovea_local_y = fovea_y - y1
                pts = cnt.reshape(-1, 2).astype(np.float64)
                fpt = np.array([fovea_local_x, fovea_local_y], dtype=np.float64)
                return float(np.min(np.linalg.norm(pts - fpt, axis=1)))

            clean_contours_with_meta = []
            for cnt, nearest_click_dist, dist_to_click in valid_contours:
                near_fovea = _nearest_to_fovea(cnt) < min_fovea_ga_dist_px
                contains_click = dist_to_click >= 0
                clean_contours_with_meta.append((cnt, contains_click, nearest_click_dist, near_fovea))

            # Sort: contains_click first, then not-near-fovea, then by click distance
            clean_contours_with_meta.sort(
                key=lambda t: (not t[1], t[3], t[2])
            )

            for cnt, contains_click, nearest_dist, near_fovea in clean_contours_with_meta:
                if near_fovea:
                    logger.debug(
                        "GA-local cluster %s: contour closest to fovea < %.0f px, skipping",
                        candidate_cluster, min_fovea_ga_dist_px,
                    )
                    continue
                best_contour = cnt
                break

            # Fallback: if all contours were near fovea, use the one closest to click
            if best_contour is None and clean_contours_with_meta:
                best_contour = clean_contours_with_meta[0][0]
                logger.debug(
                    "GA-local cluster %s: all contours near fovea, using closest to click (fallback)",
                    candidate_cluster,
                )

            if best_contour is not None:
                logger.debug(
                    "GA-local cluster %s (intensity %.1f): found region area %.0f px²",
                    candidate_cluster,
                    float(centers[candidate_cluster][0]),
                    cv2.contourArea(best_contour),
                )
                break

            # Reset before trying next cluster
            best_contour = None

        if best_contour is None:
            logger.debug("GA-local no contour found near click across all clusters")
            return []
        
        # Adjust contour back to en-face coordinates
        adjusted = best_contour.copy()
        adjusted[:, 0, 0] += x1
        adjusted[:, 0, 1] += y1
        
        # Adjust back to original image coordinates if needed
        if en_face_split_x is not None:
            adjusted[:, 0, 0] += en_face_split_x
        
        logger.debug("GA-local found region area: %.0f px²", cv2.contourArea(best_contour))
        
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
