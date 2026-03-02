"""Image Registration Service - Vessel-based landmark transfer between temporal OCT images"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from ..utils.logger import get_logger

logger = get_logger("services.image_registrar")


class ImageRegistrarService:
    """
    Service for registering two en-face OCT images using vessel landmarks.
    
    Uses ORB keypoint detection and affine registration to align images and
    transfer landmarks (fovea, disc) from a reference image to a new image.
    
    Approach:
    - Extract en-face regions from both composite images
    - Enhance vessel contrast with CLAHE
    - Detect ORB keypoints on vessel maps
    - Match features with BFMatcher + Lowe ratio test
    - Compute affine transform with RANSAC
    - Transform landmarks and compute confidence
    """
    
    def __init__(
        self,
        n_features: int = 5000,
        ratio_test_threshold: float = 0.75,
        ransac_threshold: float = 5.0,
        min_inliers_success: int = 15,
        min_inliers_low_confidence: int = 8,
        confidence_inlier_ratio_high: float = 0.4,
        confidence_inlier_ratio_low: float = 0.2,
        clahe_clip_limit: float = 3.0
    ):
        """
        Initialize image registration service.
        
        Args:
            n_features: Number of ORB features to detect (default: 5000)
            ratio_test_threshold: Lowe's ratio test threshold (default: 0.75)
            ransac_threshold: RANSAC reprojection error threshold in pixels (default: 5.0)
            min_inliers_success: Minimum inliers for "success" status (default: 15)
            min_inliers_low_confidence: Minimum inliers for "low_confidence" status (default: 8)
            confidence_inlier_ratio_high: Inlier ratio threshold for high confidence (default: 0.4)
            confidence_inlier_ratio_low: Inlier ratio threshold for low confidence (default: 0.2)
            clahe_clip_limit: CLAHE clip limit for contrast enhancement (default: 3.0)
        """
        self.n_features = n_features
        self.ratio_test_threshold = ratio_test_threshold
        self.ransac_threshold = ransac_threshold
        self.min_inliers_success = min_inliers_success
        self.min_inliers_low_confidence = min_inliers_low_confidence
        self.confidence_inlier_ratio_high = confidence_inlier_ratio_high
        self.confidence_inlier_ratio_low = confidence_inlier_ratio_low
        self.clahe_clip_limit = clahe_clip_limit
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.n_features)
        
        # Initialize BFMatcher with Hamming distance (for ORB descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _enhance_vessels(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance vessel contrast using CLAHE.
        
        Args:
            gray: Grayscale image
            
        Returns:
            CLAHE-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _detect_and_match(
        self,
        img_ref: np.ndarray,
        img_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Detect keypoints and match features between two images.
        
        Args:
            img_ref: Reference image (grayscale)
            img_new: New image to register (grayscale)
            
        Returns:
            Tuple of (src_points, dst_points, num_good_matches)
            src_points: Nx2 array of matched points in reference image
            dst_points: Nx2 array of matched points in new image
            num_good_matches: Number of good matches after ratio test
        """
        # Enhance vessel contrast
        enhanced_ref = self._enhance_vessels(img_ref)
        enhanced_new = self._enhance_vessels(img_new)
        
        # Detect keypoints and compute descriptors
        kp_ref, des_ref = self.orb.detectAndCompute(enhanced_ref, None)
        kp_new, des_new = self.orb.detectAndCompute(enhanced_new, None)
        
        if des_ref is None or des_new is None:
            logger.warning("No descriptors found in one or both images")
            return np.array([]), np.array([]), 0
        
        if len(kp_ref) < 4 or len(kp_new) < 4:
            logger.warning("Not enough keypoints: ref=%s, new=%s", len(kp_ref), len(kp_new))
            return np.array([]), np.array([]), 0
        
        # Match descriptors using kNN (k=2 for ratio test)
        try:
            matches = self.matcher.knnMatch(des_ref, des_new, k=2)
        except cv2.error as e:
            logger.error("Matching failed: %s", e)
            return np.array([]), np.array([]), 0
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_test_threshold * n.distance:
                    good_matches.append(m)
        
        logger.debug(
            "Total matches: %s, good matches after ratio test: %s",
            len(matches),
            len(good_matches),
        )
        
        if len(good_matches) < 4:
            logger.warning("Not enough good matches (need >= 4)")
            return np.array([]), np.array([]), 0
        
        # Extract matched point coordinates
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
        
        return src_pts, dst_pts, len(good_matches)
    
    def _compute_affine_transform(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Compute affine transform using RANSAC.
        
        Args:
            src_pts: Nx2 array of source points
            dst_pts: Nx2 array of destination points
            
        Returns:
            Tuple of (transform_matrix, num_inliers)
            transform_matrix: 2x3 affine transformation matrix (or None if failed)
            num_inliers: Number of inlier matches after RANSAC
        """
        if len(src_pts) < 3 or len(dst_pts) < 3:
            return None, 0
        
        # Estimate partial affine transform (4 DOF: rotation, translation, uniform scale)
        # This is more robust for OCT images from the same device
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=0.99,
            maxIters=2000
        )
        
        if matrix is None:
            logger.warning("RANSAC failed to find a transform")
            return None, 0
        
        num_inliers = np.sum(inliers) if inliers is not None else 0
        logger.debug("RANSAC: %s inliers out of %s matches", num_inliers, len(src_pts))
        
        return matrix, int(num_inliers)
    
    def _compute_confidence(
        self,
        num_good_matches: int,
        num_inliers: int
    ) -> Tuple[float, str, str]:
        """
        Compute registration confidence and status.
        
        Args:
            num_good_matches: Total number of good feature matches
            num_inliers: Number of inliers after RANSAC
            
        Returns:
            Tuple of (confidence_score, status, message)
            confidence_score: Float between 0.0 and 1.0
            status: "success", "low_confidence", or "failed"
            message: Human-readable description
        """
        if num_good_matches == 0:
            return 0.0, "failed", "No feature matches found between images"
        
        inlier_ratio = num_inliers / num_good_matches
        
        # High confidence
        if (inlier_ratio >= self.confidence_inlier_ratio_high and 
            num_inliers >= self.min_inliers_success):
            confidence = min(0.8 + (inlier_ratio - 0.4) * 0.5, 1.0)
            return (
                confidence,
                "success",
                f"High confidence registration ({num_inliers} inliers, {inlier_ratio:.1%} ratio)"
            )
        
        # Low confidence
        elif (inlier_ratio >= self.confidence_inlier_ratio_low and 
              num_inliers >= self.min_inliers_low_confidence):
            confidence = 0.4 + (inlier_ratio - 0.2) * 2.0  # Scale 0.2-0.4 ratio to 0.4-0.8 confidence
            confidence = min(confidence, 0.79)  # Cap below "success" threshold
            return (
                confidence,
                "low_confidence",
                f"Moderate confidence registration ({num_inliers} inliers, {inlier_ratio:.1%} ratio). Verify landmark positions."
            )
        
        # Failed
        else:
            confidence = min(inlier_ratio, 0.39)  # Below low_confidence threshold
            return (
                confidence,
                "failed",
                f"Registration failed ({num_inliers} inliers, {inlier_ratio:.1%} ratio). Using independent detection."
            )
    
    def register_images(
        self,
        img_ref: np.ndarray,
        img_new: np.ndarray,
        en_face_split_x_ref: int,
        en_face_split_x_new: int
    ) -> Tuple[Optional[np.ndarray], float, int, int, str, str]:
        """
        Register two composite OCT images using their en-face regions.
        
        Args:
            img_ref: Reference composite image (BGR)
            img_new: New composite image to register (BGR)
            en_face_split_x_ref: X coordinate where en-face starts in reference image
            en_face_split_x_new: X coordinate where en-face starts in new image
            
        Returns:
            Tuple of (matrix, confidence, num_matches, num_inliers, status, message)
            matrix: 2x3 affine transformation matrix (or None if failed)
            confidence: Confidence score 0.0-1.0
            num_matches: Number of good feature matches
            num_inliers: Number of RANSAC inliers
            status: "success", "low_confidence", or "failed"
            message: Human-readable status message
        """
        if not (0 <= en_face_split_x_ref < img_ref.shape[1]) or not (
            0 <= en_face_split_x_new < img_new.shape[1]
        ):
            return (
                None,
                0.0,
                0,
                0,
                "failed",
                "Invalid en-face split coordinates for one or both images",
            )

        # Extract en-face regions
        en_face_ref = img_ref[:, en_face_split_x_ref:]
        en_face_new = img_new[:, en_face_split_x_new:]
        if en_face_ref.size == 0 or en_face_new.size == 0:
            return None, 0.0, 0, 0, "failed", "Unable to extract en-face regions from one or both images"
        
        # Convert to grayscale
        if len(en_face_ref.shape) == 3:
            gray_ref = cv2.cvtColor(en_face_ref, cv2.COLOR_BGR2GRAY)
        else:
            gray_ref = en_face_ref
        
        if len(en_face_new.shape) == 3:
            gray_new = cv2.cvtColor(en_face_new, cv2.COLOR_BGR2GRAY)
        else:
            gray_new = en_face_new
        
        logger.debug("En-face shapes: ref=%s, new=%s", gray_ref.shape, gray_new.shape)
        
        # Detect and match keypoints
        src_pts, dst_pts, num_good_matches = self._detect_and_match(gray_ref, gray_new)
        
        if num_good_matches < 4:
            confidence, status, message = self._compute_confidence(num_good_matches, 0)
            return None, confidence, num_good_matches, 0, status, message
        
        # Compute affine transform
        matrix, num_inliers = self._compute_affine_transform(src_pts, dst_pts)
        
        # Compute confidence
        confidence, status, message = self._compute_confidence(num_good_matches, num_inliers)
        
        return matrix, confidence, num_good_matches, num_inliers, status, message
    
    def transform_landmarks(
        self,
        matrix: np.ndarray,
        fovea_x: float,
        fovea_y: float,
        en_face_split_x_ref: int,
        en_face_split_x_new: int,
        disc_center_x: Optional[float] = None,
        disc_center_y: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Transform landmark coordinates using the registration matrix.
        
        The matrix transforms points in en-face local coordinates. We need to:
        1. Convert reference image landmarks from original coords to en-face local coords
        2. Apply the transform
        3. Convert back to new image original coords
        
        Args:
            matrix: 2x3 affine transformation matrix
            fovea_x: Fovea X in reference image original coordinates
            fovea_y: Fovea Y in reference image original coordinates
            en_face_split_x_ref: En-face split X in reference image
            en_face_split_x_new: En-face split X in new image
            disc_center_x: Optional disc center X in reference image original coordinates
            disc_center_y: Optional disc center Y in reference image original coordinates
            
        Returns:
            Dictionary with transformed coordinates in new image original space:
            - transformed_fovea_x, transformed_fovea_y
            - transformed_disc_center_x, transformed_disc_center_y (if disc provided)
        """
        result = {}
        
        # Transform fovea
        # Convert to en-face local coordinates
        fovea_local_x = fovea_x - en_face_split_x_ref
        fovea_local_y = fovea_y
        
        # Apply affine transform
        fovea_point = np.array([[fovea_local_x, fovea_local_y]], dtype=np.float32)
        transformed_fovea = cv2.transform(fovea_point.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        
        # Convert back to original image coordinates
        result['transformed_fovea_x'] = float(transformed_fovea[0, 0] + en_face_split_x_new)
        result['transformed_fovea_y'] = float(transformed_fovea[0, 1])
        
        # Transform disc center if provided
        if disc_center_x is not None and disc_center_y is not None:
            disc_local_x = disc_center_x - en_face_split_x_ref
            disc_local_y = disc_center_y
            
            disc_point = np.array([[disc_local_x, disc_local_y]], dtype=np.float32)
            transformed_disc = cv2.transform(disc_point.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            
            result['transformed_disc_center_x'] = float(transformed_disc[0, 0] + en_face_split_x_new)
            result['transformed_disc_center_y'] = float(transformed_disc[0, 1])
        else:
            result['transformed_disc_center_x'] = None
            result['transformed_disc_center_y'] = None
        
        return result
