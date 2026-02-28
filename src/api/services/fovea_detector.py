"""Fovea Detection Service - Refactored from image_utils.py"""
from typing import Any, Dict

import numpy as np

from src.utils.image_utils import find_fovea_anatomy_aware, manual_fovea_adjustment

from ..utils.logger import get_logger

logger = get_logger("services.fovea_detector")


class FoveaDetectorService:
    """
    Service for detecting fovea location using anatomy-aware methods.
    Preserves the multi-strategy approach from image_utils.py:
    1. Green Line Anchor (Precise Y) -> Scan Line X
    2. Geometric Fallback (Approx Y) -> Scan Band X -> Refine Y
    3. Manual Adjustment (Interactive)
    """
    
    def detect_fovea(
        self,
        image: np.ndarray,
        disc_center_x: float,
        disc_center_y: float,
        disc_height_pixels: float,
        en_face_split_x: int,
        use_manual_adjustment: bool = False,
        image_name: str = "Image"
    ) -> Dict[str, Any]:
        """
        Detect fovea location using anatomy-aware logic.
        
        Args:
            image: Full composite OCT image (BGR)
            disc_center_x: X coordinate of disc center in original image
            disc_center_y: Y coordinate of disc center in original image
            disc_height_pixels: Height of disc in pixels (vertical line length)
            en_face_split_x: X coordinate where en-face region starts
            use_manual_adjustment: Enable interactive manual adjustment UI
            image_name: Name for display in manual adjustment window
        
        Returns:
            Dictionary containing:
                - fovea_x: X coordinate in original image
                - fovea_y: Y coordinate in original image
                - detection_method: Method used (green_line, geometric_fallback, manual)
                - eye_side: OD or OS
        """
        # Calculate geometric estimate for fallback
        height, width = image.shape[:2]
        en_face_width = width - en_face_split_x
        en_face_center_x = en_face_split_x + (en_face_width / 2)
        
        # Determine eye side based on disc position
        if disc_center_x > en_face_center_x:
            eye_side = "OS"  # Left eye (disc on right side of en-face)
            est_x = disc_center_x + (2.5 * disc_height_pixels)
        else:
            eye_side = "OD"  # Right eye (disc on left side of en-face)
            est_x = disc_center_x - (2.5 * disc_height_pixels)
        
        est_y = disc_center_y + (0.15 * disc_height_pixels)
        
        # Try anatomy-aware detection
        anatomy_result = find_fovea_anatomy_aware(
            image,
            en_face_split_x,
            est_x=est_x,
            est_y=est_y
        )
        
        if anatomy_result:
            initial_guess, method_name = anatomy_result
            logger.debug("%s found at %s", method_name, initial_guess)
            
            # Map method name to simpler labels
            if "Green Line" in method_name:
                detection_method = "green_line"
            elif "Geometric" in method_name:
                detection_method = "geometric_fallback"
            else:
                detection_method = "anatomy_aware"
        else:
            # Absolute fallback (rare)
            logger.warning("Using raw geometry fallback")
            initial_guess = (int(est_x), int(est_y))
            detection_method = "raw_geometry"
        
        # Manual adjustment if requested
        if use_manual_adjustment:
            adjusted = manual_fovea_adjustment(
                image,
                initial_guess,
                f"Fovea: {image_name} [{detection_method}]"
            )
            if adjusted:
                fovea_x, fovea_y = adjusted
                detection_method = "manual"
            else:
                # User cancelled, use auto-detected
                fovea_x, fovea_y = initial_guess
        else:
            fovea_x, fovea_y = initial_guess
        
        return {
            'fovea_x': float(fovea_x),
            'fovea_y': float(fovea_y),
            'detection_method': detection_method,
            'eye_side': eye_side
        }
    
    def validate_fovea_location(
        self,
        fovea_x: float,
        fovea_y: float,
        disc_center_x: float,
        disc_center_y: float,
        disc_height_pixels: float,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Validate that fovea location is anatomically plausible.
        
        Args:
            fovea_x, fovea_y: Fovea coordinates
            disc_center_x, disc_center_y: Disc center coordinates
            disc_height_pixels: Disc height in pixels
            image_width, image_height: Image dimensions
        
        Returns:
            True if location is plausible, False otherwise
        """
        # Check bounds
        if not (0 <= fovea_x < image_width and 0 <= fovea_y < image_height):
            return False
        
        # Check distance from disc (should be 2-3 disc diameters away)
        distance = np.hypot(fovea_x - disc_center_x, fovea_y - disc_center_y)
        
        min_distance = 1.5 * disc_height_pixels
        max_distance = 4.0 * disc_height_pixels
        
        if not (min_distance <= distance <= max_distance):
            logger.warning(
                "Fovea distance from disc is %.1f px (expected %.1f - %.1f px)",
                distance,
                min_distance,
                max_distance,
            )
            return False
        
        return True
