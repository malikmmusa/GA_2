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
        
        # Determine eye side based on disc position.
        # Anatomically: the fovea is always temporal to the disc.
        # OD (right eye): disc is nasal = left side → fovea is right → est_x = disc_x + offset
        # OS (left eye):  disc is nasal = right side → fovea is left → est_x = disc_x - offset
        if disc_center_x > en_face_center_x:
            eye_side = "OS"  # Left eye (disc on right side of en-face)
            est_x = disc_center_x - (2.5 * disc_height_pixels)
        else:
            eye_side = "OD"  # Right eye (disc on left side of en-face)
            est_x = disc_center_x + (2.5 * disc_height_pixels)
        
        est_y = disc_center_y + (0.15 * disc_height_pixels)
        
        # Try anatomy-aware detection
        anatomy_result = find_fovea_anatomy_aware(
            image,
            en_face_split_x,
            est_x=est_x,
            est_y=est_y,
            disc_local_x=disc_center_x - en_face_split_x,
            eye_side=eye_side,
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

        # Anatomical validation with opposite-side retry
        if not self.validate_fovea_location(
            float(fovea_x), float(fovea_y),
            disc_center_x, disc_center_y,
            disc_height_pixels, width, height
        ):
            logger.warning(
                "%s: initial fovea (%.0f, %.0f) failed validation; trying opposite eye side",
                image_name, fovea_x, fovea_y,
            )
            # Flip eye side and recompute estimate
            if eye_side == "OS":
                retry_est_x = disc_center_x + (2.5 * disc_height_pixels)
                retry_eye_side = "OD"
            else:
                retry_est_x = disc_center_x - (2.5 * disc_height_pixels)
                retry_eye_side = "OS"

            # Clamp retry estimate to the en-face region so we don't search in B-scan space
            en_face_left = float(en_face_split_x)
            en_face_right = float(width - 1)
            retry_est_x = float(np.clip(retry_est_x, en_face_left, en_face_right))

            retry_result = find_fovea_anatomy_aware(
                image, en_face_split_x, est_x=retry_est_x, est_y=est_y,
                disc_local_x=disc_center_x - en_face_split_x,
                eye_side=retry_eye_side,
            )
            if retry_result:
                retry_guess, _ = retry_result
            else:
                retry_guess = (int(retry_est_x), int(est_y))

            rx, ry = float(retry_guess[0]), float(retry_guess[1])
            if self.validate_fovea_location(
                rx, ry, disc_center_x, disc_center_y,
                disc_height_pixels, width, height
            ):
                logger.info(
                    "%s: opposite-side retry passed validation (%.0f, %.0f) eye=%s",
                    image_name, rx, ry, retry_eye_side,
                )
                fovea_x, fovea_y = rx, ry
                eye_side = retry_eye_side
                detection_method = detection_method  # retry succeeded; keep base method
            else:
                # Both sides failed validation — use a geometric estimate (disc_x ±
                # 2.5*disc_h) clamped to the en-face region.  Try both OD and OS
                # candidate positions; for each that falls within the image (not
                # clamped to the edge), sample the local mean intensity.  The foveal
                # pit is characteristically darker than surrounding retina, so the
                # darker valid candidate is picked.  If only one candidate is within
                # bounds, or intensities are tied, fall back to the heuristic eye_side.
                est_x_od = disc_center_x + (2.5 * disc_height_pixels)
                est_x_os = disc_center_x - (2.5 * disc_height_pixels)
                fallback_y = float(np.clip(est_y, 0, height - 1))

                # A candidate is "valid" if its raw estimate lies within the en-face
                # region with at least a 50-px margin from each side (enough for the
                # 11-px sampling window and to exclude black background at image edges).
                ef_left = float(en_face_split_x) + 50.0
                ef_right = float(width) - 50.0
                od_valid = ef_left <= est_x_od <= ef_right
                os_valid = ef_left <= est_x_os <= ef_right

                def _local_intensity(x: float, y: float) -> float:
                    """Mean grayscale intensity in an 11×11 neighbourhood."""
                    import cv2 as _cv2
                    ef_img = image[:, en_face_split_x:, :]
                    lx = int(np.clip(x - en_face_split_x, 5, ef_img.shape[1] - 6))
                    ly = int(np.clip(y, 5, ef_img.shape[0] - 6))
                    patch = ef_img[ly - 5:ly + 6, lx - 5:lx + 6]
                    gray = _cv2.cvtColor(patch, _cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
                    return float(np.mean(gray))

                if od_valid and os_valid:
                    intens_od = _local_intensity(est_x_od, fallback_y)
                    intens_os = _local_intensity(est_x_os, fallback_y)
                    # Minimum tissue intensity threshold: black background (scan edge)
                    # has ~0 intensity and must not beat a genuine tissue region.
                    MIN_TISSUE_INT = 15.0
                    od_tissue = intens_od >= MIN_TISSUE_INT
                    os_tissue = intens_os >= MIN_TISSUE_INT
                    if od_tissue and os_tissue:
                        # Both are real tissue; pick the darker (more pit-like)
                        if intens_od <= intens_os:
                            fovea_x, chosen_side = float(np.clip(est_x_od, ef_left, ef_right)), "OD"
                        else:
                            fovea_x, chosen_side = float(np.clip(est_x_os, ef_left, ef_right)), "OS"
                    elif od_tissue:
                        fovea_x, chosen_side = float(np.clip(est_x_od, ef_left, ef_right)), "OD"
                    elif os_tissue:
                        fovea_x, chosen_side = float(np.clip(est_x_os, ef_left, ef_right)), "OS"
                    else:
                        # Neither is tissue; fall back to heuristic eye_side
                        fovea_x = float(np.clip(est_x, en_face_split_x, width - 1))
                        chosen_side = eye_side
                    logger.warning(
                        "%s: both sides failed; darker-side fallback (%.0f, %.0f)"
                        " side=%s [OD_int=%.1f OS_int=%.1f]",
                        image_name, fovea_x, fallback_y, chosen_side, intens_od, intens_os,
                    )
                elif od_valid:
                    fovea_x, chosen_side = float(np.clip(est_x_od, ef_left, ef_right)), "OD"
                    logger.warning(
                        "%s: both sides failed; OD-only fallback (%.0f, %.0f)",
                        image_name, fovea_x, fallback_y,
                    )
                elif os_valid:
                    fovea_x, chosen_side = float(np.clip(est_x_os, ef_left, ef_right)), "OS"
                    logger.warning(
                        "%s: both sides failed; OS-only fallback (%.0f, %.0f)",
                        image_name, fovea_x, fallback_y,
                    )
                else:
                    # Both out of bounds — keep original est_x from heuristic eye_side
                    fovea_x = float(np.clip(est_x, en_face_split_x, width - 1))
                    chosen_side = eye_side
                    logger.warning(
                        "%s: both sides failed + both OOB; heuristic fallback (%.0f, %.0f)",
                        image_name, fovea_x, fallback_y,
                    )
                fovea_y = fallback_y
                detection_method = "geometric_fallback"

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
        
        # Check distance from disc (papillofoveal distance ≈ 2.2 disc diameters)
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
