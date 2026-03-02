"""Distance and progression calculation services."""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger("services.calculator")

AVERAGE_DAYS_PER_MONTH = 30.44
FLOAT_TOLERANCE = 1e-6


class DistanceCalculatorService:
    """
    Service for calculating distances between fovea and GA regions.
    Uses pixel-to-micron conversion based on the 1800 micron optic disc standard.
    """

    @staticmethod
    def _validate_distance_inputs(
        ga_region: Sequence[Tuple[int, int]],
        pixel_to_micron_ratio: float,
    ) -> None:
        if not ga_region:
            raise ValueError("Selected GA region is empty")
        if pixel_to_micron_ratio <= 0 or not math.isfinite(pixel_to_micron_ratio):
            raise ValueError("Invalid pixel-to-micron ratio")

    def calculate_fovea_to_ga_distance(
        self,
        fovea_x: float,
        fovea_y: float,
        ga_region: List[Tuple[int, int]],
        pixel_to_micron_ratio: float,
    ) -> Dict[str, float]:
        """
        Calculate the shortest distance from fovea to a GA region.
        """
        self._validate_distance_inputs(ga_region, pixel_to_micron_ratio)

        fovea_pt = np.array([fovea_x, fovea_y], dtype=np.float64)
        ga_points = np.asarray(ga_region, dtype=np.float64)

        distances = np.linalg.norm(ga_points - fovea_pt, axis=1)
        min_idx = int(np.argmin(distances))
        min_distance_pixels = float(distances[min_idx])
        nearest_point = ga_points[min_idx]
        min_distance_microns = min_distance_pixels * pixel_to_micron_ratio

        logger.debug(
            "Fovea to GA: %.1f px = %.1f µm",
            min_distance_pixels,
            min_distance_microns,
        )

        return {
            "distance_pixels": min_distance_pixels,
            "distance_microns": float(min_distance_microns),
            "nearest_ga_point_x": int(nearest_point[0]),
            "nearest_ga_point_y": int(nearest_point[1]),
        }

    def calculate_distances_to_all_regions(
        self,
        fovea_x: float,
        fovea_y: float,
        ga_regions: List[List[Tuple[int, int]]],
        pixel_to_micron_ratio: float,
    ) -> List[Dict[str, float]]:
        """
        Calculate distances from fovea to all GA regions.
        """
        results: List[Dict[str, float]] = []
        for i, region in enumerate(ga_regions):
            distance_info = self.calculate_fovea_to_ga_distance(
                fovea_x=fovea_x,
                fovea_y=fovea_y,
                ga_region=region,
                pixel_to_micron_ratio=pixel_to_micron_ratio,
            )
            distance_info["region_index"] = i
            results.append(distance_info)

        results.sort(key=lambda item: item["distance_microns"])
        return results


class ProgressionCalculatorService:
    """
    Service for calculating GA progression rate and predicting foveal involvement.
    """

    @staticmethod
    def _build_error_result(message: str, *, days_elapsed: int = 0) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_message": message,
            "days_elapsed": days_elapsed,
            "distance_change_microns": 0.0,
            "rate_microns_per_day": None,
            "rate_microns_per_month": None,
            "predicted_foveal_involvement_date": None,
        }

    @staticmethod
    def _is_valid_distance(value: float) -> bool:
        return math.isfinite(value) and value >= 0

    def calculate_progression(
        self,
        date_before: str,
        date_after: str,
        distance_before_microns: float,
        distance_after_microns: float,
        eye_side_before: str,
        eye_side_after: str,
    ) -> Dict[str, Any]:
        """
        Calculate GA progression rate and predict foveal involvement date.
        """
        if eye_side_before != eye_side_after:
            return self._build_error_result(
                f"Eye mismatch: Before is {eye_side_before}, After is {eye_side_after}"
            )

        if not self._is_valid_distance(distance_before_microns) or not self._is_valid_distance(
            distance_after_microns
        ):
            return self._build_error_result("Distances must be finite, non-negative values")

        try:
            dt_before = datetime.fromisoformat(date_before)
            dt_after = datetime.fromisoformat(date_after)
        except ValueError as exc:
            return self._build_error_result(f"Invalid date format: {str(exc)}")

        if dt_after <= dt_before:
            return self._build_error_result("After date must be later than before date")

        days_elapsed = (dt_after - dt_before).days
        if days_elapsed <= 0:
            return self._build_error_result("Date range must span at least one day")

        distance_change = distance_before_microns - distance_after_microns
        logger.debug("Time elapsed: %s days", days_elapsed)
        logger.debug("Distance change: %.3f µm", distance_change)

        if distance_change > FLOAT_TOLERANCE:
            rate_per_day = distance_change / days_elapsed
            if rate_per_day <= 0 or not math.isfinite(rate_per_day):
                return self._build_error_result(
                    "Unable to compute a valid progression rate",
                    days_elapsed=days_elapsed,
                )

            rate_per_month = rate_per_day * AVERAGE_DAYS_PER_MONTH
            days_to_fovea = max(0.0, distance_after_microns / rate_per_day)
            predicted_foveal_involvement_date = None

            try:
                predicted_date = dt_after + timedelta(days=days_to_fovea)
                predicted_foveal_involvement_date = predicted_date.strftime("%Y-%m-%d")
            except OverflowError:
                logger.warning("Predicted foveal involvement date overflow for %.3f days", days_to_fovea)

            return {
                "status": "progression",
                "error_message": None,
                "days_elapsed": days_elapsed,
                "distance_change_microns": round(distance_change, 2),
                "rate_microns_per_day": round(rate_per_day, 3),
                "rate_microns_per_month": round(rate_per_month, 2),
                "predicted_foveal_involvement_date": predicted_foveal_involvement_date,
            }

        if math.isclose(distance_change, 0.0, abs_tol=FLOAT_TOLERANCE):
            logger.info("No progression detected")
            return {
                "status": "no_progression",
                "error_message": None,
                "days_elapsed": days_elapsed,
                "distance_change_microns": 0.0,
                "rate_microns_per_day": 0.0,
                "rate_microns_per_month": 0.0,
                "predicted_foveal_involvement_date": None,
            }

        logger.warning("Negative progression detected (%.3f µm)", distance_change)
        return {
            "status": "error",
            "error_message": (
                f"Negative progression detected ({distance_change:.1f} µm). "
                "GA appears further from fovea. Check measurements."
            ),
            "days_elapsed": days_elapsed,
            "distance_change_microns": round(distance_change, 2),
            "rate_microns_per_day": None,
            "rate_microns_per_month": None,
            "predicted_foveal_involvement_date": None,
        }
