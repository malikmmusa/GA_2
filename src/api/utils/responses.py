"""Helpers for consistent API response shaping."""

from typing import Any, Dict, List, Tuple

from ..models.schemas import (
    DistanceCalculationResponse,
    GASegmentationResponse,
    ProgressionCalculationResponse,
)


def build_ga_segmentation_response(
    regions: List[List[Tuple[int, int]]],
    confidence: float = 1.0,
    auto_measurement_reliable: bool = True,
) -> GASegmentationResponse:
    """Build the GA response.

    Confidence defaults to 1.0/reliable so click-driven local segmentation, where
    the user has already told us where the lesion is, is never flagged.
    """
    return GASegmentationResponse(
        regions=regions,
        region_count=len(regions),
        confidence=confidence,
        auto_measurement_reliable=auto_measurement_reliable,
    )


def build_distance_response(payload: Dict[str, Any]) -> DistanceCalculationResponse:
    return DistanceCalculationResponse(**payload)


def build_progression_response(payload: Dict[str, Any]) -> ProgressionCalculationResponse:
    return ProgressionCalculationResponse(**payload)
