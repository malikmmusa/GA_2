"""Helpers for consistent API response shaping."""

from typing import Any, Dict, List, Tuple

from ..models.schemas import (
    DistanceCalculationResponse,
    GASegmentationResponse,
    ProgressionCalculationResponse,
)


def build_ga_segmentation_response(
    regions: List[List[Tuple[int, int]]]
) -> GASegmentationResponse:
    return GASegmentationResponse(
        regions=regions,
        region_count=len(regions),
    )


def build_distance_response(payload: Dict[str, Any]) -> DistanceCalculationResponse:
    return DistanceCalculationResponse(**payload)


def build_progression_response(payload: Dict[str, Any]) -> ProgressionCalculationResponse:
    return ProgressionCalculationResponse(**payload)
