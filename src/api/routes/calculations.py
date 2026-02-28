"""Distance and progression calculation endpoints."""
from fastapi import APIRouter, HTTPException

from ..dependencies import get_distance_calculator, get_progression_calculator
from ..models.schemas import (
    DistanceCalculationRequest,
    DistanceCalculationResponse,
    ProgressionCalculationRequest,
    ProgressionCalculationResponse
)
from ..utils.errors import route_error_handler
from ..utils.responses import build_distance_response, build_progression_response

router = APIRouter()

@router.post("/calculate-distance", response_model=DistanceCalculationResponse)
@route_error_handler("Distance calculation")
async def calculate_distance(
    request: DistanceCalculationRequest
) -> DistanceCalculationResponse:
    """
    Calculate the shortest distance from fovea to a selected GA region.
    
    Uses the optic disc as the anatomical reference (1800 microns)
    to convert pixel distances to microns.
    
    Args:
        request: Distance calculation parameters
    
    Returns:
        DistanceCalculationResponse with distance in pixels and microns
    """
    calculator = get_distance_calculator()

    if request.selected_ga_region_index >= len(request.ga_regions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region index: {request.selected_ga_region_index}"
        )

    selected_region = request.ga_regions[request.selected_ga_region_index]
    if not selected_region:
        raise HTTPException(
            status_code=400,
            detail=f"GA region {request.selected_ga_region_index} is empty",
        )

    result = calculator.calculate_fovea_to_ga_distance(
        fovea_x=request.fovea_x,
        fovea_y=request.fovea_y,
        ga_region=selected_region,
        pixel_to_micron_ratio=request.pixel_to_micron_ratio
    )

    return build_distance_response(result)

@router.post("/calculate-progression", response_model=ProgressionCalculationResponse)
@route_error_handler("Progression calculation")
async def calculate_progression(
    request: ProgressionCalculationRequest
) -> ProgressionCalculationResponse:
    """
    Calculate GA progression rate and predict foveal involvement date.
    
    Compares before and after images to determine:
    - Time elapsed between images
    - Distance change (should be positive for progression)
    - Rate of progression (microns/day and microns/month)
    - Predicted date when GA will reach the fovea
    
    Returns an error if:
    - Images are from different eyes (OD vs OS mismatch)
    - Negative progression detected (GA appears further away)
    - Dates are invalid
    
    Args:
        request: Progression calculation parameters
    
    Returns:
        ProgressionCalculationResponse with rate and prediction
    """
    calculator = get_progression_calculator()

    result = calculator.calculate_progression(
        date_before=request.date_before,
        date_after=request.date_after,
        distance_before_microns=request.distance_before_microns,
        distance_after_microns=request.distance_after_microns,
        eye_side_before=request.eye_side_before,
        eye_side_after=request.eye_side_after
    )

    return build_progression_response(result)
