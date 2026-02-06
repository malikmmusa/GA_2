"""Distance and progression calculation endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict

from ..services.calculator import DistanceCalculatorService, ProgressionCalculatorService
from ..models.schemas import (
    DistanceCalculationRequest,
    DistanceCalculationResponse,
    ProgressionCalculationRequest,
    ProgressionCalculationResponse
)

router = APIRouter()

# Initialize services (singletons)
distance_calculator = None
progression_calculator = None

def get_distance_calculator() -> DistanceCalculatorService:
    """Get or initialize the distance calculator service (singleton)."""
    global distance_calculator
    if distance_calculator is None:
        distance_calculator = DistanceCalculatorService()
    return distance_calculator

def get_progression_calculator() -> ProgressionCalculatorService:
    """Get or initialize the progression calculator service (singleton)."""
    global progression_calculator
    if progression_calculator is None:
        progression_calculator = ProgressionCalculatorService()
    return progression_calculator

@router.post("/calculate-distance", response_model=DistanceCalculationResponse)
async def calculate_distance(request: DistanceCalculationRequest) -> Dict:
    """
    Calculate the shortest distance from fovea to a selected GA region.
    
    Uses the optic disc as the anatomical reference (1800 microns)
    to convert pixel distances to microns.
    
    Args:
        request: Distance calculation parameters
    
    Returns:
        DistanceCalculationResponse with distance in pixels and microns
    """
    try:
        calculator = get_distance_calculator()
        
        # Validate region index
        if request.selected_ga_region_index < 0 or request.selected_ga_region_index >= len(request.ga_regions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid region index: {request.selected_ga_region_index}"
            )
        
        # Get selected region
        selected_region = request.ga_regions[request.selected_ga_region_index]
        
        # Calculate distance
        result = calculator.calculate_fovea_to_ga_distance(
            fovea_x=request.fovea_x,
            fovea_y=request.fovea_y,
            ga_region=selected_region,
            pixel_to_micron_ratio=request.pixel_to_micron_ratio
        )
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Distance calculation failed: {str(e)}"
        )

@router.post("/calculate-progression", response_model=ProgressionCalculationResponse)
async def calculate_progression(request: ProgressionCalculationRequest) -> Dict:
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
    try:
        calculator = get_progression_calculator()
        
        result = calculator.calculate_progression(
            date_before=request.date_before,
            date_after=request.date_after,
            distance_before_microns=request.distance_before_microns,
            distance_after_microns=request.distance_after_microns,
            eye_side_before=request.eye_side_before,
            eye_side_after=request.eye_side_after
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Progression calculation failed: {str(e)}"
        )
