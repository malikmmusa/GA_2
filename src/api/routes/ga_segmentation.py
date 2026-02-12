"""GA segmentation endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Dict, Optional

from ..services.ga_segmenter import GASegmenterService
from ..models.schemas import GASegmentationResponse

router = APIRouter()

# Initialize service (singleton)
ga_segmenter = None

def get_ga_segmenter() -> GASegmenterService:
    """Get or initialize the GA segmenter service (singleton)."""
    global ga_segmenter
    if ga_segmenter is None:
        ga_segmenter = GASegmenterService()
    return ga_segmenter

@router.post("/segment-ga", response_model=GASegmentationResponse)
async def segment_ga(
    file: UploadFile = File(...),
    disc_center_x: Optional[float] = Query(None),
    disc_center_y: Optional[float] = Query(None),
    disc_height_pixels: Optional[float] = Query(None),
    en_face_split_x: Optional[int] = Query(None),
    fovea_x: Optional[float] = Query(None),
    fovea_y: Optional[float] = Query(None)
) -> Dict:
    """
    Segment Geographic Atrophy (GA) regions using multi-feature K-means clustering.
    
    The algorithm uses:
    - Multi-channel features (intensity + texture + contrast)
    - K-means clustering (5 clusters with GA likelihood scoring)
    - Morphological cleanup with watershed splitting
    - Anatomy-aware region scoring
    
    Optionally masks out the optic disc and uses fovea for anatomy-aware scoring.
    
    Args:
        file: Uploaded OCT image
        disc_center_x: Optional disc center X for masking
        disc_center_y: Optional disc center Y for masking
        disc_height_pixels: Optional disc height for masking
        en_face_split_x: Optional split point to extract en-face region
        fovea_x: Optional fovea X for anatomy-aware scoring
        fovea_y: Optional fovea Y for anatomy-aware scoring
    
    Returns:
        GASegmentationResponse with list of GA region contours
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Get segmenter
        segmenter = get_ga_segmenter()
        
        # Segment GA regions
        contours = segmenter.segment_ga_regions(
            image=image,
            disc_center_x=disc_center_x,
            disc_center_y=disc_center_y,
            disc_height_pixels=disc_height_pixels,
            en_face_split_x=en_face_split_x,
            fovea_x=fovea_x,
            fovea_y=fovea_y
        )
        
        # Convert to JSON-serializable format
        regions = segmenter.contours_to_json(contours)
        
        return {
            'regions': regions,
            'region_count': len(regions)
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GA segmentation failed: {str(e)}"
        )


@router.post("/segment-ga-local", response_model=GASegmentationResponse)
async def segment_ga_local(
    file: UploadFile = File(...),
    click_x: float = Query(..., description="X coordinate of user click"),
    click_y: float = Query(..., description="Y coordinate of user click"),
    disc_center_x: Optional[float] = Query(None),
    disc_center_y: Optional[float] = Query(None),
    disc_height_pixels: Optional[float] = Query(None),
    en_face_split_x: Optional[int] = Query(None)
) -> Dict:
    """
    Segment GA region locally around a clicked point (fallback for missed regions).
    
    Uses relaxed clustering parameters and selects the cluster that the clicked pixel
    belongs to. This is more permissive than global segmentation and helps catch
    GA regions that were filtered out in the global pass.
    
    Args:
        file: Uploaded OCT image
        click_x: X coordinate of user click (original image space)
        click_y: Y coordinate of user click (original image space)
        disc_center_x: Optional disc center X for masking and crop radius
        disc_center_y: Optional disc center Y for masking
        disc_height_pixels: Optional disc height for crop radius calculation
        en_face_split_x: Optional split point to extract en-face region
    
    Returns:
        GASegmentationResponse with 0 or 1 region
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Get segmenter
        segmenter = get_ga_segmenter()
        
        # Segment GA region locally
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y,
            disc_center_x=disc_center_x,
            disc_center_y=disc_center_y,
            disc_height_pixels=disc_height_pixels,
            en_face_split_x=en_face_split_x
        )
        
        # Convert to JSON-serializable format
        regions = segmenter.contours_to_json(contours)
        
        return {
            'regions': regions,
            'region_count': len(regions)
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local GA segmentation failed: {str(e)}"
        )
