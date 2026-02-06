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
    en_face_split_x: Optional[int] = Query(None)
) -> Dict:
    """
    Segment Geographic Atrophy (GA) regions using K-means clustering.
    
    The algorithm uses:
    - CLAHE contrast enhancement
    - K-means clustering (3 clusters)
    - Morphological cleanup
    - Size, circularity, and location filtering
    
    Optionally masks out the optic disc if coordinates provided.
    
    Args:
        file: Uploaded OCT image
        disc_center_x: Optional disc center X for masking
        disc_center_y: Optional disc center Y for masking
        disc_height_pixels: Optional disc height for masking
        en_face_split_x: Optional split point to extract en-face region
    
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
            detail=f"GA segmentation failed: {str(e)}"
        )
