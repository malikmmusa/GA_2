"""Fovea detection endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
from typing import Dict

from ..services.fovea_detector import FoveaDetectorService
from ..models.schemas import FoveaDetectionRequest, FoveaDetectionResponse

router = APIRouter()

# Initialize service (singleton)
fovea_detector = None

def get_fovea_detector() -> FoveaDetectorService:
    """Get or initialize the fovea detector service (singleton)."""
    global fovea_detector
    if fovea_detector is None:
        fovea_detector = FoveaDetectorService()
    return fovea_detector

@router.post("/detect-fovea", response_model=FoveaDetectionResponse)
async def detect_fovea(
    file: UploadFile = File(...),
    request_data: str = Form(...)
) -> Dict:
    """
    Detect fovea location in an OCT image.
    
    Uses anatomy-aware detection methods:
    1. Green Line Anchor (if scan line present)
    2. Geometric Fallback (based on disc position)
    3. Manual Adjustment (optional interactive mode)
    
    Args:
        file: Uploaded OCT image
        request_data: JSON string containing detection parameters including disc coordinates
    
    Returns:
        FoveaDetectionResponse with fovea coordinates and detection method
    """
    try:
        # Parse JSON string into Pydantic model
        try:
            request_dict = json.loads(request_data)
            request = FoveaDetectionRequest(**request_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in request_data: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request_data format: {str(e)}"
            )
        
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Get detector
        detector = get_fovea_detector()
        
        # Detect fovea
        result = detector.detect_fovea(
            image=image,
            disc_center_x=request.disc_center_x,
            disc_center_y=request.disc_center_y,
            disc_height_pixels=request.disc_height_pixels,
            en_face_split_x=request.en_face_split_x,
            use_manual_adjustment=request.use_manual_adjustment,
            image_name=file.filename or "Image"
        )
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fovea detection failed: {str(e)}"
        )
