"""Optic disc detection endpoint."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Dict

from ..services.disc_detector import DiscDetectorService
from ..models.schemas import DiscDetectionResponse

router = APIRouter()

# Initialize service (singleton pattern)
disc_detector = None

def get_disc_detector() -> DiscDetectorService:
    """Get or initialize the disc detector service (singleton)."""
    global disc_detector
    if disc_detector is None:
        disc_detector = DiscDetectorService()
    return disc_detector

@router.post("/detect-disc", response_model=DiscDetectionResponse)
async def detect_optic_disc(file: UploadFile = File(...)) -> Dict:
    """
    Detect optic disc in an uploaded OCT image.
    
    The endpoint accepts a composite OCT image (B-scan + En-face),
    automatically splits it, and returns the optic disc coordinates
    with a vertical line representing 1800 microns.
    
    Args:
        file: Uploaded image file (PNG, JPG, etc.)
    
    Returns:
        DiscDetectionResponse with coordinates and conversion factors
    
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Could not decode image."
            )
        
        # Get disc detector service
        detector = get_disc_detector()
        
        # Perform detection
        result = detector.detect_from_image(image, image_name=file.filename)
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Disc detection failed: {str(e)}"
        )

@router.get("/disc-detector/status")
async def get_disc_detector_status():
    """
    Check if the disc detector service is initialized.
    
    Returns:
        Status information about the disc detector service
    """
    detector = get_disc_detector()
    return {
        "status": "ready",
        "model_path": detector.model_path,
        "device": str(detector.device),
        "img_size": detector.img_size
    }
