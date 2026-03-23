"""Optic disc detection endpoint."""
from fastapi import APIRouter, File, UploadFile

from ..dependencies import get_disc_detector
from ..models.schemas import DiscDetectionResponse, DiscDetectorStatusResponse
from ..utils.errors import route_error_handler
from ..utils.status import build_status_payload
from ..utils.uploads import decode_uploaded_image

router = APIRouter()

@router.post("/detect-disc", response_model=DiscDetectionResponse)
@route_error_handler("Disc detection")
async def detect_optic_disc(file: UploadFile = File(...)) -> DiscDetectionResponse:
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
    image = await decode_uploaded_image(file, file_role="image")

    # Get disc detector service
    detector = get_disc_detector()

    # Perform detection
    result = detector.detect_from_image(image, image_name=file.filename)
    return DiscDetectionResponse(**result)

@router.get("/disc-detector/status", response_model=DiscDetectorStatusResponse)
async def get_disc_detector_status() -> DiscDetectorStatusResponse:
    """
    Check if the disc detector service is initialized.
    
    Returns:
        Status information about the disc detector service
    """
    detector = get_disc_detector()
    return DiscDetectorStatusResponse(
        **build_status_payload(
            "ready" if detector.model is not None else "fallback",
            model_loaded=detector.model is not None,
            model_path=detector.loaded_model_path or detector.model_path,
            device=str(detector.device),
            img_size=detector.img_size,
        )
    )
