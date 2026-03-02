"""Fovea detection endpoint."""
from fastapi import APIRouter, File, UploadFile, Form

from ..dependencies import get_fovea_detector
from ..models.schemas import FoveaDetectionRequest, FoveaDetectionResponse
from ..utils.errors import route_error_handler
from ..utils.request_parsers import parse_form_json
from ..utils.uploads import decode_uploaded_image

router = APIRouter()

@router.post("/detect-fovea", response_model=FoveaDetectionResponse)
@route_error_handler("Fovea detection")
async def detect_fovea(
    file: UploadFile = File(...),
    request_data: str = Form(...)
) -> FoveaDetectionResponse:
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
    request = parse_form_json(request_data, FoveaDetectionRequest)

    image = await decode_uploaded_image(file, file_role="image")

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

    return FoveaDetectionResponse(**result)
