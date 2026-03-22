"""GA segmentation endpoint."""
from fastapi import APIRouter, File, UploadFile, Query
from typing import Optional

from ..dependencies import get_ga_segmenter
from ..models.schemas import GASegmentationResponse
from ..utils.errors import route_error_handler
from ..utils.responses import build_ga_segmentation_response
from ..utils.uploads import decode_uploaded_image

router = APIRouter()

@router.post("/segment-ga", response_model=GASegmentationResponse)
@route_error_handler("GA segmentation")
async def segment_ga(
    file: UploadFile = File(...),
    disc_center_x: Optional[float] = Query(None),
    disc_center_y: Optional[float] = Query(None),
    disc_height_pixels: Optional[float] = Query(None),
    en_face_split_x: Optional[int] = Query(None),
    fovea_x: Optional[float] = Query(None),
    fovea_y: Optional[float] = Query(None)
) -> GASegmentationResponse:
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
    image = await decode_uploaded_image(file, file_role="image")

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

    return build_ga_segmentation_response(regions)


@router.post("/segment-ga-local", response_model=GASegmentationResponse)
@route_error_handler("Local GA segmentation")
async def segment_ga_local(
    file: UploadFile = File(...),
    click_x: float = Query(..., description="X coordinate of user click"),
    click_y: float = Query(..., description="Y coordinate of user click"),
    disc_center_x: Optional[float] = Query(None),
    disc_center_y: Optional[float] = Query(None),
    disc_height_pixels: Optional[float] = Query(None),
    en_face_split_x: Optional[int] = Query(None),
    fovea_x: Optional[float] = Query(None, description="Fovea X for proximity filtering"),
    fovea_y: Optional[float] = Query(None, description="Fovea Y for proximity filtering"),
) -> GASegmentationResponse:
    """
    Segment GA region locally around a clicked point (fallback for missed regions).
    """
    image = await decode_uploaded_image(file, file_role="image")

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
        en_face_split_x=en_face_split_x,
        fovea_x=fovea_x,
        fovea_y=fovea_y,
    )

    # Convert to JSON-serializable format
    regions = segmenter.contours_to_json(contours)

    return build_ga_segmentation_response(regions)
