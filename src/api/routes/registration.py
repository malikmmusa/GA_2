"""Image registration endpoint for landmark transfer between temporal images."""
from fastapi import APIRouter, File, UploadFile, Form

from ..dependencies import get_registrar
from ..models.schemas import (
    ImageRegistrationRequest,
    ImageRegistrationResponse,
    RegistrarStatusResponse,
)
from ..utils.errors import route_error_handler
from ..utils.logger import get_logger
from ..utils.request_parsers import parse_form_json
from ..utils.status import build_status_payload
from ..utils.uploads import decode_uploaded_image

router = APIRouter()
logger = get_logger("routes.registration")


@router.post("/register-images", response_model=ImageRegistrationResponse)
@route_error_handler("Image registration")
async def register_images(
    file_reference: UploadFile = File(..., description="Reference image (Image 1 / Before)"),
    file_new: UploadFile = File(..., description="New image to register (Image 2 / After)"),
    request_data: str = Form(..., description="JSON string with registration parameters")
) -> ImageRegistrationResponse:
    """
    Register two OCT images and transfer landmarks from reference to new image.
    
    This endpoint uses vessel-based feature matching to align two en-face OCT images
    and transfer confirmed landmarks (fovea, disc) from the reference image to the
    new image coordinate space.
    
    Args:
        file_reference: Reference image file (Image 1 with confirmed landmarks)
        file_new: New image file to register (Image 2)
        request_data: JSON string containing:
            - en_face_split_x_ref: En-face region start X in reference image
            - en_face_split_x_new: En-face region start X in new image
            - fovea_x: Confirmed fovea X coordinate in reference image
            - fovea_y: Confirmed fovea Y coordinate in reference image
            - disc_center_x (optional): Disc center X in reference image
            - disc_center_y (optional): Disc center Y in reference image
    
    Returns:
        ImageRegistrationResponse with transformed landmarks and confidence metrics
    
    """
    params = parse_form_json(request_data, ImageRegistrationRequest)

    img_ref = await decode_uploaded_image(file_reference, file_role="reference image")
    img_new = await decode_uploaded_image(file_new, file_role="new image")

    logger.debug("Reference image: %s, New image: %s", img_ref.shape, img_new.shape)
    logger.debug("Fovea in reference: (%.1f, %.1f)", params.fovea_x, params.fovea_y)

    # Get registrar service
    service = get_registrar()

    # Perform registration
    matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
        img_ref,
        img_new,
        params.en_face_split_x_ref,
        params.en_face_split_x_new
    )

    # If registration failed, return failure response
    if matrix is None or status == "failed":
        return ImageRegistrationResponse(
            transformed_fovea_x=params.fovea_x,  # Return original coordinates as fallback
            transformed_fovea_y=params.fovea_y,
            transformed_disc_center_x=params.disc_center_x,
            transformed_disc_center_y=params.disc_center_y,
            en_face_split_x_ref=params.en_face_split_x_ref,
            en_face_split_x_new=params.en_face_split_x_new,
            confidence=confidence,
            num_matches=num_matches,
            num_inliers=num_inliers,
            status=status,
            message=message
        )

    # Transform landmarks using the registration matrix
    transformed = service.transform_landmarks(
        matrix,
        params.fovea_x,
        params.fovea_y,
        params.en_face_split_x_ref,
        params.en_face_split_x_new,
        params.disc_center_x,
        params.disc_center_y
    )

    # Flatten 2x3 matrix to 6 floats [a, b, tx, c, d, ty] for client-side use
    matrix_flat = [float(value) for value in matrix.reshape(-1)]

    logger.debug(
        "Transformed fovea: (%.1f, %.1f)",
        transformed["transformed_fovea_x"],
        transformed["transformed_fovea_y"],
    )
    logger.info("Registration status: %s, confidence: %.2f", status, confidence)

    return ImageRegistrationResponse(
        transformed_fovea_x=transformed['transformed_fovea_x'],
        transformed_fovea_y=transformed['transformed_fovea_y'],
        transformed_disc_center_x=transformed['transformed_disc_center_x'],
        transformed_disc_center_y=transformed['transformed_disc_center_y'],
        transform_matrix=matrix_flat,
        en_face_split_x_ref=params.en_face_split_x_ref,
        en_face_split_x_new=params.en_face_split_x_new,
        confidence=confidence,
        num_matches=num_matches,
        num_inliers=num_inliers,
        status=status,
        message=message
    )


@router.get("/registrar/status", response_model=RegistrarStatusResponse)
async def get_registrar_status() -> RegistrarStatusResponse:
    """
    Check if the image registrar service is initialized.
    
    Returns:
        Status information about the registrar service
    """
    service = get_registrar()
    return RegistrarStatusResponse(
        **build_status_payload(
            "ready",
            n_features=service.n_features,
            ratio_test_threshold=service.ratio_test_threshold,
            ransac_threshold=service.ransac_threshold,
            min_inliers_success=service.min_inliers_success,
            min_inliers_low_confidence=service.min_inliers_low_confidence,
        )
    )
