"""Image registration endpoint for landmark transfer between temporal images."""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
from typing import Dict, Optional

from ..services.image_registrar import ImageRegistrarService
from ..models.schemas import ImageRegistrationResponse

router = APIRouter()

# Initialize service (singleton pattern)
registrar = None

def get_registrar() -> ImageRegistrarService:
    """Get or initialize the image registrar service (singleton)."""
    global registrar
    if registrar is None:
        registrar = ImageRegistrarService()
    return registrar


@router.post("/register-images", response_model=ImageRegistrationResponse)
async def register_images(
    file_reference: UploadFile = File(..., description="Reference image (Image 1 / Before)"),
    file_new: UploadFile = File(..., description="New image to register (Image 2 / After)"),
    request_data: str = Form(..., description="JSON string with registration parameters")
) -> Dict:
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
    
    Raises:
        HTTPException: If image processing or registration fails
    """
    try:
        # Parse request data
        try:
            params = json.loads(request_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in request_data: {str(e)}"
            )
        
        # Validate required parameters
        required_fields = ['en_face_split_x_ref', 'en_face_split_x_new', 'fovea_x', 'fovea_y']
        missing_fields = [f for f in required_fields if f not in params]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        en_face_split_x_ref = int(params['en_face_split_x_ref'])
        en_face_split_x_new = int(params['en_face_split_x_new'])
        fovea_x = float(params['fovea_x'])
        fovea_y = float(params['fovea_y'])
        disc_center_x = float(params['disc_center_x']) if 'disc_center_x' in params else None
        disc_center_y = float(params['disc_center_y']) if 'disc_center_y' in params else None
        
        # Read uploaded files
        contents_ref = await file_reference.read()
        contents_new = await file_new.read()
        
        # Decode images
        nparr_ref = np.frombuffer(contents_ref, np.uint8)
        img_ref = cv2.imdecode(nparr_ref, cv2.IMREAD_COLOR)
        
        nparr_new = np.frombuffer(contents_new, np.uint8)
        img_new = cv2.imdecode(nparr_new, cv2.IMREAD_COLOR)
        
        if img_ref is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid reference image file. Could not decode image."
            )
        
        if img_new is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid new image file. Could not decode image."
            )
        
        print(f"[Registration API] Reference image: {img_ref.shape}, New image: {img_new.shape}")
        print(f"[Registration API] Fovea in reference: ({fovea_x:.1f}, {fovea_y:.1f})")
        
        # Get registrar service
        service = get_registrar()
        
        # Perform registration
        matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
            img_ref,
            img_new,
            en_face_split_x_ref,
            en_face_split_x_new
        )
        
        # If registration failed, return failure response
        if matrix is None or status == "failed":
            return ImageRegistrationResponse(
                transformed_fovea_x=fovea_x,  # Return original coordinates as fallback
                transformed_fovea_y=fovea_y,
                transformed_disc_center_x=disc_center_x,
                transformed_disc_center_y=disc_center_y,
                confidence=confidence,
                num_matches=num_matches,
                num_inliers=num_inliers,
                status=status,
                message=message
            )
        
        # Transform landmarks using the registration matrix
        transformed = service.transform_landmarks(
            matrix,
            fovea_x,
            fovea_y,
            en_face_split_x_ref,
            en_face_split_x_new,
            disc_center_x,
            disc_center_y
        )
        
        # Flatten 2x3 matrix to 6 floats [a, b, tx, c, d, ty] for client-side use
        matrix_flat = [float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]),
                       float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])]
        
        print(f"[Registration API] Transformed fovea: ({transformed['transformed_fovea_x']:.1f}, {transformed['transformed_fovea_y']:.1f})")
        print(f"[Registration API] Status: {status}, Confidence: {confidence:.2f}")
        
        return ImageRegistrationResponse(
            transformed_fovea_x=transformed['transformed_fovea_x'],
            transformed_fovea_y=transformed['transformed_fovea_y'],
            transformed_disc_center_x=transformed['transformed_disc_center_x'],
            transformed_disc_center_y=transformed['transformed_disc_center_y'],
            transform_matrix=matrix_flat,
            en_face_split_x_ref=en_face_split_x_ref,
            en_face_split_x_new=en_face_split_x_new,
            confidence=confidence,
            num_matches=num_matches,
            num_inliers=num_inliers,
            status=status,
            message=message
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Registration API] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Image registration failed: {str(e)}"
        )


@router.get("/registrar/status")
async def get_registrar_status():
    """
    Check if the image registrar service is initialized.
    
    Returns:
        Status information about the registrar service
    """
    service = get_registrar()
    return {
        "status": "ready",
        "n_features": service.n_features,
        "ratio_test_threshold": service.ratio_test_threshold,
        "ransac_threshold": service.ransac_threshold,
        "min_inliers_success": service.min_inliers_success,
        "min_inliers_low_confidence": service.min_inliers_low_confidence
    }
