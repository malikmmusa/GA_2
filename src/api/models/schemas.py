"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional, Tuple

from ..constants import DISC_DIAMETER_MICRONS


class DiscDetectionResponse(BaseModel):
    """Response model for optic disc detection."""
    
    disc_center_x: float = Field(..., description="X coordinate of disc center in original image")
    disc_center_y: float = Field(..., description="Y coordinate of disc center in original image")
    disc_top_y: float = Field(..., description="Y coordinate of disc top in original image")
    disc_bottom_y: float = Field(..., description="Y coordinate of disc bottom in original image")
    disc_height_pixels: float = Field(
        ...,
        gt=0,
        description=f"Height of disc in pixels ({DISC_DIAMETER_MICRONS:.0f} microns)"
    )
    pixel_to_micron_ratio: float = Field(
        ...,
        gt=0,
        description=f"Conversion factor: {DISC_DIAMETER_MICRONS:.0f} / disc_height_pixels"
    )
    en_face_split_x: int = Field(
        ...,
        ge=0,
        description="X coordinate where en-face region starts in original image"
    )
    image_format: Literal["heidelberg", "standalone"] = Field(
        default="heidelberg",
        description="Detected image format: 'heidelberg' (composite B-scan+en-face) or 'standalone' (Cirrus/single panel)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "disc_center_x": 1250.5,
                "disc_center_y": 512.0,
                "disc_top_y": 412.5,
                "disc_bottom_y": 611.5,
                "disc_height_pixels": 199.0,
                "pixel_to_micron_ratio": 9.045,
                "en_face_split_x": 850,
            }
        }
    )


class FoveaDetectionRequest(BaseModel):
    """Request model for fovea detection."""
    
    disc_center_x: float
    disc_center_y: float
    disc_height_pixels: float = Field(..., gt=0)
    en_face_split_x: int = Field(..., ge=0)
    use_manual_adjustment: bool = Field(default=False, description="Enable interactive manual adjustment UI")


class FoveaDetectionResponse(BaseModel):
    """Response model for fovea detection."""
    
    fovea_x: float = Field(..., description="X coordinate of fovea in original image")
    fovea_y: float = Field(..., description="Y coordinate of fovea in original image")
    detection_method: Literal[
        "green_line",
        "geometric_fallback",
        "anatomy_aware",
        "raw_geometry",
        "manual",
    ] = Field(..., description="Method used for fovea localization")
    eye_side: Literal["OD", "OS"] = Field(..., description="OD (right eye) or OS (left eye)")


class ImageRegistrationRequest(BaseModel):
    """Request model for image registration."""

    en_face_split_x_ref: int = Field(..., ge=0)
    en_face_split_x_new: int = Field(..., ge=0)
    fovea_x: float
    fovea_y: float
    disc_center_x: Optional[float] = None
    disc_center_y: Optional[float] = None


class GASegmentationResponse(BaseModel):
    """Response model for GA segmentation."""
    
    regions: List[List[Tuple[int, int]]] = Field(..., description="List of GA regions, each as list of (x,y) contour points")
    region_count: int = Field(..., description="Number of GA regions detected")


class DistanceCalculationRequest(BaseModel):
    """Request model for distance calculation."""
    
    fovea_x: float
    fovea_y: float
    selected_ga_region_index: int = Field(..., ge=0)
    ga_regions: List[List[Tuple[int, int]]]
    pixel_to_micron_ratio: float = Field(..., gt=0)


class DistanceCalculationResponse(BaseModel):
    """Response model for distance calculation."""
    
    distance_pixels: float = Field(..., ge=0)
    distance_microns: float = Field(..., ge=0)
    nearest_ga_point_x: int
    nearest_ga_point_y: int


class ProgressionCalculationRequest(BaseModel):
    """Request model for progression analysis."""
    
    date_before: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date string (YYYY-MM-DD)")
    date_after: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date string (YYYY-MM-DD)")
    distance_before_microns: float = Field(..., ge=0)
    distance_after_microns: float = Field(..., ge=0)
    eye_side_before: Literal["OD", "OS"]
    eye_side_after: Literal["OD", "OS"]


class ProgressionCalculationResponse(BaseModel):
    """Response model for progression analysis."""
    
    days_elapsed: int
    distance_change_microns: float
    rate_microns_per_day: Optional[float] = None
    rate_microns_per_month: Optional[float] = None
    rate_microns_per_year: Optional[float] = None
    predicted_foveal_involvement_date: Optional[str] = None
    years_until_involvement: Optional[float] = None
    status: Literal["progression", "no_progression", "error"] = Field(
        ...,
        description="Progression analysis status"
    )
    error_message: Optional[str] = None


class ImageRegistrationResponse(BaseModel):
    """Response model for image registration and landmark transfer."""
    
    transformed_fovea_x: float = Field(..., description="Fovea X coordinate in Image 2 space after registration")
    transformed_fovea_y: float = Field(..., description="Fovea Y coordinate in Image 2 space after registration")
    transformed_disc_center_x: Optional[float] = Field(None, description="Disc center X in Image 2 space (for validation)")
    transformed_disc_center_y: Optional[float] = Field(None, description="Disc center Y in Image 2 space (for validation)")
    transform_matrix: Optional[List[float]] = Field(None, description="2x3 affine matrix as 6 floats [a,b,tx,c,d,ty] for client-side transform")
    en_face_split_x_ref: Optional[int] = Field(None, description="En-face split X used for reference image")
    en_face_split_x_new: Optional[int] = Field(None, description="En-face split X used for new image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Registration confidence score (0.0-1.0)")
    num_matches: int = Field(..., ge=0, description="Number of good feature matches found")
    num_inliers: int = Field(..., ge=0, description="Number of inliers after RANSAC")
    status: Literal["success", "low_confidence", "failed"] = Field(
        ...,
        description="Registration status"
    )
    message: Optional[str] = Field(None, description="Human-readable status message")


class RootStatusResponse(BaseModel):
    """Response model for API root status."""

    status: Literal["operational"]
    message: str
    version: str
    docs: str


class HealthStatusResponse(BaseModel):
    """Response model for health checks."""

    status: Literal["healthy"]


class DiscDetectorStatusResponse(BaseModel):
    """Response model for disc detector service status."""

    status: Literal["ready"]
    model_path: str
    device: str
    img_size: int


class RegistrarStatusResponse(BaseModel):
    """Response model for image registrar service status."""

    status: Literal["ready"]
    n_features: int
    ratio_test_threshold: float
    ransac_threshold: float
    min_inliers_success: int
    min_inliers_low_confidence: int
