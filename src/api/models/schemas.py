"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional


class DiscDetectionResponse(BaseModel):
    """Response model for optic disc detection."""
    
    disc_center_x: float = Field(..., description="X coordinate of disc center in original image")
    disc_center_y: float = Field(..., description="Y coordinate of disc center in original image")
    disc_top_y: float = Field(..., description="Y coordinate of disc top in original image")
    disc_bottom_y: float = Field(..., description="Y coordinate of disc bottom in original image")
    disc_height_pixels: float = Field(..., description="Height of disc in pixels (1800 microns)")
    pixel_to_micron_ratio: float = Field(..., description="Conversion factor: 1800 / disc_height_pixels")
    en_face_split_x: int = Field(..., description="X coordinate where en-face region starts in original image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "disc_center_x": 1250.5,
                "disc_center_y": 512.0,
                "disc_top_y": 412.5,
                "disc_bottom_y": 611.5,
                "disc_height_pixels": 199.0,
                "pixel_to_micron_ratio": 9.045,
                "en_face_split_x": 850
            }
        }


class FoveaDetectionRequest(BaseModel):
    """Request model for fovea detection."""
    
    disc_center_x: float
    disc_center_y: float
    disc_height_pixels: float
    en_face_split_x: int
    use_manual_adjustment: bool = Field(default=False, description="Enable interactive manual adjustment UI")


class FoveaDetectionResponse(BaseModel):
    """Response model for fovea detection."""
    
    fovea_x: float = Field(..., description="X coordinate of fovea in original image")
    fovea_y: float = Field(..., description="Y coordinate of fovea in original image")
    detection_method: str = Field(..., description="Method used: 'green_line', 'geometric_fallback', or 'manual'")
    eye_side: str = Field(..., description="OD (right eye) or OS (left eye)")


class GASegmentationResponse(BaseModel):
    """Response model for GA segmentation."""
    
    regions: List[List[Tuple[int, int]]] = Field(..., description="List of GA regions, each as list of (x,y) contour points")
    region_count: int = Field(..., description="Number of GA regions detected")


class DistanceCalculationRequest(BaseModel):
    """Request model for distance calculation."""
    
    fovea_x: float
    fovea_y: float
    selected_ga_region_index: int
    ga_regions: List[List[Tuple[int, int]]]
    pixel_to_micron_ratio: float


class DistanceCalculationResponse(BaseModel):
    """Response model for distance calculation."""
    
    distance_pixels: float
    distance_microns: float
    nearest_ga_point_x: int
    nearest_ga_point_y: int


class ProgressionCalculationRequest(BaseModel):
    """Request model for progression analysis."""
    
    date_before: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    date_after: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    distance_before_microns: float
    distance_after_microns: float
    eye_side_before: str
    eye_side_after: str


class ProgressionCalculationResponse(BaseModel):
    """Response model for progression analysis."""
    
    days_elapsed: int
    distance_change_microns: float
    rate_microns_per_day: Optional[float] = None
    rate_microns_per_month: Optional[float] = None
    predicted_foveal_involvement_date: Optional[str] = None
    status: str = Field(..., description="'progression', 'no_progression', or 'error'")
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
    confidence: float = Field(..., description="Registration confidence score (0.0-1.0)")
    num_matches: int = Field(..., description="Number of good feature matches found")
    num_inliers: int = Field(..., description="Number of inliers after RANSAC")
    status: str = Field(..., description="'success', 'low_confidence', or 'failed'")
    message: Optional[str] = Field(None, description="Human-readable status message")
