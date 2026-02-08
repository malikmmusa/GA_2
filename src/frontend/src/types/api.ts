/**
 * TypeScript types for API requests and responses
 */

export interface DiscDetectionResponse {
  disc_center_x: number;
  disc_center_y: number;
  disc_top_y: number;
  disc_bottom_y: number;
  disc_height_pixels: number;
  pixel_to_micron_ratio: number;
  en_face_split_x: number;
}

export interface FoveaDetectionRequest {
  disc_center_x: number;
  disc_center_y: number;
  disc_height_pixels: number;
  en_face_split_x: number;
  use_manual_adjustment: boolean;
}

export interface FoveaDetectionResponse {
  fovea_x: number;
  fovea_y: number;
  detection_method: string;
  eye_side: 'OD' | 'OS';
}

export interface GASegmentationResponse {
  regions: Array<Array<[number, number]>>;
  region_count: number;
}

export interface DistanceCalculationRequest {
  fovea_x: number;
  fovea_y: number;
  selected_ga_region_index: number;
  ga_regions: Array<Array<[number, number]>>;
  pixel_to_micron_ratio: number;
}

export interface DistanceCalculationResponse {
  distance_pixels: number;
  distance_microns: number;
  nearest_ga_point_x: number;
  nearest_ga_point_y: number;
}

export interface ProgressionCalculationRequest {
  date_before: string;
  date_after: string;
  distance_before_microns: number;
  distance_after_microns: number;
  eye_side_before: 'OD' | 'OS';
  eye_side_after: 'OD' | 'OS';
}

export interface ProgressionCalculationResponse {
  status: 'progression' | 'no_progression' | 'error';
  error_message: string | null;
  days_elapsed: number;
  distance_change_microns: number;
  rate_microns_per_day: number | null;
  rate_microns_per_month: number | null;
  predicted_foveal_involvement_date: string | null;
}

/**
 * Complete image analysis result
 */
export interface ImageAnalysis {
  imageFile: File;
  imageUrl: string;
  date: string;
  disc?: DiscDetectionResponse;
  fovea?: FoveaDetectionResponse;
  gaRegions?: GASegmentationResponse;
  selectedGARegionIndex?: number;
  distance?: DistanceCalculationResponse;
}

/**
 * Application state
 */
export interface AppState {
  imageBefore: ImageAnalysis | null;
  imageAfter: ImageAnalysis | null;
  progression: ProgressionCalculationResponse | null;
  isProcessingBefore: boolean;
  isProcessingAfter: boolean;
  error: string | null;
}
