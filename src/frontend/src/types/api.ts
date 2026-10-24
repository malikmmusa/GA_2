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
  image_format: 'heidelberg' | 'standalone';
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
  detection_method:
    | 'green_line'
    | 'geometric_fallback'
    | 'anatomy_aware'
    | 'raw_geometry'
    | 'manual'
    | 'registered';
  eye_side: 'OD' | 'OS';
}

export interface GASegmentationResponse {
  regions: Array<Array<[number, number]>>;
  region_count: number;
  /** Confidence (0-1) that an automatic measurement from these regions is usable. */
  confidence?: number;
  /**
   * False when automatic segmentation should not be measured from and the user
   * should place the GA point manually.
   */
  auto_measurement_reliable?: boolean;
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
  rate_microns_per_year: number | null;
  predicted_foveal_involvement_date: string | null;
  years_until_involvement: number | null;
}

export interface ImageRegistrationResponse {
  transformed_fovea_x: number;
  transformed_fovea_y: number;
  transformed_disc_center_x: number | null;
  transformed_disc_center_y: number | null;
  transform_matrix: [number, number, number, number, number, number] | null;
  en_face_split_x_ref: number | null;
  en_face_split_x_new: number | null;
  confidence: number;
  num_matches: number;
  num_inliers: number;
  status: 'success' | 'low_confidence' | 'failed';
  message: string | null;
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
  /** True when distance was set by manual point click (not a segmented region) */
  isManualGAPoint?: boolean;
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
