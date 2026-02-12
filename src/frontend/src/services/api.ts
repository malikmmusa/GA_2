/**
 * API service for communicating with the backend
 */
import axios from 'axios';
import type {
  DiscDetectionResponse,
  FoveaDetectionRequest,
  FoveaDetectionResponse,
  GASegmentationResponse,
  DistanceCalculationRequest,
  DistanceCalculationResponse,
  ProgressionCalculationRequest,
  ProgressionCalculationResponse,
} from '../types/api';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Detect optic disc in an OCT image
 */
export async function detectDisc(imageFile: File): Promise<DiscDetectionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await api.post<DiscDetectionResponse>('/detect-disc', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Detect fovea in an OCT image
 */
export async function detectFovea(
  imageFile: File,
  request: FoveaDetectionRequest
): Promise<FoveaDetectionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('request_data', JSON.stringify(request));

  const response = await api.post<FoveaDetectionResponse>('/detect-fovea', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Segment GA regions in an OCT image
 */
export async function segmentGA(
  imageFile: File,
  options?: {
    disc_center_x?: number;
    disc_center_y?: number;
    disc_height_pixels?: number;
    en_face_split_x?: number;
    fovea_x?: number;
    fovea_y?: number;
  }
): Promise<GASegmentationResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const params = new URLSearchParams();
  if (options?.disc_center_x !== undefined) params.append('disc_center_x', options.disc_center_x.toString());
  if (options?.disc_center_y !== undefined) params.append('disc_center_y', options.disc_center_y.toString());
  if (options?.disc_height_pixels !== undefined) params.append('disc_height_pixels', options.disc_height_pixels.toString());
  if (options?.en_face_split_x !== undefined) params.append('en_face_split_x', options.en_face_split_x.toString());
  if (options?.fovea_x !== undefined) params.append('fovea_x', options.fovea_x.toString());
  if (options?.fovea_y !== undefined) params.append('fovea_y', options.fovea_y.toString());

  const url = `/segment-ga${params.toString() ? '?' + params.toString() : ''}`;

  const response = await api.post<GASegmentationResponse>(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Segment GA region locally around a clicked point (fallback for missed regions)
 */
export async function segmentGALocal(
  imageFile: File,
  clickX: number,
  clickY: number,
  options?: {
    disc_center_x?: number;
    disc_center_y?: number;
    disc_height_pixels?: number;
    en_face_split_x?: number;
  }
): Promise<GASegmentationResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const params = new URLSearchParams();
  params.append('click_x', clickX.toString());
  params.append('click_y', clickY.toString());
  if (options?.disc_center_x !== undefined) params.append('disc_center_x', options.disc_center_x.toString());
  if (options?.disc_center_y !== undefined) params.append('disc_center_y', options.disc_center_y.toString());
  if (options?.disc_height_pixels !== undefined) params.append('disc_height_pixels', options.disc_height_pixels.toString());
  if (options?.en_face_split_x !== undefined) params.append('en_face_split_x', options.en_face_split_x.toString());

  const url = `/segment-ga-local?${params.toString()}`;

  const response = await api.post<GASegmentationResponse>(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Calculate distance from fovea to GA region
 */
export async function calculateDistance(
  request: DistanceCalculationRequest
): Promise<DistanceCalculationResponse> {
  const response = await api.post<DistanceCalculationResponse>('/calculate-distance', request);
  return response.data;
}

/**
 * Calculate progression between two images
 */
export async function calculateProgression(
  request: ProgressionCalculationRequest
): Promise<ProgressionCalculationResponse> {
  const response = await api.post<ProgressionCalculationResponse>('/calculate-progression', request);
  return response.data;
}
