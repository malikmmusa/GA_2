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
  ImageRegistrationResponse,
} from '../types/api';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

type MultipartFieldValue = string | Blob;
type QueryValue = string | number | boolean | null | undefined;

function createFormData(fields: Array<[string, MultipartFieldValue]>): FormData {
  const formData = new FormData();
  fields.forEach(([name, value]) => {
    formData.append(name, value);
  });
  return formData;
}

function buildQueryString(params: Record<string, QueryValue>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    searchParams.append(key, String(value));
  });

  const serialized = searchParams.toString();
  return serialized ? `?${serialized}` : '';
}

/**
 * Detect optic disc in an OCT image
 */
export async function detectDisc(imageFile: File): Promise<DiscDetectionResponse> {
  const formData = createFormData([['file', imageFile]]);
  const response = await api.post<DiscDetectionResponse>('/detect-disc', formData);

  return response.data;
}

/**
 * Detect fovea in an OCT image
 */
export async function detectFovea(
  imageFile: File,
  request: FoveaDetectionRequest
): Promise<FoveaDetectionResponse> {
  const formData = createFormData([
    ['file', imageFile],
    ['request_data', JSON.stringify(request)],
  ]);
  const response = await api.post<FoveaDetectionResponse>('/detect-fovea', formData);

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
  const formData = createFormData([['file', imageFile]]);
  const queryString = buildQueryString({
    disc_center_x: options?.disc_center_x,
    disc_center_y: options?.disc_center_y,
    disc_height_pixels: options?.disc_height_pixels,
    en_face_split_x: options?.en_face_split_x,
    fovea_x: options?.fovea_x,
    fovea_y: options?.fovea_y,
  });
  const response = await api.post<GASegmentationResponse>(`/segment-ga${queryString}`, formData);

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
  const formData = createFormData([['file', imageFile]]);
  const queryString = buildQueryString({
    click_x: clickX,
    click_y: clickY,
    disc_center_x: options?.disc_center_x,
    disc_center_y: options?.disc_center_y,
    disc_height_pixels: options?.disc_height_pixels,
    en_face_split_x: options?.en_face_split_x,
  });
  const response = await api.post<GASegmentationResponse>(
    `/segment-ga-local${queryString}`,
    formData
  );

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

/**
 * Register two images and transfer landmarks from reference to new image
 */
export async function registerImages(
  referenceFile: File,
  newFile: File,
  request: {
    en_face_split_x_ref: number;
    en_face_split_x_new: number;
    fovea_x: number;
    fovea_y: number;
    disc_center_x?: number;
    disc_center_y?: number;
  }
): Promise<ImageRegistrationResponse> {
  const formData = createFormData([
    ['file_reference', referenceFile],
    ['file_new', newFile],
    ['request_data', JSON.stringify(request)],
  ]);
  const response = await api.post<ImageRegistrationResponse>('/register-images', formData);

  return response.data;
}
