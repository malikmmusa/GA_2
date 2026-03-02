/**
 * Main Application Component
 * Orchestrates the complete workflow for GA progression analysis
 */
import { useCallback, useEffect, useState } from 'react';
import type { SetStateAction } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import * as api from './services/api';
import type { ImageAnalysis, AppState, ImageRegistrationResponse } from './types/api';
import { extractErrorMessage } from './utils/errorHandling';
import { OPTIC_DISC_DIAMETER_MICRONS } from './constants/measurements';
import './index.css';

function App() {
  type Target = 'before' | 'after';
  type ImageStateKey = 'imageBefore' | 'imageAfter';
  type ProcessingStateKey = 'isProcessingBefore' | 'isProcessingAfter';

  const [state, setState] = useState<AppState>({
    imageBefore: null,
    imageAfter: null,
    progression: null,
    isProcessingBefore: false,
    isProcessingAfter: false,
    error: null,
  });

  // Track fovea confirmation state for each image
  const [foveaConfirmedBefore, setFoveaConfirmedBefore] = useState(false);
  const [foveaConfirmedAfter, setFoveaConfirmedAfter] = useState(false);

  // Track GA confirmation state for each image
  const [gaConfirmedBefore, setGAConfirmedBefore] = useState(false);
  const [gaConfirmedAfter, setGAConfirmedAfter] = useState(false);
  const [manualGAModeBefore, setManualGAModeBefore] = useState(false);
  const [manualGAModeAfter, setManualGAModeAfter] = useState(false);

  // Track local GA segmentation processing state
  const [isProcessingLocalGABefore, setIsProcessingLocalGABefore] = useState(false);
  const [isProcessingLocalGAAfter, setIsProcessingLocalGAAfter] = useState(false);

  // Track GA region selection messages
  const [gaMessageBefore, setGAMessageBefore] = useState<string | null>(null);
  const [gaMessageAfter, setGAMessageAfter] = useState<string | null>(null);

  // Track image registration state
  const [registrationResult, setRegistrationResult] = useState<ImageRegistrationResponse | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);

  const getImageKey = (target: Target): ImageStateKey =>
    target === 'before' ? 'imageBefore' : 'imageAfter';
  const getProcessingKey = (target: Target): ProcessingStateKey =>
    target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter';
  const getImageAnalysis = (target: Target, appState: AppState = state): ImageAnalysis | null =>
    appState[getImageKey(target)];
  const isFoveaConfirmedForTarget = (target: Target): boolean =>
    target === 'before' ? foveaConfirmedBefore : foveaConfirmedAfter;
  const setLocalGAProcessingForTarget = (target: Target, value: boolean): void => {
    if (target === 'before') {
      setIsProcessingLocalGABefore(value);
      return;
    }
    setIsProcessingLocalGAAfter(value);
  };
  const setGAMessageForTarget = (target: Target, value: string | null): void => {
    if (target === 'before') {
      setGAMessageBefore(value);
      return;
    }
    setGAMessageAfter(value);
  };
  const setManualGAModeForTarget = (
    target: Target,
    value: SetStateAction<boolean>
  ): void => {
    if (target === 'before') {
      setManualGAModeBefore(value);
      return;
    }
    setManualGAModeAfter(value);
  };
  const resetTargetConfirmationState = (target: Target): void => {
    if (target === 'before') {
      setFoveaConfirmedBefore(false);
      setGAConfirmedBefore(false);
      setManualGAModeBefore(false);
      setGAMessageBefore(null);
      return;
    }
    setFoveaConfirmedAfter(false);
    setGAConfirmedAfter(false);
    setManualGAModeAfter(false);
    setGAMessageAfter(null);
  };
  const setTargetProcessingState = (
    target: Target,
    isProcessing: boolean,
    error: string | null = null
  ): void => {
    const processingKey = getProcessingKey(target);
    setState((prev) => ({
      ...prev,
      [processingKey]: isProcessing,
      error,
    }));
  };
  const setTargetImageState = (
    target: Target,
    image: ImageAnalysis,
    options: { clearProgression?: boolean; isProcessing?: boolean } = {}
  ): void => {
    const { clearProgression = true, isProcessing } = options;
    const imageKey = getImageKey(target);
    const processingUpdate =
      isProcessing === undefined ? {} : { [getProcessingKey(target)]: isProcessing };

    setState((prev) => ({
      ...prev,
      [imageKey]: image,
      ...processingUpdate,
      ...(clearProgression ? { progression: null } : {}),
    }));
  };
  const mergeTargetImageState = (
    target: Target,
    merge: (current: ImageAnalysis) => ImageAnalysis,
    options: { clearProgression?: boolean; isProcessing?: boolean } = {}
  ): void => {
    const { clearProgression = true, isProcessing } = options;
    const imageKey = getImageKey(target);
    const processingKey = getProcessingKey(target);

    setState((prev) => {
      const current = prev[imageKey];
      if (!current) return prev;
      return {
        ...prev,
        [imageKey]: merge(current),
        ...(isProcessing === undefined ? {} : { [processingKey]: isProcessing }),
        ...(clearProgression ? { progression: null } : {}),
      };
    });
  };

  /**
   * Cleanup URL.createObjectURL when component unmounts or images change
   * Prevents memory leaks
   */
  useEffect(() => {
    return () => {
      if (state.imageBefore?.imageUrl) {
        URL.revokeObjectURL(state.imageBefore.imageUrl);
      }
      if (state.imageAfter?.imageUrl) {
        URL.revokeObjectURL(state.imageAfter.imageUrl);
      }
    };
  }, [state.imageBefore?.imageUrl, state.imageAfter?.imageUrl]);

  /**
   * Auto-trigger registration when both images are loaded
   * Runs once to get the transform matrix for live fovea transfer
   */
  useEffect(() => {
    const imageBefore = state.imageBefore;
    const imageAfter = state.imageAfter;
    const canRegister =
      imageBefore?.disc &&
      imageBefore?.fovea &&
      imageBefore?.imageFile &&
      imageAfter?.disc &&
      imageAfter?.fovea &&
      imageAfter?.imageFile &&
      !registrationResult &&
      !isRegistering;

    if (canRegister && imageBefore && imageAfter) {
      void attemptRegistration(imageBefore, imageAfter);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    state.imageBefore,
    state.imageAfter,
    registrationResult,
    isRegistering,
  ]);

  /**
   * Auto-calculate progression when both images are ready
   * This prevents infinite loops by using useEffect instead of calling during setState
   */
  useEffect(() => {
    const imageBefore = state.imageBefore;
    const imageAfter = state.imageAfter;
    const canCalculateProgression =
      imageBefore?.distance &&
      imageAfter?.distance &&
      imageBefore?.fovea &&
      imageAfter?.fovea &&
      gaConfirmedBefore &&
      gaConfirmedAfter &&
      !state.progression &&
      !state.isProcessingBefore &&
      !state.isProcessingAfter;

    if (canCalculateProgression && imageBefore && imageAfter) {
      void calculateProgression(imageBefore, imageAfter);
    }
  }, [
    state.imageBefore,
    state.imageAfter,
    gaConfirmedBefore,
    gaConfirmedAfter,
    state.progression,
    state.isProcessingBefore,
    state.isProcessingAfter,
  ]);

  /**
   * Process an uploaded image through the complete analysis pipeline
   */
  const processImage = async (
    file: File,
    date: string,
    target: Target
  ) => {
    // Reset per-image confirmation state when uploading new image
    resetTargetConfirmationState(target);
    setRegistrationResult(null);

    const previousImageUrl = getImageAnalysis(target)?.imageUrl ?? null;
    let imageUrl: string | null = null;
    try {
      setTargetProcessingState(target, true);

      // Create image URL for display
      imageUrl = URL.createObjectURL(file);
      if (previousImageUrl && previousImageUrl !== imageUrl) {
        URL.revokeObjectURL(previousImageUrl);
      }

      // Initialize image analysis object
      const imageAnalysis: ImageAnalysis = {
        imageFile: file,
        imageUrl,
        date,
      };

      // Step 1: Detect disc
      const discResult = await api.detectDisc(file);
      imageAnalysis.disc = discResult;

      // Step 2: Detect fovea
      const foveaResult = await api.detectFovea(file, {
        disc_center_x: discResult.disc_center_x,
        disc_center_y: discResult.disc_center_y,
        disc_height_pixels: discResult.disc_height_pixels,
        en_face_split_x: discResult.en_face_split_x,
        use_manual_adjustment: false,
      });
      imageAnalysis.fovea = foveaResult;
      // STOP HERE - Do not auto-proceed to GA segmentation
      // User must confirm fovea location before continuing

      // Update state (progression will be calculated by useEffect)
      setTargetImageState(target, imageAnalysis, { isProcessing: false, clearProgression: true });

      // NOTE: Registration is triggered automatically via useEffect when both images are loaded.
      // Live fovea transfer happens during drag via handleFoveaAdjust + transform matrix.
    } catch (error: unknown) {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
      setTargetProcessingState(
        target,
        false,
        extractErrorMessage(error, `Failed to process ${target} image`)
      );
    }
  };

  /**
   * Apply the registration transform matrix to a point in Image 1 space,
   * returning the corresponding point in Image 2 space.
   * Pure math, no API call -- instant.
   */
  const applyTransformToFovea = useCallback((
    foveaX: number,
    foveaY: number,
    reg: ImageRegistrationResponse
  ): { x: number; y: number } | null => {
    if (!reg.transform_matrix || reg.transform_matrix.length !== 6 ||
        reg.en_face_split_x_ref == null || reg.en_face_split_x_new == null) {
      return null;
    }

    const [a, b, tx, c, d, ty] = reg.transform_matrix;
    const splitRef = reg.en_face_split_x_ref;
    const splitNew = reg.en_face_split_x_new;

    // Convert to en-face local coords
    const localX = foveaX - splitRef;
    const localY = foveaY;

    // Apply 2x3 affine transform
    const newLocalX = a * localX + b * localY + tx;
    const newLocalY = c * localX + d * localY + ty;

    // Convert back to original image coords
    return {
      x: newLocalX + splitNew,
      y: newLocalY,
    };
  }, []);

  /**
   * Attempt to register Image 2 to Image 1 and store the transform matrix.
   * Called once when both images are loaded.
   */
  const attemptRegistration = useCallback(async (
    imageBefore: ImageAnalysis,
    imageAfter: ImageAnalysis
  ) => {
    if (!imageBefore.imageFile || !imageAfter.imageFile || !imageBefore.fovea || !imageBefore.disc || !imageAfter.disc) {
      return;
    }

    try {
      setIsRegistering(true);

      const result = await api.registerImages(
        imageBefore.imageFile,
        imageAfter.imageFile,
        {
          en_face_split_x_ref: imageBefore.disc.en_face_split_x,
          en_face_split_x_new: imageAfter.disc.en_face_split_x,
          fovea_x: imageBefore.fovea.fovea_x,
          fovea_y: imageBefore.fovea.fovea_y,
          disc_center_x: imageBefore.disc.disc_center_x,
          disc_center_y: imageBefore.disc.disc_center_y,
        }
      );

      setRegistrationResult(result);

      // If registration succeeded (any confidence), immediately apply the transform
      // to update Image 2's fovea based on Image 1's current fovea
      if (result.status !== 'failed' && result.transform_matrix) {
        const transformed = applyTransformToFovea(
          imageBefore.fovea.fovea_x,
          imageBefore.fovea.fovea_y,
          result
        );

        if (transformed) {
          setState((prev) => {
            if (!prev.imageAfter?.fovea) return prev;
            return {
              ...prev,
              imageAfter: {
                ...prev.imageAfter,
                fovea: {
                  ...prev.imageAfter.fovea,
                  fovea_x: transformed.x,
                  fovea_y: transformed.y,
                  detection_method: 'registered',
                },
              },
            };
          });
        }
      }
    } catch {
      setRegistrationResult(null);
    } finally {
      setIsRegistering(false);
    }
  }, [applyTransformToFovea]);

  /**
   * Continue processing after fovea confirmation (Steps 3-4: GA segmentation + distance)
   */
  const continueAfterFoveaConfirmation = async (target: Target) => {
    const imageAnalysis = getImageAnalysis(target);
    
    if (!imageAnalysis || !imageAnalysis.fovea || !imageAnalysis.disc || !imageAnalysis.imageFile) {
      return;
    }

    try {
      setTargetProcessingState(target, true);

      // Step 3: Segment GA regions
      const gaResult = await api.segmentGA(imageAnalysis.imageFile, {
        disc_center_x: imageAnalysis.disc.disc_center_x,
        disc_center_y: imageAnalysis.disc.disc_center_y,
        disc_height_pixels: imageAnalysis.disc.disc_height_pixels,
        en_face_split_x: imageAnalysis.disc.en_face_split_x,
      });

      // Update state using functional updater to preserve any concurrent updates
      // (e.g. registration may have updated the fovea on imageAfter)
      mergeTargetImageState(
        target,
        (latestImage) => ({
          ...latestImage, // Uses latest state, preserves registered fovea
          gaRegions: gaResult,
        }),
        { isProcessing: false, clearProgression: true }
      );
    } catch (error: unknown) {
      setTargetProcessingState(
        target,
        false,
        extractErrorMessage(error, `Failed to segment GA for ${target} image`)
      );
    }
  };

  /**
   * Handle unified fovea confirmation for both images
   */
  const handleConfirmBothFoveas = async () => {
    // Set both confirmation states immediately
    setFoveaConfirmedBefore(true);
    setFoveaConfirmedAfter(true);

    // Process both sides in parallel (GA segmentation)
    // Registration already ran automatically when both images loaded,
    // and fovea adjustments were applied live via transform matrix.
    await Promise.all([
      continueAfterFoveaConfirmation('before'),
      continueAfterFoveaConfirmation('after')
    ]);
  };

  /**
   * Handle unified GA confirmation for both images (single button)
   */
  const handleConfirmBothGA = () => {
    setGAConfirmedBefore(true);
    setGAConfirmedAfter(true);
  };

  /**
   * Handle fovea adjustment click
   */
  const handleFoveaAdjust = (target: Target, x: number, y: number) => {
    const isConfirmed = isFoveaConfirmedForTarget(target);

    // Defense in depth: Block adjustment if fovea is already confirmed
    if (isConfirmed) {
      return;
    }

    // Update fovea coordinates for the adjusted image
    setState((prev) => {
      const imageKey = getImageKey(target);
      const currentImage = prev[imageKey];
      if (!currentImage?.fovea) return prev;
      const updates: Partial<AppState> = {
        [imageKey]: {
          ...currentImage,
          fovea: {
            ...currentImage.fovea,
            fovea_x: x,
            fovea_y: y,
            detection_method: 'manual',
          },
        },
      };

      // LIVE REGISTRATION: When adjusting Image 1 and we have a transform matrix,
      // instantly update Image 2's fovea using client-side affine math
      if (target === 'before' && registrationResult?.transform_matrix && prev.imageAfter?.fovea) {
        const transformed = applyTransformToFovea(x, y, registrationResult);
        if (transformed) {
          updates.imageAfter = {
            ...prev.imageAfter,
            fovea: {
              ...prev.imageAfter.fovea,
              fovea_x: transformed.x,
              fovea_y: transformed.y,
              detection_method: 'registered',
            },
          };
        }
      }

      return { ...prev, ...updates };
    });
  };

  /**
   * Handle disc adjustment (drag handles)
   */
  const handleDiscAdjust = (target: Target, centerX: number, topY: number, bottomY: number) => {
    const isConfirmed = isFoveaConfirmedForTarget(target);

    // Block adjustment if fovea is already confirmed
    if (isConfirmed) {
      return;
    }

    // Recalculate disc parameters
    const disc_height_pixels = bottomY - topY;
    const pixel_to_micron_ratio = OPTIC_DISC_DIAMETER_MICRONS / disc_height_pixels;
    const disc_center_y = (topY + bottomY) / 2;

    setState((prev) => {
      const imageKey = getImageKey(target);
      const currentImage = prev[imageKey];
      if (!currentImage?.disc) return prev;
      const updates: Partial<AppState> = {
        [imageKey]: {
          ...currentImage,
          disc: {
            ...currentImage.disc,
            disc_center_x: centerX,
            disc_center_y: disc_center_y,
            disc_top_y: topY,
            disc_bottom_y: bottomY,
            disc_height_pixels: disc_height_pixels,
            pixel_to_micron_ratio: pixel_to_micron_ratio,
          },
        },
      };

      // LIVE REGISTRATION: When adjusting Image 1's disc and we have a transform matrix,
      // instantly update Image 2's disc position using client-side affine math
      if (target === 'before' && registrationResult?.transform_matrix && prev.imageAfter?.disc) {
        const transformedTop = applyTransformToFovea(centerX, topY, registrationResult);
        const transformedBottom = applyTransformToFovea(centerX, bottomY, registrationResult);

        if (transformedTop && transformedBottom) {
          const newHeight = transformedBottom.y - transformedTop.y;
          const newRatio =
            newHeight > 0
              ? OPTIC_DISC_DIAMETER_MICRONS / newHeight
              : prev.imageAfter.disc.pixel_to_micron_ratio;
          const newCenterY = (transformedTop.y + transformedBottom.y) / 2;

          updates.imageAfter = {
            ...prev.imageAfter,
            disc: {
              ...prev.imageAfter.disc,
              disc_center_x: transformedTop.x,
              disc_center_y: newCenterY,
              disc_top_y: transformedTop.y,
              disc_bottom_y: transformedBottom.y,
              disc_height_pixels: newHeight,
              pixel_to_micron_ratio: newRatio,
            },
          };
        }
      }

      return { ...prev, ...updates };
    });
  };

  /**
   * Calculate progression between before and after images
   */
  const calculateProgression = async (
    before: ImageAnalysis,
    after: ImageAnalysis
  ) => {
    try {
      if (!before.distance || !after.distance || !before.fovea || !after.fovea) {
        return;
      }

      const progressionResult = await api.calculateProgression({
        date_before: before.date,
        date_after: after.date,
        distance_before_microns: before.distance.distance_microns,
        distance_after_microns: after.distance.distance_microns,
        eye_side_before: before.fovea.eye_side,
        eye_side_after: after.fovea.eye_side,
      });

      setState((prev) => ({
        ...prev,
        progression: progressionResult,
      }));
    } catch (error: unknown) {
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Progression calculation failed'),
      }));
    }
  };

  /**
   * Handle date change
   */
  const handleDateChange = (target: Target, newDate: string) => {
    const imageAnalysis = getImageAnalysis(target);
    
    if (!imageAnalysis) return;

    // Update the date in the ImageAnalysis object
    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      date: newDate,
    };

    setTargetImageState(target, updatedImage);

  };

  /**
   * Helper: Find nearest region to a point
   */
  const findNearestRegion = (
    x: number,
    y: number,
    regions: Array<Array<[number, number]>>
  ): { index: number; distance: number } | null => {
    if (!regions || regions.length === 0) return null;

    let minDistance = Infinity;
    let nearestIndex = -1;

    regions.forEach((region, index) => {
      if (!region || region.length === 0) return;

      // Calculate distance to all points in this region
      region.forEach((point) => {
        const dx = x - point[0];
        const dy = y - point[1];
        const distance = Math.hypot(dx, dy);

        if (distance < minDistance) {
          minDistance = distance;
          nearestIndex = index;
        }
      });
    });

    return nearestIndex >= 0 ? { index: nearestIndex, distance: minDistance } : null;
  };

  /**
   * Handle GA region selection
   */
  const handleGARegionSelect = async (
    target: Target,
    regionIndex: number
  ) => {
    const imageAnalysis = getImageAnalysis(target);
    if (!imageAnalysis?.gaRegions || !imageAnalysis.disc || !imageAnalysis.fovea) {
      return;
    }

    try {
      // Calculate distance to new region
      const distanceResult = await api.calculateDistance({
        fovea_x: imageAnalysis.fovea.fovea_x,
        fovea_y: imageAnalysis.fovea.fovea_y,
        selected_ga_region_index: regionIndex,
        ga_regions: imageAnalysis.gaRegions.regions,
        pixel_to_micron_ratio: imageAnalysis.disc.pixel_to_micron_ratio,
      });

      // Update state (progression will be recalculated by useEffect)
      const updatedImage: ImageAnalysis = {
        ...imageAnalysis,
        selectedGARegionIndex: regionIndex,
        distance: distanceResult,
        isManualGAPoint: false,
      };

      setTargetImageState(target, updatedImage);

      // Clear any error messages
      setGAMessageForTarget(target, null);
    } catch (error: unknown) {
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Failed to calculate distance'),
      }));
    }
  };

  /**
   * Handle GA area click (outside existing regions - triggers localized segmentation)
   */
  const handleGAAreaClick = async (target: Target, x: number, y: number) => {
    const imageAnalysis = getImageAnalysis(target);
    if (!imageAnalysis?.gaRegions || !imageAnalysis.disc || !imageAnalysis.fovea || !imageAnalysis.imageFile) {
      return;
    }

    // First check if there's a nearby existing region
    const nearest = findNearestRegion(x, y, imageAnalysis.gaRegions.regions);
    if (nearest && nearest.distance < 50) {
      // Click is close to an existing region, select it
      await handleGARegionSelect(target, nearest.index);
      return;
    }

    // No nearby region, try localized segmentation
    try {
      setLocalGAProcessingForTarget(target, true);
      setGAMessageForTarget(target, null);

      const localResult = await api.segmentGALocal(
        imageAnalysis.imageFile,
        x,
        y,
        {
          disc_center_x: imageAnalysis.disc.disc_center_x,
          disc_center_y: imageAnalysis.disc.disc_center_y,
          disc_height_pixels: imageAnalysis.disc.disc_height_pixels,
          en_face_split_x: imageAnalysis.disc.en_face_split_x,
        }
      );

      if (localResult.region_count > 0) {
        // Found a region! Add it to the existing regions
        const newRegions = [...imageAnalysis.gaRegions.regions, ...localResult.regions];
        const newRegionIndex = newRegions.length - 1;

        // Calculate distance to the new region
        const distanceResult = await api.calculateDistance({
          fovea_x: imageAnalysis.fovea.fovea_x,
          fovea_y: imageAnalysis.fovea.fovea_y,
          selected_ga_region_index: newRegionIndex,
          ga_regions: newRegions,
          pixel_to_micron_ratio: imageAnalysis.disc.pixel_to_micron_ratio,
        });

        const updatedImage: ImageAnalysis = {
          ...imageAnalysis,
          gaRegions: {
            regions: newRegions,
            region_count: newRegions.length,
          },
          selectedGARegionIndex: newRegionIndex,
          distance: distanceResult,
          isManualGAPoint: false,
        };

        setTargetImageState(target, updatedImage);
      } else {
        // No region found
        const message = 'No GA detected in this area. Try clicking another spot.';
        setGAMessageForTarget(target, message);
        setTimeout(() => setGAMessageForTarget(target, null), 3000);
      }
    } catch (error: unknown) {
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Local GA segmentation failed'),
      }));
    } finally {
      setLocalGAProcessingForTarget(target, false);
    }
  };

  /**
   * Handle GA retry (clear selection and let user click again)
   */
  const handleGARetry = (target: Target) => {
    const imageAnalysis = getImageAnalysis(target);
    if (!imageAnalysis) return;

    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      selectedGARegionIndex: undefined,
      distance: undefined,
      isManualGAPoint: false,
    };

    setTargetImageState(target, updatedImage);

    setManualGAModeForTarget(target, false);
    setGAMessageForTarget(target, null);

  };

  /**
   * Toggle manual GA point selection mode
   */
  const handleManualGAModeToggle = (target: Target) => {
    const imageAnalysis = getImageAnalysis(target);
    if (!imageAnalysis) return;

    setManualGAModeForTarget(target, (prev) => !prev);

    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      selectedGARegionIndex: undefined,
      distance: undefined,
      isManualGAPoint: false,
    };

    setTargetImageState(target, updatedImage);
  };

  /**
   * Handle manual GA point click (distance from fovea to clicked point)
   */
  const handleManualGAPoint = (target: Target, x: number, y: number) => {
    const imageAnalysis = getImageAnalysis(target);
    if (!imageAnalysis?.fovea || !imageAnalysis?.disc) return;

    const dx = x - imageAnalysis.fovea.fovea_x;
    const dy = y - imageAnalysis.fovea.fovea_y;
    const distancePixels = Math.hypot(dx, dy);
    const distanceMicrons = distancePixels * imageAnalysis.disc.pixel_to_micron_ratio;

    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      selectedGARegionIndex: undefined,
      distance: {
        distance_pixels: distancePixels,
        distance_microns: distanceMicrons,
        nearest_ga_point_x: x,
        nearest_ga_point_y: y,
      },
      isManualGAPoint: true,
    };

    setTargetImageState(target, updatedImage);

    setGAMessageForTarget(target, null);
  };

  /**
   * Handle PDF download
   */
  const handleDownloadPDF = () => {
    alert('PDF generation will be implemented in Phase 3');
  };

  const handleUnsupportedEyeSideOverride = () => {
    setState((prev) => ({
      ...prev,
      error: 'Manual eye-side override is not supported yet.',
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-primary mb-2">Atrophy Advisor</h1>
          <p className="text-gray-600">OCT Image Analysis for Geographic Atrophy Progression</p>
        </header>

        {/* Error Display */}
        {state.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800 font-semibold">Error</p>
            <p className="text-red-700 text-sm mt-1">{state.error}</p>
            <button
              onClick={() => setState((prev) => ({ ...prev, error: null }))}
              className="text-red-600 underline text-sm mt-2"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Main Content: Dual Image Upload */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Before Image */}
          <div>
            <ImageUpload
              title="IMAGE 1 (BEFORE)"
              onImageUpload={(file, date) => processImage(file, date, 'before')}
              onDateChange={(date) => handleDateChange('before', date)}
              currentDate={state.imageBefore?.date}
              eyeSide={state.imageBefore?.fovea?.eye_side}
              onEyeSideChange={handleUnsupportedEyeSideOverride}
              isProcessing={state.isProcessingBefore}
            />
            <div className="mt-4">
              <ImageCanvas
                imageAnalysis={state.imageBefore}
                onFoveaClick={(x, y) => handleFoveaAdjust('before', x, y)}
                onGARegionClick={(regionIndex) =>
                  handleGARegionSelect('before', regionIndex)
                }
                onGAAreaClick={(x, y) => handleGAAreaClick('before', x, y)}
                onManualGAPointClick={(x, y) => handleManualGAPoint('before', x, y)}
                onManualGAModeToggle={() => handleManualGAModeToggle('before')}
                onDiscAdjust={(centerX, topY, bottomY) => handleDiscAdjust('before', centerX, topY, bottomY)}
                foveaConfirmed={foveaConfirmedBefore}
                gaConfirmed={gaConfirmedBefore}
                manualGAMode={manualGAModeBefore}
                isProcessingGA={isProcessingLocalGABefore}
              />
              {foveaConfirmedBefore && state.imageBefore?.fovea && (
                <p className="text-sm text-green-600 mt-2">✓ Fovea confirmed</p>
              )}
              {gaMessageBefore && (
                <p className="text-sm text-orange-600 mt-2">{gaMessageBefore}</p>
              )}
              {foveaConfirmedBefore &&
               (state.imageBefore?.selectedGARegionIndex !== undefined || state.imageBefore?.distance) &&
               !gaConfirmedBefore && (
                <div className="mt-4">
                  <button
                    onClick={() => handleGARetry('before')}
                    className="w-full px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              )}
              {gaConfirmedBefore && (
                <p className="text-sm text-green-600 mt-2">✓ GA region confirmed</p>
              )}
            </div>
          </div>

          {/* After Image */}
          <div>
            <ImageUpload
              title="IMAGE 2 (AFTER)"
              onImageUpload={(file, date) => processImage(file, date, 'after')}
              onDateChange={(date) => handleDateChange('after', date)}
              currentDate={state.imageAfter?.date}
              eyeSide={state.imageAfter?.fovea?.eye_side}
              onEyeSideChange={handleUnsupportedEyeSideOverride}
              isProcessing={state.isProcessingAfter}
            />
            <div className="mt-4">
              <ImageCanvas
                imageAnalysis={state.imageAfter}
                onFoveaClick={(x, y) => handleFoveaAdjust('after', x, y)}
                onGARegionClick={(regionIndex) =>
                  handleGARegionSelect('after', regionIndex)
                }
                onGAAreaClick={(x, y) => handleGAAreaClick('after', x, y)}
                onManualGAPointClick={(x, y) => handleManualGAPoint('after', x, y)}
                onManualGAModeToggle={() => handleManualGAModeToggle('after')}
                onDiscAdjust={(centerX, topY, bottomY) => handleDiscAdjust('after', centerX, topY, bottomY)}
                foveaConfirmed={foveaConfirmedAfter}
                gaConfirmed={gaConfirmedAfter}
                manualGAMode={manualGAModeAfter}
                isProcessingGA={isProcessingLocalGAAfter}
                registrationSuggestion={registrationResult && registrationResult.status === 'low_confidence' ? {
                  fovea_x: registrationResult.transformed_fovea_x,
                  fovea_y: registrationResult.transformed_fovea_y,
                } : undefined}
              />
              {foveaConfirmedAfter && state.imageAfter?.fovea && (
                <p className="text-sm text-green-600 mt-2">✓ Fovea confirmed</p>
              )}
              {/* Registration status badge */}
              {isRegistering && (
                <p className="text-sm text-blue-600 mt-2">🔄 Aligning with Image 1...</p>
              )}
              {registrationResult && registrationResult.status === 'success' && (
                <p className="text-sm text-green-600 mt-2 font-semibold">
                  ✓ Fovea auto-aligned (high confidence: {(registrationResult.confidence * 100).toFixed(0)}%)
                </p>
              )}
              {registrationResult && registrationResult.status === 'low_confidence' && (
                <p className="text-sm text-yellow-600 mt-2 font-semibold">
                  ⚠ Auto-aligned (moderate confidence: {(registrationResult.confidence * 100).toFixed(0)}%). Verify position.
                </p>
              )}
              {gaMessageAfter && (
                <p className="text-sm text-orange-600 mt-2">{gaMessageAfter}</p>
              )}
              {foveaConfirmedAfter &&
               (state.imageAfter?.selectedGARegionIndex !== undefined || state.imageAfter?.distance) &&
               !gaConfirmedAfter && (
                <div className="mt-4">
                  <button
                    onClick={() => handleGARetry('after')}
                    className="w-full px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              )}
              {gaConfirmedAfter && (
                <p className="text-sm text-green-600 mt-2">✓ GA region confirmed</p>
              )}
            </div>
          </div>
        </div>

        {/* Unified Fovea Confirmation Button */}
        {state.imageBefore?.fovea && 
         state.imageAfter?.fovea && 
         (!foveaConfirmedBefore || !foveaConfirmedAfter) && (
          <div className="mb-8">
            <button
              onClick={handleConfirmBothFoveas}
              disabled={state.isProcessingBefore || state.isProcessingAfter}
              className="btn-primary w-full max-w-2xl mx-auto block"
            >
              Confirm Fovea on Both Images & Continue
            </button>
          </div>
        )}

        {/* Unified GA Confirmation Button */}
        {foveaConfirmedBefore &&
         foveaConfirmedAfter &&
         state.imageBefore?.distance &&
         state.imageAfter?.distance &&
         (!gaConfirmedBefore || !gaConfirmedAfter) && (
          <div className="mb-8">
            <button
              onClick={handleConfirmBothGA}
              disabled={
                state.isProcessingBefore ||
                state.isProcessingAfter ||
                isProcessingLocalGABefore ||
                isProcessingLocalGAAfter
              }
              className="btn-primary w-full max-w-2xl mx-auto block"
            >
              Confirm GA Regions on Both Images
            </button>
          </div>
        )}

        {/* Results Panel */}
        <ResultsPanel
          imageBefore={state.imageBefore}
          imageAfter={state.imageAfter}
          progression={state.progression}
          onDownloadPDF={handleDownloadPDF}
        />

        {/* Footer */}
        <footer className="text-center text-gray-500 text-sm mt-12">
          <p>Atrophy Advisor v1.0 | For Research Use Only</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
