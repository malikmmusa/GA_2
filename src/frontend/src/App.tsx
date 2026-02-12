/**
 * Main Application Component
 * Orchestrates the complete workflow for GA progression analysis
 */
import { useState, useEffect } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import * as api from './services/api';
import type { ImageAnalysis, AppState, ImageRegistrationResponse } from './types/api';
import { extractErrorMessage } from './utils/errorHandling';
import './index.css';

function App() {
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

  // Track local GA segmentation processing state
  const [isProcessingLocalGABefore, setIsProcessingLocalGABefore] = useState(false);
  const [isProcessingLocalGAAfter, setIsProcessingLocalGAAfter] = useState(false);

  // Track GA region selection messages
  const [gaMessageBefore, setGAMessageBefore] = useState<string | null>(null);
  const [gaMessageAfter, setGAMessageAfter] = useState<string | null>(null);

  // Track image registration state
  const [registrationResult, setRegistrationResult] = useState<ImageRegistrationResponse | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);

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
    const canRegister =
      state.imageBefore?.disc &&
      state.imageBefore?.fovea &&
      state.imageBefore?.imageFile &&
      state.imageAfter?.disc &&
      state.imageAfter?.fovea &&
      state.imageAfter?.imageFile &&
      !registrationResult &&
      !isRegistering;

    if (canRegister) {
      attemptRegistration(state.imageBefore!, state.imageAfter!);
    }
  }, [
    state.imageBefore?.disc,
    state.imageBefore?.fovea,
    state.imageBefore?.imageFile,
    state.imageAfter?.disc,
    state.imageAfter?.fovea,
    state.imageAfter?.imageFile,
    registrationResult,
    isRegistering,
  ]);

  /**
   * Auto-calculate progression when both images are ready
   * This prevents infinite loops by using useEffect instead of calling during setState
   */
  useEffect(() => {
    const canCalculateProgression =
      state.imageBefore?.distance &&
      state.imageAfter?.distance &&
      state.imageBefore?.fovea &&
      state.imageAfter?.fovea &&
      gaConfirmedBefore &&
      gaConfirmedAfter &&
      !state.progression &&
      !state.isProcessingBefore &&
      !state.isProcessingAfter;

    if (canCalculateProgression) {
      calculateProgression(state.imageBefore!, state.imageAfter!);
    }
  }, [
    state.imageBefore?.distance,
    state.imageAfter?.distance,
    state.imageBefore?.fovea,
    state.imageAfter?.fovea,
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
    target: 'before' | 'after'
  ) => {
    // Reset fovea confirmation state when uploading new image
    if (target === 'before') {
      setFoveaConfirmedBefore(false);
    } else {
      setFoveaConfirmedAfter(false);
    }

    try {
      setState((prev) => ({ 
        ...prev, 
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: true, 
        error: null 
      }));

      // Create image URL for display
      const imageUrl = URL.createObjectURL(file);

      // Initialize image analysis object
      const imageAnalysis: ImageAnalysis = {
        imageFile: file,
        imageUrl,
        date,
      };

      // Step 1: Detect disc
      console.log(`[${target}] Detecting disc...`);
      const discResult = await api.detectDisc(file);
      imageAnalysis.disc = discResult;

      // Step 2: Detect fovea
      console.log(`[${target}] Detecting fovea...`);
      const foveaResult = await api.detectFovea(file, {
        disc_center_x: discResult.disc_center_x,
        disc_center_y: discResult.disc_center_y,
        disc_height_pixels: discResult.disc_height_pixels,
        en_face_split_x: discResult.en_face_split_x,
        use_manual_adjustment: false,
      });
      imageAnalysis.fovea = foveaResult;

      console.log(`[${target}] Fovea detected. Waiting for user confirmation...`);
      // STOP HERE - Do not auto-proceed to GA segmentation
      // User must confirm fovea location before continuing

      // Update state (progression will be calculated by useEffect)
      setState((prev) => ({
        ...prev,
        [target === 'before' ? 'imageBefore' : 'imageAfter']: imageAnalysis,
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: false,
        progression: null, // Clear old progression to trigger recalculation
      }));

      console.log(`[${target}] Analysis complete!`);

      // NOTE: Registration is triggered automatically via useEffect when both images are loaded.
      // Live fovea transfer happens during drag via handleFoveaAdjust + transform matrix.
    } catch (error: any) {
      console.error(`[${target}] Error:`, error);
      setState((prev) => ({
        ...prev,
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: false,
        error: extractErrorMessage(error, `Failed to process ${target} image`),
      }));
    }
  };

  /**
   * Apply the registration transform matrix to a point in Image 1 space,
   * returning the corresponding point in Image 2 space.
   * Pure math, no API call -- instant.
   */
  const applyTransformToFovea = (
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
  };

  /**
   * Attempt to register Image 2 to Image 1 and store the transform matrix.
   * Called once when both images are loaded.
   */
  const attemptRegistration = async (
    imageBefore: ImageAnalysis,
    imageAfter: ImageAnalysis
  ) => {
    if (!imageBefore.imageFile || !imageAfter.imageFile || !imageBefore.fovea || !imageBefore.disc || !imageAfter.disc) {
      console.log('[registration] Missing required data for registration');
      return;
    }

    try {
      setIsRegistering(true);
      console.log('[registration] Starting registration...');

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
      console.log(`[registration] Result: ${result.status}, confidence: ${result.confidence.toFixed(2)}, hasMatrix: ${!!result.transform_matrix}`);

      // If registration succeeded (any confidence), immediately apply the transform
      // to update Image 2's fovea based on Image 1's current fovea
      if (result.status !== 'failed' && result.transform_matrix) {
        const transformed = applyTransformToFovea(
          imageBefore.fovea.fovea_x,
          imageBefore.fovea.fovea_y,
          result
        );

        if (transformed) {
          console.log(`[registration] Applying registered fovea to Image 2: (${transformed.x.toFixed(1)}, ${transformed.y.toFixed(1)})`);
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
    } catch (error: any) {
      console.error('[registration] Error:', error);
    } finally {
      setIsRegistering(false);
    }
  };

  /**
   * Continue processing after fovea confirmation (Steps 3-4: GA segmentation + distance)
   */
  const continueAfterFoveaConfirmation = async (target: 'before' | 'after') => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    
    if (!imageAnalysis || !imageAnalysis.fovea || !imageAnalysis.disc || !imageAnalysis.imageFile) {
      console.error(`[${target}] Cannot continue: missing fovea or disc data`);
      return;
    }

    try {
      setState((prev) => ({ 
        ...prev, 
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: true, 
        error: null 
      }));

      // Step 3: Segment GA regions
      console.log(`[${target}] Segmenting GA...`);
      console.log(`[${target}] GA Segmentation Input:`, {
        hasImageFile: !!imageAnalysis.imageFile,
        hasDisc: !!imageAnalysis.disc,
        hasFovea: !!imageAnalysis.fovea,
      });
      
      const gaResult = await api.segmentGA(imageAnalysis.imageFile, {
        disc_center_x: imageAnalysis.disc.disc_center_x,
        disc_center_y: imageAnalysis.disc.disc_center_y,
        disc_height_pixels: imageAnalysis.disc.disc_height_pixels,
        en_face_split_x: imageAnalysis.disc.en_face_split_x,
      });
      
      console.log(`[${target}] GA Segmentation Result:`, {
        regionCount: gaResult.region_count,
        hasRegions: !!gaResult.regions,
        regionsLength: gaResult.regions?.length,
      });

      // Update state using functional updater to preserve any concurrent updates
      // (e.g. registration may have updated the fovea on imageAfter)
      setState((prev) => {
        const stateKey = target === 'before' ? 'imageBefore' : 'imageAfter';
        const latestImage = prev[stateKey];
        if (!latestImage) return prev;

        return {
          ...prev,
          [stateKey]: {
            ...latestImage, // Uses latest state, preserves registered fovea
            gaRegions: gaResult,
          },
          [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: false,
          progression: null,
        };
      });

      console.log(`[${target}] GA segmentation complete! User must click to select a region.`);
    } catch (error: any) {
      console.error(`[${target}] Error:`, error);
      setState((prev) => ({
        ...prev,
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: false,
        error: extractErrorMessage(error, `Failed to segment GA for ${target} image`),
      }));
    }
  };

  /**
   * Handle fovea confirmation button click
   */
  const handleFoveaConfirm = async (target: 'before' | 'after') => {
    if (target === 'before') {
      setFoveaConfirmedBefore(true);
    } else {
      setFoveaConfirmedAfter(true);
    }

    // Continue to GA segmentation
    await continueAfterFoveaConfirmation(target);
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
   * Handle fovea adjustment click
   */
  const handleFoveaAdjust = async (target: 'before' | 'after', x: number, y: number) => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    const isConfirmed = target === 'before' ? foveaConfirmedBefore : foveaConfirmedAfter;

    // Defense in depth: Block adjustment if fovea is already confirmed
    if (isConfirmed) {
      console.log(`[${target}] Fovea adjustment blocked - already confirmed`);
      return;
    }

    if (!imageAnalysis?.fovea) return;

    // Update fovea coordinates for the adjusted image
    setState((prev) => {
      const updates: Partial<AppState> = {
        [target === 'before' ? 'imageBefore' : 'imageAfter']: {
          ...imageAnalysis,
          fovea: {
            ...imageAnalysis.fovea!,
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
  const handleDiscAdjust = (target: 'before' | 'after', centerX: number, topY: number, bottomY: number) => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    const isConfirmed = target === 'before' ? foveaConfirmedBefore : foveaConfirmedAfter;

    // Block adjustment if fovea is already confirmed
    if (isConfirmed) {
      console.log(`[${target}] Disc adjustment blocked - fovea already confirmed`);
      return;
    }

    if (!imageAnalysis?.disc) return;

    // Recalculate disc parameters
    const disc_height_pixels = bottomY - topY;
    const pixel_to_micron_ratio = 1800 / disc_height_pixels;
    const disc_center_y = (topY + bottomY) / 2;

    setState((prev) => {
      const updates: Partial<AppState> = {
        [target === 'before' ? 'imageBefore' : 'imageAfter']: {
          ...imageAnalysis,
          disc: {
            ...imageAnalysis.disc!,
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
          const newRatio = newHeight > 0 ? 1800 / newHeight : prev.imageAfter.disc.pixel_to_micron_ratio;
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

      console.log('[progression] Calculating...');
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

      console.log('[progression] Complete!');
    } catch (error: any) {
      console.error('[progression] Error:', error);
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Progression calculation failed'),
      }));
    }
  };

  /**
   * Handle date change
   */
  const handleDateChange = (target: 'before' | 'after', newDate: string) => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    
    if (!imageAnalysis) return;

    // Update the date in the ImageAnalysis object
    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      date: newDate,
    };

    setState((prev) => ({
      ...prev,
      [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
      progression: null, // Clear progression to trigger recalculation
    }));

    console.log(`[${target}] Date updated to ${newDate}`);
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
        const distance = Math.sqrt(dx * dx + dy * dy);

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
    target: 'before' | 'after',
    regionIndex: number
  ) => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
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
      };

      setState((prev) => ({
        ...prev,
        [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
        progression: null, // Clear to trigger recalculation
      }));

      // Clear any error messages
      if (target === 'before') {
        setGAMessageBefore(null);
      } else {
        setGAMessageAfter(null);
      }
    } catch (error: any) {
      console.error('Error selecting GA region:', error);
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Failed to calculate distance'),
      }));
    }
  };

  /**
   * Handle GA area click (outside existing regions - triggers localized segmentation)
   */
  const handleGAAreaClick = async (target: 'before' | 'after', x: number, y: number) => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    if (!imageAnalysis?.gaRegions || !imageAnalysis.disc || !imageAnalysis.fovea || !imageAnalysis.imageFile) {
      return;
    }

    // First check if there's a nearby existing region
    const nearest = findNearestRegion(x, y, imageAnalysis.gaRegions.regions);
    if (nearest && nearest.distance < 50) {
      // Click is close to an existing region, select it
      console.log(`[${target}] Click near existing region ${nearest.index} (distance: ${nearest.distance.toFixed(1)}px)`);
      await handleGARegionSelect(target, nearest.index);
      return;
    }

    // No nearby region, try localized segmentation
    try {
      if (target === 'before') {
        setIsProcessingLocalGABefore(true);
        setGAMessageBefore(null);
      } else {
        setIsProcessingLocalGAAfter(true);
        setGAMessageAfter(null);
      }

      console.log(`[${target}] Attempting local segmentation at (${x.toFixed(1)}, ${y.toFixed(1)})`);

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

      console.log(`[${target}] Local segmentation result: ${localResult.region_count} regions`);

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
        };

        setState((prev) => ({
          ...prev,
          [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
          progression: null,
        }));

        console.log(`[${target}] Local region added and selected`);
      } else {
        // No region found
        const message = 'No GA detected in this area. Try clicking another spot.';
        console.log(`[${target}] ${message}`);
        
        if (target === 'before') {
          setGAMessageBefore(message);
          setTimeout(() => setGAMessageBefore(null), 3000);
        } else {
          setGAMessageAfter(message);
          setTimeout(() => setGAMessageAfter(null), 3000);
        }
      }
    } catch (error: any) {
      console.error(`[${target}] Local segmentation error:`, error);
      setState((prev) => ({
        ...prev,
        error: extractErrorMessage(error, 'Local GA segmentation failed'),
      }));
    } finally {
      if (target === 'before') {
        setIsProcessingLocalGABefore(false);
      } else {
        setIsProcessingLocalGAAfter(false);
      }
    }
  };

  /**
   * Handle GA retry (clear selection and let user click again)
   */
  const handleGARetry = (target: 'before' | 'after') => {
    const imageAnalysis = target === 'before' ? state.imageBefore : state.imageAfter;
    if (!imageAnalysis) return;

    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      selectedGARegionIndex: undefined,
      distance: undefined,
    };

    setState((prev) => ({
      ...prev,
      [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
      progression: null,
    }));

    // Clear any messages
    if (target === 'before') {
      setGAMessageBefore(null);
    } else {
      setGAMessageAfter(null);
    }

    console.log(`[${target}] GA selection cleared, user can click again`);
  };

  /**
   * Handle PDF download
   */
  const handleDownloadPDF = () => {
    alert('PDF generation will be implemented in Phase 3');
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
              onEyeSideChange={(side) => {
                // Manual eye side override (would need API support)
                console.log('Eye side override:', side);
              }}
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
                onDiscAdjust={(centerX, topY, bottomY) => handleDiscAdjust('before', centerX, topY, bottomY)}
                foveaConfirmed={foveaConfirmedBefore}
                isProcessingGA={isProcessingLocalGABefore}
              />
              {foveaConfirmedBefore && state.imageBefore?.fovea && (
                <p className="text-sm text-green-600 mt-2">✓ Fovea confirmed</p>
              )}
              {gaMessageBefore && (
                <p className="text-sm text-orange-600 mt-2">{gaMessageBefore}</p>
              )}
              {/* GA Confirmation Buttons */}
              {foveaConfirmedBefore && 
               state.imageBefore?.selectedGARegionIndex !== undefined &&
               !gaConfirmedBefore && (
                <div className="mt-4 flex gap-2">
                  <button
                    onClick={() => setGAConfirmedBefore(true)}
                    className="btn-primary flex-1"
                  >
                    Confirm GA Region
                  </button>
                  <button
                    onClick={() => handleGARetry('before')}
                    className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded transition-colors"
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
              onEyeSideChange={(side) => {
                // Manual eye side override (would need API support)
                console.log('Eye side override:', side);
              }}
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
                onDiscAdjust={(centerX, topY, bottomY) => handleDiscAdjust('after', centerX, topY, bottomY)}
                foveaConfirmed={foveaConfirmedAfter}
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
              {/* GA Confirmation Buttons */}
              {foveaConfirmedAfter && 
               state.imageAfter?.selectedGARegionIndex !== undefined &&
               !gaConfirmedAfter && (
                <div className="mt-4 flex gap-2">
                  <button
                    onClick={() => setGAConfirmedAfter(true)}
                    className="btn-primary flex-1"
                  >
                    Confirm GA Region
                  </button>
                  <button
                    onClick={() => handleGARetry('after')}
                    className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded transition-colors"
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
