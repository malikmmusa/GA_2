/**
 * Main Application Component
 * Orchestrates the complete workflow for GA progression analysis
 */
import { useState, useEffect } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import * as api from './services/api';
import type { ImageAnalysis, AppState } from './types/api';
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

      // Store GA regions WITHOUT auto-selecting
      const updatedAnalysis: ImageAnalysis = {
        ...imageAnalysis,
        gaRegions: gaResult,
      };

      // Update state - user must click to select a region
      setState((prev) => ({
        ...prev,
        [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedAnalysis,
        [target === 'before' ? 'isProcessingBefore' : 'isProcessingAfter']: false,
        progression: null,
      }));

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

    // Process both sides in parallel
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

    // Update fovea coordinates
    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      fovea: {
        ...imageAnalysis.fovea,
        fovea_x: x,
        fovea_y: y,
        detection_method: 'manual',
      },
    };

    setState((prev) => ({
      ...prev,
      [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
    }));

    console.log(`[${target}] Fovea adjusted to (${x.toFixed(1)}, ${y.toFixed(1)})`);
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

    // Update disc coordinates
    const updatedImage: ImageAnalysis = {
      ...imageAnalysis,
      disc: {
        ...imageAnalysis.disc,
        disc_center_x: centerX,
        disc_center_y: disc_center_y,
        disc_top_y: topY,
        disc_bottom_y: bottomY,
        disc_height_pixels: disc_height_pixels,
        pixel_to_micron_ratio: pixel_to_micron_ratio,
      },
    };

    setState((prev) => ({
      ...prev,
      [target === 'before' ? 'imageBefore' : 'imageAfter']: updatedImage,
    }));

    console.log(`[${target}] Disc adjusted: height=${disc_height_pixels.toFixed(1)}px, ratio=${pixel_to_micron_ratio.toFixed(3)}µm/px`);
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
              />
              {foveaConfirmedAfter && state.imageAfter?.fovea && (
                <p className="text-sm text-green-600 mt-2">✓ Fovea confirmed</p>
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
