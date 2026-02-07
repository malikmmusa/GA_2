/**
 * ImageCanvas Component
 * Displays OCT image with annotations (disc, fovea, GA regions)
 * Supports interactive fovea adjustment and GA region selection
 */
import React, { useRef, useEffect, useState } from 'react';
import type { ImageAnalysis } from '../types/api';

interface ImageCanvasProps {
  imageAnalysis: ImageAnalysis | null;
  onFoveaClick?: (x: number, y: number) => void;
  onGARegionClick?: (regionIndex: number) => void;
  foveaConfirmed?: boolean;
}

/**
 * Helper function to draw annotated OCT image on a canvas
 * Used by both inline canvas and modal canvas
 */
function drawAnnotatedImage(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  imageAnalysis: ImageAnalysis,
  scale: number,
  hoveredRegionIndex: number | null,
  foveaConfirmed: boolean
): void {
  // Draw image
  ctx.drawImage(image, 0, 0, image.width * scale, image.height * scale);

  // Draw disc (red vertical line)
  if (imageAnalysis.disc) {
    ctx.strokeStyle = 'rgb(255, 0, 0)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(
      imageAnalysis.disc.disc_center_x * scale,
      imageAnalysis.disc.disc_top_y * scale
    );
    ctx.lineTo(
      imageAnalysis.disc.disc_center_x * scale,
      imageAnalysis.disc.disc_bottom_y * scale
    );
    ctx.stroke();
  }

  // Draw fovea (green circle)
  if (imageAnalysis.fovea) {
    ctx.fillStyle = 'rgb(0, 255, 0)';
    ctx.beginPath();
    ctx.arc(
      imageAnalysis.fovea.fovea_x * scale,
      imageAnalysis.fovea.fovea_y * scale,
      10,
      0,
      2 * Math.PI
    );
    ctx.fill();

    // White border
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw GA regions (filled masks with outlines for visibility)
  if (imageAnalysis.gaRegions?.regions && foveaConfirmed) {
    imageAnalysis.gaRegions.regions.forEach((region, index) => {
      // Defensive check: ensure region is valid array with points
      if (!region || !Array.isArray(region) || region.length === 0) {
        return;
      }

      const isSelected = imageAnalysis.selectedGARegionIndex === index;
      const isHovered = hoveredRegionIndex === index;

      // Build the path
      ctx.beginPath();
      region.forEach((point, i) => {
        // Defensive check: ensure point has x and y coordinates
        if (!point || point.length < 2) return;
        
        const x = point[0] * scale;
        const y = point[1] * scale;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();

      // Fill FIRST (always fill all regions for visibility)
      if (isSelected) {
        // Selected: cyan fill (more opaque)
        ctx.fillStyle = 'rgba(0, 255, 255, 0.35)';
      } else if (isHovered) {
        // Hovered: bright yellow fill
        ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
      } else {
        // Unselected: softer yellow fill
        ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
      }
      ctx.fill();

      // Stroke SECOND (draw outline on top)
      ctx.strokeStyle = isSelected ? 'rgb(0, 255, 255)' : 'rgb(255, 255, 0)';
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();
    });
  }

  // Draw distance measurement line (cyan)
  if (imageAnalysis.distance && imageAnalysis.fovea) {
    ctx.strokeStyle = 'rgb(0, 255, 255)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(
      imageAnalysis.fovea.fovea_x * scale,
      imageAnalysis.fovea.fovea_y * scale
    );
    ctx.lineTo(
      imageAnalysis.distance.nearest_ga_point_x * scale,
      imageAnalysis.distance.nearest_ga_point_y * scale
    );
    ctx.stroke();
  }
}

export const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageAnalysis,
  onFoveaClick,
  onGARegionClick,
  foveaConfirmed = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modalCanvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [hoveredRegionIndex, setHoveredRegionIndex] = useState<number | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  // Load image
  useEffect(() => {
    if (!imageAnalysis?.imageUrl) {
      setImage(null);
      return;
    }

    const img = new Image();
    img.onload = () => {
      setImage(img);
    };
    img.onerror = (error) => {
      console.error('Failed to load image:', error);
      setImage(null);
    };
    img.src = imageAnalysis.imageUrl;

    // Cleanup on unmount or URL change
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageAnalysis?.imageUrl]);

  // Draw canvas
  useEffect(() => {
    if (!canvasRef.current || !image || !imageAnalysis) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to fit container (max 800px width)
    const maxWidth = 800;
    const newScale = Math.min(maxWidth / image.width, 1);

    canvas.width = image.width * newScale;
    canvas.height = image.height * newScale;

    // Use helper function to draw everything
    drawAnnotatedImage(ctx, image, imageAnalysis, newScale, hoveredRegionIndex, foveaConfirmed);
  }, [image, imageAnalysis, hoveredRegionIndex, foveaConfirmed]);

  // Draw modal canvas when modal is open
  useEffect(() => {
    if (!modalOpen || !modalCanvasRef.current || !image || !imageAnalysis) return;

    const canvas = modalCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate scale to fit viewport (90% of viewport size)
    const scaleW = (window.innerWidth * 0.9) / image.width;
    const scaleH = (window.innerHeight * 0.9) / image.height;
    const modalScale = Math.min(scaleW, scaleH);

    canvas.width = image.width * modalScale;
    canvas.height = image.height * modalScale;

    // Use helper function to draw everything
    drawAnnotatedImage(ctx, image, imageAnalysis, modalScale, null, false);
  }, [modalOpen, image, imageAnalysis, foveaConfirmed]);

  // Handle Escape key to close modal
  useEffect(() => {
    if (!modalOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setModalOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [modalOpen]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis || !image) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    // Account for both canvas internal scaling AND CSS scaling
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // GATING LOGIC: After fovea is confirmed, ONLY allow GA selection
    if (foveaConfirmed) {
      // Only check GA regions, do NOT fall through to fovea adjustment
      if (imageAnalysis.gaRegions?.regions && onGARegionClick) {
        for (let i = 0; i < imageAnalysis.gaRegions.regions.length; i++) {
          const region = imageAnalysis.gaRegions.regions[i];
          if (region && isPointInPolygon(x, y, region)) {
            onGARegionClick(i);
            return;
          }
        }
      }
      // Fovea is confirmed - do not allow adjustment even if click misses GA
      return;
    }

    // BEFORE confirmation: Allow fovea adjustment
    if (onFoveaClick) {
      onFoveaClick(x, y);
    }
  };

  // Handle modal canvas click
  const handleModalCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!modalCanvasRef.current || !imageAnalysis || !image || !onFoveaClick) return;

    const canvas = modalCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Place fovea at clicked position
    onFoveaClick(x, y);

    // Immediately redraw the modal canvas to show the updated fovea position
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const scaleW = (window.innerWidth * 0.9) / image.width;
      const scaleH = (window.innerHeight * 0.9) / image.height;
      const modalScale = Math.min(scaleW, scaleH);
      
      // Create updated imageAnalysis with new fovea position
      const updatedAnalysis: ImageAnalysis = {
        ...imageAnalysis,
        fovea: imageAnalysis.fovea ? {
          ...imageAnalysis.fovea,
          fovea_x: x,
          fovea_y: y,
        } : undefined,
      };
      
      drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, null, false);
    }
  };

  // Handle mouse move for hover effects
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !image || !imageAnalysis?.gaRegions?.regions) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Check which GA region is hovered
    let foundIndex: number | null = null;
    for (let i = 0; i < imageAnalysis.gaRegions.regions.length; i++) {
      const region = imageAnalysis.gaRegions.regions[i];
      if (region && isPointInPolygon(x, y, region)) {
        foundIndex = i;
        break;
      }
    }

    if (foundIndex !== hoveredRegionIndex) {
      setHoveredRegionIndex(foundIndex);
    }
  };

  if (!imageAnalysis) {
    return (
      <div className="card flex items-center justify-center h-96">
        <p className="text-gray-400">No image uploaded</p>
      </div>
    );
  }

  return (
    <div className="card">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={() => setHoveredRegionIndex(null)}
        className="cursor-pointer border border-gray-200 rounded"
        style={{ maxWidth: '100%' }}
      />

      {/* Status Text */}
      {imageAnalysis && (
        <div className="mt-4 space-y-2">
          {imageAnalysis.disc && (
            <p className="text-sm text-gray-600">
              ✓ Disc detected: {imageAnalysis.disc.disc_height_pixels.toFixed(1)} px = 1800 µm
            </p>
          )}
          {imageAnalysis.fovea && (
            <>
              <p className="text-sm text-gray-600">
                ✓ Fovea: {imageAnalysis.fovea.detection_method} ({imageAnalysis.fovea.eye_side})
              </p>
              {!foveaConfirmed && (
                <p className="text-sm text-blue-600 font-semibold">
                  👆 Click on image to adjust fovea location, then confirm below
                </p>
              )}
            </>
          )}
          {imageAnalysis.gaRegions && (
            <>
              <p className="text-sm text-gray-600">
                ✓ GA regions: {imageAnalysis.gaRegions.region_count} detected
              </p>
              {foveaConfirmed && imageAnalysis.gaRegions.region_count > 0 && (
                <p className="text-sm text-blue-600 font-semibold">
                  👆 Click on a highlighted GA region to select it
                </p>
              )}
            </>
          )}
          {imageAnalysis.distance && (
            <p className="text-sm font-semibold text-blue-600">
              Distance: {imageAnalysis.distance.distance_microns.toFixed(1)} µm
            </p>
          )}
        </div>
      )}

      {/* Expand Image Button - only shown before fovea confirmation */}
      {imageAnalysis?.fovea && !foveaConfirmed && (
        <button
          onClick={() => setModalOpen(true)}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors"
        >
          🔍 Expand Image for Precise Fovea Placement
        </button>
      )}

      {/* Full-Screen Modal */}
      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80"
          onClick={(e) => {
            // Close modal if clicking backdrop
            if (e.target === e.currentTarget) {
              setModalOpen(false);
            }
          }}
        >
          <div className="relative">
            {/* Close button */}
            <button
              onClick={() => setModalOpen(false)}
              className="absolute -top-12 right-0 text-white text-2xl font-bold hover:text-gray-300 transition-colors"
              aria-label="Close modal"
            >
              ✕ Close
            </button>
            
            {/* Modal Canvas */}
            <canvas
              ref={modalCanvasRef}
              onClick={handleModalCanvasClick}
              className="border-4 border-white rounded cursor-crosshair"
              style={{ maxWidth: '90vw', maxHeight: '90vh' }}
            />
            
            {/* Instructions */}
            <p className="text-white text-center mt-4 text-lg">
              Click on the image to place the fovea at the exact position
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Check if a point is inside a polygon using ray casting algorithm
 */
function isPointInPolygon(x: number, y: number, polygon: Array<[number, number]>): boolean {
  // Defensive checks
  if (!polygon || !Array.isArray(polygon) || polygon.length < 3) {
    return false;
  }

  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const pointI = polygon[i];
    const pointJ = polygon[j];

    // Ensure points are valid
    if (!pointI || !pointJ || pointI.length < 2 || pointJ.length < 2) {
      continue;
    }

    const xi = pointI[0];
    const yi = pointI[1];
    const xj = pointJ[0];
    const yj = pointJ[1];

    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
