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

export const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageAnalysis,
  onFoveaClick,
  onGARegionClick,
  foveaConfirmed = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [scale, setScale] = useState<number>(1);
  const [hoveredRegionIndex, setHoveredRegionIndex] = useState<number | null>(null);

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
    setScale(newScale);

    canvas.width = image.width * newScale;
    canvas.height = image.height * newScale;

    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // Draw disc (red vertical line)
    if (imageAnalysis.disc) {
      ctx.strokeStyle = 'rgb(255, 0, 0)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(
        imageAnalysis.disc.disc_center_x * newScale,
        imageAnalysis.disc.disc_top_y * newScale
      );
      ctx.lineTo(
        imageAnalysis.disc.disc_center_x * newScale,
        imageAnalysis.disc.disc_bottom_y * newScale
      );
      ctx.stroke();
    }

    // Draw fovea (green circle)
    if (imageAnalysis.fovea) {
      ctx.fillStyle = 'rgb(0, 255, 0)';
      ctx.beginPath();
      ctx.arc(
        imageAnalysis.fovea.fovea_x * newScale,
        imageAnalysis.fovea.fovea_y * newScale,
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
    console.log('[ImageCanvas] GA Visibility State:', {
      hasGARegions: !!imageAnalysis.gaRegions,
      regionCount: imageAnalysis.gaRegions?.region_count,
      regionsArray: imageAnalysis.gaRegions?.regions?.length,
      foveaConfirmed,
    });
    
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
          
          const x = point[0] * newScale;
          const y = point[1] * newScale;
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
        imageAnalysis.fovea.fovea_x * newScale,
        imageAnalysis.fovea.fovea_y * newScale
      );
      ctx.lineTo(
        imageAnalysis.distance.nearest_ga_point_x * newScale,
        imageAnalysis.distance.nearest_ga_point_y * newScale
      );
      ctx.stroke();
    }
  }, [image, imageAnalysis, hoveredRegionIndex, foveaConfirmed]);

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

  // Handle mouse move for hover effects
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis?.gaRegions?.regions || !image) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Check which region is hovered
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
