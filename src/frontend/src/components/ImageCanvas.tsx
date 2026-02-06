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
}

export const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageAnalysis,
  onFoveaClick,
  onGARegionClick,
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

    // Draw GA regions (yellow outlines)
    if (imageAnalysis.gaRegions?.regions) {
      imageAnalysis.gaRegions.regions.forEach((region, index) => {
        // Defensive check: ensure region is valid array with points
        if (!region || !Array.isArray(region) || region.length === 0) {
          return;
        }

        const isSelected = imageAnalysis.selectedGARegionIndex === index;
        const isHovered = hoveredRegionIndex === index;

        ctx.strokeStyle = isSelected ? 'rgb(0, 255, 255)' : 'rgb(255, 255, 0)';
        ctx.lineWidth = isHovered ? 3 : 2;

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
        ctx.stroke();

        // Fill with semi-transparent color if selected
        if (isSelected) {
          ctx.fillStyle = 'rgba(0, 255, 255, 0.2)';
          ctx.fill();
        }
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
  }, [image, imageAnalysis, hoveredRegionIndex]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis || !image) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

    // Check if clicked on a GA region
    if (imageAnalysis.gaRegions?.regions && onGARegionClick) {
      for (let i = 0; i < imageAnalysis.gaRegions.regions.length; i++) {
        const region = imageAnalysis.gaRegions.regions[i];
        if (region && isPointInPolygon(x, y, region)) {
          onGARegionClick(i);
          return;
        }
      }
    }

    // Otherwise, treat as fovea adjustment click
    if (onFoveaClick) {
      onFoveaClick(x, y);
    }
  };

  // Handle mouse move for hover effects
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis?.gaRegions?.regions || !image) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

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
            <p className="text-sm text-gray-600">
              ✓ Fovea: {imageAnalysis.fovea.detection_method} ({imageAnalysis.fovea.eye_side})
            </p>
          )}
          {imageAnalysis.gaRegions && (
            <p className="text-sm text-gray-600">
              ✓ GA regions: {imageAnalysis.gaRegions.region_count} detected
            </p>
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
