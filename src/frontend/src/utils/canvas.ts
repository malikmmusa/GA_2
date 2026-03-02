import type { MouseEvent } from 'react';

export interface ImageCoordinates {
  x: number;
  y: number;
}

/**
 * Convert mouse coordinates from rendered canvas space to original image space.
 */
export function getImageCoordinates(
  event: MouseEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
  image: HTMLImageElement
): ImageCoordinates {
  const rect = canvas.getBoundingClientRect();
  const scaleX = image.width / rect.width;
  const scaleY = image.height / rect.height;

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}
