/**
 * ImageCanvas Component
 * Displays OCT image with annotations (disc, fovea, GA regions)
 * Supports interactive fovea adjustment and GA region selection
 */
import React, { useRef, useEffect, useState } from 'react';
import type { ImageAnalysis } from '../types/api';

// Constants
const FOVEA_RADIUS = 4; // Display radius in pixels for fovea marker
const FOVEA_HIT_RADIUS = 12; // Hit-test radius in image pixels for drag interaction
const DISC_BRACKET_HALF_WIDTH = 8; // Half-width of bracket/serif marks at disc endpoints
const DISC_HIT_RADIUS = 12; // Hit-test radius for disc handles
const DISC_BODY_HIT_RADIUS = 10; // Hit-test radius for disc body (horizontal)
const MIN_DISC_HEIGHT = 30; // Minimum disc height in pixels

interface ImageCanvasProps {
  imageAnalysis: ImageAnalysis | null;
  onFoveaClick?: (x: number, y: number) => void;
  onGARegionClick?: (regionIndex: number) => void;
  onGAAreaClick?: (x: number, y: number) => void;
  onDiscAdjust?: (centerX: number, topY: number, bottomY: number) => void;
  foveaConfirmed?: boolean;
  isProcessingGA?: boolean;
}

/**
 * Check if a point is near the fovea marker (for drag interaction)
 */
function isPointNearFovea(
  x: number,
  y: number,
  imageAnalysis: ImageAnalysis | null,
  hitRadius: number = FOVEA_HIT_RADIUS
): boolean {
  if (!imageAnalysis?.fovea) return false;
  
  const dx = x - imageAnalysis.fovea.fovea_x;
  const dy = y - imageAnalysis.fovea.fovea_y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  return distance <= hitRadius;
}

/**
 * Check which part of the disc line is being clicked/hovered
 */
function getDiscHitZone(
  x: number,
  y: number,
  disc: { disc_center_x: number; disc_top_y: number; disc_bottom_y: number } | undefined
): 'top' | 'bottom' | 'body' | null {
  if (!disc) return null;

  const { disc_center_x, disc_top_y, disc_bottom_y } = disc;

  // Check top handle
  const distToTop = Math.sqrt(
    Math.pow(x - disc_center_x, 2) + Math.pow(y - disc_top_y, 2)
  );
  if (distToTop <= DISC_HIT_RADIUS) {
    return 'top';
  }

  // Check bottom handle
  const distToBottom = Math.sqrt(
    Math.pow(x - disc_center_x, 2) + Math.pow(y - disc_bottom_y, 2)
  );
  if (distToBottom <= DISC_HIT_RADIUS) {
    return 'bottom';
  }

  // Check body (line between handles)
  const horizontalDist = Math.abs(x - disc_center_x);
  if (horizontalDist <= DISC_BODY_HIT_RADIUS &&
      y >= disc_top_y - 5 &&
      y <= disc_bottom_y + 5) {
    return 'body';
  }

  return null;
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
  foveaConfirmed: boolean,
  hoveredDiscZone: 'top' | 'bottom' | 'body' | null = null
): void {
  // Draw image
  ctx.drawImage(image, 0, 0, image.width * scale, image.height * scale);

  // Draw disc (red vertical line with drag handles before fovea confirmation)
  if (imageAnalysis.disc) {
    const discCenterX = imageAnalysis.disc.disc_center_x * scale;
    const discTopY = imageAnalysis.disc.disc_top_y * scale;
    const discBottomY = imageAnalysis.disc.disc_bottom_y * scale;

    // Draw line
    ctx.strokeStyle = 'rgb(255, 0, 0)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(discCenterX, discTopY);
    ctx.lineTo(discCenterX, discBottomY);
    ctx.stroke();

    // Draw drag handles as bracket/serif marks (before fovea confirmation only)
    if (!foveaConfirmed) {
      const bracketHalfWidth = DISC_BRACKET_HALF_WIDTH;
      
      // Top handle (horizontal bracket mark)
      const topHovered = hoveredDiscZone === 'top';
      ctx.strokeStyle = 'rgb(255, 0, 0)';
      ctx.lineWidth = topHovered ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(discCenterX - bracketHalfWidth, discTopY);
      ctx.lineTo(discCenterX + bracketHalfWidth, discTopY);
      ctx.stroke();
      
      // White outline for top bracket
      ctx.strokeStyle = 'white';
      ctx.lineWidth = topHovered ? 4 : 3;
      ctx.globalAlpha = topHovered ? 0.6 : 0.4;
      ctx.beginPath();
      ctx.moveTo(discCenterX - bracketHalfWidth, discTopY);
      ctx.lineTo(discCenterX + bracketHalfWidth, discTopY);
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      // Bottom handle (horizontal bracket mark)
      const bottomHovered = hoveredDiscZone === 'bottom';
      ctx.strokeStyle = 'rgb(255, 0, 0)';
      ctx.lineWidth = bottomHovered ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(discCenterX - bracketHalfWidth, discBottomY);
      ctx.lineTo(discCenterX + bracketHalfWidth, discBottomY);
      ctx.stroke();
      
      // White outline for bottom bracket
      ctx.strokeStyle = 'white';
      ctx.lineWidth = bottomHovered ? 4 : 3;
      ctx.globalAlpha = bottomHovered ? 0.6 : 0.4;
      ctx.beginPath();
      ctx.moveTo(discCenterX - bracketHalfWidth, discBottomY);
      ctx.lineTo(discCenterX + bracketHalfWidth, discBottomY);
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }
  }

  // Draw fovea (green circle)
  if (imageAnalysis.fovea) {
    ctx.fillStyle = 'rgb(0, 255, 0)';
    ctx.beginPath();
    ctx.arc(
      imageAnalysis.fovea.fovea_x * scale,
      imageAnalysis.fovea.fovea_y * scale,
      FOVEA_RADIUS,
      0,
      2 * Math.PI
    );
    ctx.fill();

    // White border
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw GA region (only the selected one, if any)
  if (imageAnalysis.gaRegions?.regions && foveaConfirmed && imageAnalysis.selectedGARegionIndex !== undefined) {
    const selectedIndex = imageAnalysis.selectedGARegionIndex;
    const region = imageAnalysis.gaRegions.regions[selectedIndex];
    
    // Defensive check: ensure region is valid array with points
    if (region && Array.isArray(region) && region.length > 0) {
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

      // Fill with cyan (selected region)
      ctx.fillStyle = 'rgba(0, 255, 255, 0.35)';
      ctx.fill();

      // Stroke with cyan outline
      ctx.strokeStyle = 'rgb(0, 255, 255)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
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
  onGAAreaClick,
  onDiscAdjust,
  foveaConfirmed = false,
  isProcessingGA = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modalCanvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [hoveredRegionIndex, setHoveredRegionIndex] = useState<number | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  
  // Fovea drag state
  const [isDraggingFovea, setIsDraggingFovea] = useState(false);
  const [isHoveringFovea, setIsHoveringFovea] = useState(false);
  const isDraggingRef = useRef(false);
  const wasDraggingRef = useRef(false);

  // Disc drag state
  const [isDraggingDisc, setIsDraggingDisc] = useState(false);
  const [discDragType, setDiscDragType] = useState<'top' | 'bottom' | 'body' | null>(null);
  const [discDragOffset, setDiscDragOffset] = useState({ dx: 0, dy: 0 });
  const [hoveredDiscZone, setHoveredDiscZone] = useState<'top' | 'bottom' | 'body' | null>(null);
  const isDraggingDiscRef = useRef(false);
  const wasDraggingDiscRef = useRef(false);

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
    drawAnnotatedImage(ctx, image, imageAnalysis, newScale, hoveredRegionIndex, foveaConfirmed, hoveredDiscZone);
  }, [image, imageAnalysis, hoveredRegionIndex, foveaConfirmed, hoveredDiscZone]);

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
    drawAnnotatedImage(ctx, image, imageAnalysis, modalScale, null, false, null);
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

  // Handle canvas mouse down
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis || !image || foveaConfirmed) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Priority: disc handles > fovea > disc body
    
    // Check disc hit zones first
    const discZone = getDiscHitZone(x, y, imageAnalysis.disc);
    if (discZone && onDiscAdjust) {
      setIsDraggingDisc(true);
      setDiscDragType(discZone);
      isDraggingDiscRef.current = true;
      wasDraggingDiscRef.current = false;
      
      // For body drag, store offset from center
      if (discZone === 'body' && imageAnalysis.disc) {
        setDiscDragOffset({
          dx: x - imageAnalysis.disc.disc_center_x,
          dy: y - ((imageAnalysis.disc.disc_top_y + imageAnalysis.disc.disc_bottom_y) / 2),
        });
      }
      
      e.preventDefault();
      return;
    }

    // Check if clicking near fovea to start drag
    if (isPointNearFovea(x, y, imageAnalysis)) {
      setIsDraggingFovea(true);
      isDraggingRef.current = true;
      wasDraggingRef.current = false;
      e.preventDefault(); // Prevent text selection
    }
  };

  // Handle canvas mouse up
  const handleCanvasMouseUp = () => {
    if (isDraggingRef.current) {
      wasDraggingRef.current = true;
      // Small delay to let click event know we were dragging
      setTimeout(() => {
        wasDraggingRef.current = false;
      }, 10);
    }
    if (isDraggingDiscRef.current) {
      wasDraggingDiscRef.current = true;
      setTimeout(() => {
        wasDraggingDiscRef.current = false;
      }, 10);
    }
    setIsDraggingFovea(false);
    isDraggingRef.current = false;
    setIsDraggingDisc(false);
    setDiscDragType(null);
    isDraggingDiscRef.current = false;
  };

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis || !image) return;

    // Don't process click if we just finished dragging
    if (wasDraggingRef.current || wasDraggingDiscRef.current) return;

    // Don't allow clicks during GA processing
    if (isProcessingGA) return;

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
      // Restrict clicks to en-face region
      if (imageAnalysis.disc?.en_face_split_x !== undefined && x < imageAnalysis.disc.en_face_split_x) {
        console.log('[GA] Click on B-scan side ignored');
        return;
      }

      // Check if click is inside any existing region
      if (imageAnalysis.gaRegions?.regions && onGARegionClick) {
        for (let i = 0; i < imageAnalysis.gaRegions.regions.length; i++) {
          const region = imageAnalysis.gaRegions.regions[i];
          if (region && isPointInPolygon(x, y, region)) {
            console.log(`[GA] Click inside region ${i}`);
            onGARegionClick(i);
            return;
          }
        }
      }

      // Click outside all existing regions - trigger area click for localized segmentation
      if (onGAAreaClick) {
        console.log(`[GA] Click outside existing regions at (${x.toFixed(1)}, ${y.toFixed(1)})`);
        onGAAreaClick(x, y);
      }
      
      return;
    }

    // BEFORE confirmation: Allow fovea adjustment
    if (onFoveaClick) {
      onFoveaClick(x, y);
    }
  };

  // Modal canvas drag state
  const [isDraggingModalFovea, setIsDraggingModalFovea] = useState(false);
  const [isHoveringModalFovea, setIsHoveringModalFovea] = useState(false);
  const isDraggingModalRef = useRef(false);
  const wasDraggingModalRef = useRef(false);

  // Modal disc drag state
  const [isDraggingModalDisc, setIsDraggingModalDisc] = useState(false);
  const [modalDiscDragType, setModalDiscDragType] = useState<'top' | 'bottom' | 'body' | null>(null);
  const [modalDiscDragOffset, setModalDiscDragOffset] = useState({ dx: 0, dy: 0 });
  const [hoveredModalDiscZone, setHoveredModalDiscZone] = useState<'top' | 'bottom' | 'body' | null>(null);
  const isDraggingModalDiscRef = useRef(false);
  const wasDraggingModalDiscRef = useRef(false);

  // Handle modal canvas mouse down
  const handleModalCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!modalCanvasRef.current || !imageAnalysis || !image) return;

    const canvas = modalCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Priority: disc handles > fovea > disc body
    
    // Check disc hit zones first
    const discZone = getDiscHitZone(x, y, imageAnalysis.disc);
    if (discZone && onDiscAdjust) {
      setIsDraggingModalDisc(true);
      setModalDiscDragType(discZone);
      isDraggingModalDiscRef.current = true;
      wasDraggingModalDiscRef.current = false;
      
      // For body drag, store offset from center
      if (discZone === 'body' && imageAnalysis.disc) {
        setModalDiscDragOffset({
          dx: x - imageAnalysis.disc.disc_center_x,
          dy: y - ((imageAnalysis.disc.disc_top_y + imageAnalysis.disc.disc_bottom_y) / 2),
        });
      }
      
      e.preventDefault();
      return;
    }

    // Check if clicking near fovea to start drag
    if (isPointNearFovea(x, y, imageAnalysis)) {
      setIsDraggingModalFovea(true);
      isDraggingModalRef.current = true;
      wasDraggingModalRef.current = false;
      e.preventDefault();
    }
  };

  // Handle modal canvas mouse up
  const handleModalCanvasMouseUp = () => {
    if (isDraggingModalRef.current) {
      wasDraggingModalRef.current = true;
      setTimeout(() => {
        wasDraggingModalRef.current = false;
      }, 10);
    }
    if (isDraggingModalDiscRef.current) {
      wasDraggingModalDiscRef.current = true;
      setTimeout(() => {
        wasDraggingModalDiscRef.current = false;
      }, 10);
    }
    setIsDraggingModalFovea(false);
    isDraggingModalRef.current = false;
    setIsDraggingModalDisc(false);
    setModalDiscDragType(null);
    isDraggingModalDiscRef.current = false;
  };

  // Handle modal canvas mouse move
  const handleModalCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!modalCanvasRef.current || !imageAnalysis || !image) return;

    const canvas = modalCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Handle dragging disc
    if (isDraggingModalDiscRef.current && imageAnalysis.disc && onDiscAdjust) {
      const disc = imageAnalysis.disc;
      const enFaceSplitX = disc.en_face_split_x || 0;

      if (modalDiscDragType === 'top') {
        // Drag top handle - clamp to stay above bottom
        const newTopY = Math.max(0, Math.min(y, disc.disc_bottom_y - MIN_DISC_HEIGHT));
        onDiscAdjust(disc.disc_center_x, newTopY, disc.disc_bottom_y);
      } else if (modalDiscDragType === 'bottom') {
        // Drag bottom handle - clamp to stay below top
        const newBottomY = Math.min(image.height, Math.max(y, disc.disc_top_y + MIN_DISC_HEIGHT));
        onDiscAdjust(disc.disc_center_x, disc.disc_top_y, newBottomY);
      } else if (modalDiscDragType === 'body') {
        // Drag body - translate entire line
        const centerX = x - modalDiscDragOffset.dx;
        const centerY = y - modalDiscDragOffset.dy;
        
        const height = disc.disc_bottom_y - disc.disc_top_y;
        let newCenterX = Math.max(enFaceSplitX, Math.min(image.width, centerX));
        let newTopY = Math.max(0, Math.min(image.height - height, centerY - height / 2));
        let newBottomY = newTopY + height;
        
        onDiscAdjust(newCenterX, newTopY, newBottomY);
      }
      wasDraggingModalDiscRef.current = true;
      
      // Immediately redraw
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const scaleW = (window.innerWidth * 0.9) / image.width;
        const scaleH = (window.innerHeight * 0.9) / image.height;
        const modalScale = Math.min(scaleW, scaleH);
        
        // Calculate updated disc values
        let newDisc = { ...disc };
        if (modalDiscDragType === 'top') {
          newDisc.disc_top_y = Math.max(0, Math.min(y, disc.disc_bottom_y - MIN_DISC_HEIGHT));
        } else if (modalDiscDragType === 'bottom') {
          newDisc.disc_bottom_y = Math.min(image.height, Math.max(y, disc.disc_top_y + MIN_DISC_HEIGHT));
        } else if (modalDiscDragType === 'body') {
          const centerX = x - modalDiscDragOffset.dx;
          const centerY = y - modalDiscDragOffset.dy;
          const height = disc.disc_bottom_y - disc.disc_top_y;
          newDisc.disc_center_x = Math.max(enFaceSplitX, Math.min(image.width, centerX));
          newDisc.disc_top_y = Math.max(0, Math.min(image.height - height, centerY - height / 2));
          newDisc.disc_bottom_y = newDisc.disc_top_y + height;
        }
        
        const updatedAnalysis: ImageAnalysis = {
          ...imageAnalysis,
          disc: newDisc,
        };
        
        drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, null, false, modalDiscDragType);
      }
      return;
    }

    // Handle dragging fovea
    if (isDraggingModalRef.current && onFoveaClick) {
      onFoveaClick(x, y);
      wasDraggingModalRef.current = true;
      
      // Immediately redraw
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const scaleW = (window.innerWidth * 0.9) / image.width;
        const scaleH = (window.innerHeight * 0.9) / image.height;
        const modalScale = Math.min(scaleW, scaleH);
        
        const updatedAnalysis: ImageAnalysis = {
          ...imageAnalysis,
          fovea: imageAnalysis.fovea ? {
            ...imageAnalysis.fovea,
            fovea_x: x,
            fovea_y: y,
          } : undefined,
        };
        
        drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, null, false, null);
      }
      return;
    }

    // Check if hovering over disc (for cursor feedback)
    if (imageAnalysis.disc) {
      const discZone = getDiscHitZone(x, y, imageAnalysis.disc);
      if (discZone !== hoveredModalDiscZone) {
        setHoveredModalDiscZone(discZone);
      }
    }

    // Check if hovering over fovea
    if (imageAnalysis.fovea) {
      const hovering = isPointNearFovea(x, y, imageAnalysis);
      if (hovering !== isHoveringModalFovea) {
        setIsHoveringModalFovea(hovering);
      }
    }
  };

  // Handle modal canvas click
  const handleModalCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!modalCanvasRef.current || !imageAnalysis || !image || !onFoveaClick) return;

    // Don't process click if we just finished dragging
    if (wasDraggingModalRef.current || wasDraggingModalDiscRef.current) return;

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
      
      drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, null, false, null);
    }
  };

  // Handle mouse move for hover effects and dragging
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !image || !imageAnalysis) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert from screen pixels to original image pixels
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Handle dragging disc
    if (isDraggingDiscRef.current && imageAnalysis.disc && onDiscAdjust) {
      const disc = imageAnalysis.disc;
      const enFaceSplitX = disc.en_face_split_x || 0;

      if (discDragType === 'top') {
        // Drag top handle - clamp to stay above bottom
        const newTopY = Math.max(0, Math.min(y, disc.disc_bottom_y - MIN_DISC_HEIGHT));
        onDiscAdjust(disc.disc_center_x, newTopY, disc.disc_bottom_y);
      } else if (discDragType === 'bottom') {
        // Drag bottom handle - clamp to stay below top
        const newBottomY = Math.min(image.height, Math.max(y, disc.disc_top_y + MIN_DISC_HEIGHT));
        onDiscAdjust(disc.disc_center_x, disc.disc_top_y, newBottomY);
      } else if (discDragType === 'body') {
        // Drag body - translate entire line
        const centerX = x - discDragOffset.dx;
        const centerY = y - discDragOffset.dy;
        
        const height = disc.disc_bottom_y - disc.disc_top_y;
        let newCenterX = Math.max(enFaceSplitX, Math.min(image.width, centerX));
        let newTopY = Math.max(0, Math.min(image.height - height, centerY - height / 2));
        let newBottomY = newTopY + height;
        
        onDiscAdjust(newCenterX, newTopY, newBottomY);
      }
      wasDraggingDiscRef.current = true;
      return;
    }

    // Handle dragging fovea
    if (isDraggingRef.current && onFoveaClick) {
      onFoveaClick(x, y);
      wasDraggingRef.current = true;
      return;
    }

    // Check if hovering over disc (for cursor feedback)
    if (!foveaConfirmed && imageAnalysis.disc) {
      const discZone = getDiscHitZone(x, y, imageAnalysis.disc);
      if (discZone !== hoveredDiscZone) {
        setHoveredDiscZone(discZone);
      }
    }

    // Check if hovering over fovea (for cursor feedback)
    if (!foveaConfirmed && imageAnalysis.fovea) {
      const hovering = isPointNearFovea(x, y, imageAnalysis);
      if (hovering !== isHoveringFovea) {
        setIsHoveringFovea(hovering);
      }
    }

    // No hover effects for GA regions since we only show the selected one
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
        onMouseDown={handleCanvasMouseDown}
        onMouseUp={handleCanvasMouseUp}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={() => {
          setIsHoveringFovea(false);
          setHoveredDiscZone(null);
          handleCanvasMouseUp();
        }}
        className="cursor-pointer border border-gray-200 rounded"
        style={{ 
          maxWidth: '100%',
          cursor: isDraggingDisc 
            ? (discDragType === 'body' ? 'move' : 'ns-resize')
            : isDraggingFovea 
            ? 'grabbing' 
            : hoveredDiscZone && !foveaConfirmed
            ? (hoveredDiscZone === 'body' ? 'move' : 'ns-resize')
            : isHoveringFovea && !foveaConfirmed 
            ? 'grab' 
            : 'pointer'
        }}
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
                  👆 Click or drag the fovea marker to adjust its location, then confirm below
                </p>
              )}
            </>
          )}
          {imageAnalysis.gaRegions && foveaConfirmed && (
            <>
              {imageAnalysis.selectedGARegionIndex === undefined ? (
                <p className="text-sm text-blue-600 font-semibold">
                  👆 Click on the GA area you want to analyze
                </p>
              ) : (
                <p className="text-sm text-gray-600">
                  ✓ GA region selected
                </p>
              )}
              {isProcessingGA && (
                <p className="text-sm text-orange-600 font-semibold">
                  ⏳ Analyzing selected area...
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
          🔍 Expand Image for Precise Adjustment
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
              onMouseDown={handleModalCanvasMouseDown}
              onMouseUp={handleModalCanvasMouseUp}
              onMouseMove={handleModalCanvasMouseMove}
              onMouseLeave={() => {
                setIsHoveringModalFovea(false);
                setHoveredModalDiscZone(null);
                handleModalCanvasMouseUp();
              }}
              className="border-4 border-white rounded"
              style={{ 
                maxWidth: '90vw', 
                maxHeight: '90vh',
                cursor: isDraggingModalDisc
                  ? (modalDiscDragType === 'body' ? 'move' : 'ns-resize')
                  : isDraggingModalFovea 
                  ? 'grabbing'
                  : hoveredModalDiscZone
                  ? (hoveredModalDiscZone === 'body' ? 'move' : 'ns-resize')
                  : isHoveringModalFovea 
                  ? 'grab' 
                  : 'crosshair'
              }}
            />
            
            {/* Instructions */}
            <p className="text-white text-center mt-4 text-lg">
              Drag fovea (green) or disc bracket handles (red) to adjust
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
