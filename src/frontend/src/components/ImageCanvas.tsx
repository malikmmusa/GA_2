/**
 * ImageCanvas Component
 * Displays OCT image with annotations (disc, fovea, GA regions)
 * Supports interactive fovea adjustment and GA region selection
 */
import React, { useRef, useEffect, useState } from 'react';
import type { ImageAnalysis } from '../types/api';
import { getImageCoordinates } from '../utils/canvas';

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
  onManualGAPointClick?: (x: number, y: number) => void;
  onManualGAModeToggle?: () => void;
  onDiscAdjust?: (centerX: number, topY: number, bottomY: number) => void;
  foveaConfirmed?: boolean;
  gaConfirmed?: boolean;
  manualGAMode?: boolean;
  isProcessingGA?: boolean;
  registrationSuggestion?: {
    fovea_x: number;
    fovea_y: number;
  };
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
  const distance = Math.hypot(dx, dy);
  
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
  const distToTop = Math.hypot(x - disc_center_x, y - disc_top_y);
  if (distToTop <= DISC_HIT_RADIUS) {
    return 'top';
  }

  // Check bottom handle
  const distToBottom = Math.hypot(x - disc_center_x, y - disc_bottom_y);
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
  foveaConfirmed: boolean,
  hoveredDiscZone: 'top' | 'bottom' | 'body' | null = null,
  registrationSuggestion?: { fovea_x: number; fovea_y: number },
  manualGAMode: boolean = false
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

  // Draw registration suggestion (yellow circle) until user manually places fovea
  if (registrationSuggestion && !foveaConfirmed && imageAnalysis.fovea?.detection_method !== 'manual') {
    ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
    ctx.beginPath();
    ctx.arc(
      registrationSuggestion.fovea_x * scale,
      registrationSuggestion.fovea_y * scale,
      FOVEA_RADIUS + 2,
      0,
      2 * Math.PI
    );
    ctx.fill();

    // Orange border
    ctx.strokeStyle = 'rgb(255, 165, 0)';
    ctx.lineWidth = 2;
    ctx.stroke();
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

  // Draw GA region (only the selected one, if any; skip when manual mode or manual point)
  if (
    imageAnalysis.gaRegions?.regions &&
    foveaConfirmed &&
    imageAnalysis.selectedGARegionIndex !== undefined &&
    !imageAnalysis.isManualGAPoint &&
    !manualGAMode
  ) {
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

  // Draw distance measurement line (cyan); hide when in manual mode without a point set
  if (
    imageAnalysis.distance &&
    imageAnalysis.fovea &&
    !(manualGAMode && !imageAnalysis.isManualGAPoint)
  ) {
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
  onManualGAPointClick,
  onManualGAModeToggle,
  onDiscAdjust,
  foveaConfirmed = false,
  gaConfirmed = false,
  manualGAMode = false,
  isProcessingGA = false,
  registrationSuggestion,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modalCanvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
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
    drawAnnotatedImage(
      ctx,
      image,
      imageAnalysis,
      newScale,
      foveaConfirmed,
      hoveredDiscZone,
      registrationSuggestion,
      manualGAMode
    );
  }, [image, imageAnalysis, foveaConfirmed, hoveredDiscZone, registrationSuggestion, manualGAMode]);

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
    drawAnnotatedImage(
      ctx,
      image,
      imageAnalysis,
      modalScale,
      foveaConfirmed,
      null,
      registrationSuggestion,
      manualGAMode
    );
  }, [
    modalOpen,
    image,
    imageAnalysis,
    foveaConfirmed,
    registrationSuggestion,
    manualGAMode,
  ]);

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
    const { x, y } = getImageCoordinates(e, canvas, image);

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

  // Handle clicks once fovea is confirmed (GA interaction mode)
  const handleConfirmedFoveaInteraction = (x: number, y: number): boolean => {
    if (!foveaConfirmed) return false;
    const enFaceSplitX = imageAnalysis?.disc?.en_face_split_x;
    const regions = imageAnalysis?.gaRegions?.regions;

    // Restrict clicks to en-face region
    if (enFaceSplitX !== undefined && x < enFaceSplitX) {
      return true;
    }

    if (manualGAMode && onManualGAPointClick) {
      onManualGAPointClick(x, y);
      return true;
    }

    // Check if click is inside any existing region
    if (regions && onGARegionClick) {
      for (let i = 0; i < regions.length; i++) {
        const region = regions[i];
        if (region && isPointInPolygon(x, y, region)) {
          onGARegionClick(i);
          return true;
        }
      }
    }

    // Click outside all existing regions - trigger area click for localized segmentation
    if (onGAAreaClick) {
      onGAAreaClick(x, y);
    }

    return true;
  };

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageAnalysis || !image) return;

    // Don't process click if we just finished dragging
    if (wasDraggingRef.current || wasDraggingDiscRef.current) return;

    // Don't allow clicks during GA processing
    if (isProcessingGA) return;

    const canvas = canvasRef.current;
    const { x, y } = getImageCoordinates(e, canvas, image);

    // GATING LOGIC: After fovea is confirmed, ONLY allow GA selection
    if (handleConfirmedFoveaInteraction(x, y)) {
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
    if (!modalCanvasRef.current || !imageAnalysis || !image || foveaConfirmed) return;

    const canvas = modalCanvasRef.current;
    const { x, y } = getImageCoordinates(e, canvas, image);

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
    const { x, y } = getImageCoordinates(e, canvas, image);

    // Handle dragging disc
    if (isDraggingModalDiscRef.current && imageAnalysis.disc && onDiscAdjust) {
      const disc = imageAnalysis.disc;
      const enFaceSplitX = disc.en_face_split_x ?? 0;

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
        const newCenterX = Math.max(enFaceSplitX, Math.min(image.width, centerX));
        const newTopY = Math.max(0, Math.min(image.height - height, centerY - height / 2));
        const newBottomY = newTopY + height;
        
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
        const newDisc = { ...disc };
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
        
        drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, false, modalDiscDragType);
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
        
        drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, false, null);
      }
      return;
    }

    // Check if hovering over disc (for cursor feedback)
    if (!foveaConfirmed && imageAnalysis.disc) {
      const discZone = getDiscHitZone(x, y, imageAnalysis.disc);
      if (discZone !== hoveredModalDiscZone) {
        setHoveredModalDiscZone(discZone);
      }
    }

    // Check if hovering over fovea
    if (!foveaConfirmed && imageAnalysis.fovea) {
      const hovering = isPointNearFovea(x, y, imageAnalysis);
      if (hovering !== isHoveringModalFovea) {
        setIsHoveringModalFovea(hovering);
      }
    }
  };

  // Handle modal canvas click
  const handleModalCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!modalCanvasRef.current || !imageAnalysis || !image) return;

    // Don't process click if we just finished dragging
    if (wasDraggingModalRef.current || wasDraggingModalDiscRef.current) return;

    // Don't allow clicks during GA processing
    if (isProcessingGA) return;

    const canvas = modalCanvasRef.current;
    const { x, y } = getImageCoordinates(e, canvas, image);

    // GATING LOGIC: After fovea is confirmed, ONLY allow GA selection
    if (handleConfirmedFoveaInteraction(x, y)) {
      return;
    }

    if (!onFoveaClick) return;

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
      
      drawAnnotatedImage(ctx, image, updatedAnalysis, modalScale, false, null);
    }
  };

  // Handle mouse move for hover effects and dragging
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !image || !imageAnalysis) return;

    const canvas = canvasRef.current;
    const { x, y } = getImageCoordinates(e, canvas, image);

    // Handle dragging disc
    if (isDraggingDiscRef.current && imageAnalysis.disc && onDiscAdjust) {
      const disc = imageAnalysis.disc;
      const enFaceSplitX = disc.en_face_split_x ?? 0;

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
        const newCenterX = Math.max(enFaceSplitX, Math.min(image.width, centerX));
        const newTopY = Math.max(0, Math.min(image.height - height, centerY - height / 2));
        const newBottomY = newTopY + height;
        
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

  const canExpandForFoveaAdjustment = !!imageAnalysis.fovea && !foveaConfirmed;
  const canExpandForGASelection =
    !!imageAnalysis.fovea &&
    foveaConfirmed &&
    !!imageAnalysis.gaRegions &&
    !gaConfirmed;
  const canExpandImage = canExpandForFoveaAdjustment || canExpandForGASelection;
  const shouldShowManualGAButton = !!imageAnalysis.gaRegions && foveaConfirmed && !gaConfirmed;
  const manualGAButtonBaseClass =
    'absolute top-2 left-2 z-10 px-3 py-1.5 text-xs font-semibold rounded-full border shadow-sm transition-colors';

  const getManualGAButtonVariantClass = (variant: 'inline' | 'modal'): string => {
    if (manualGAMode) {
      return variant === 'modal'
        ? 'bg-gray-800/90 hover:bg-gray-900 text-white border-gray-600'
        : 'bg-gray-700/85 hover:bg-gray-800 text-white border-gray-600';
    }

    return variant === 'modal'
      ? 'bg-white/90 hover:bg-white text-gray-900 border-gray-300'
      : 'bg-white/85 hover:bg-white text-gray-800 border-gray-300';
  };

  const renderManualGAButton = (variant: 'inline' | 'modal') => {
    if (!shouldShowManualGAButton) return null;

    return (
      <button
        type="button"
        onClick={onManualGAModeToggle}
        className={`${manualGAButtonBaseClass} ${getManualGAButtonVariantClass(variant)}`}
      >
        {manualGAMode ? 'Cancel Manual' : 'Manual'}
      </button>
    );
  };

  return (
    <div className="card relative">
      {renderManualGAButton('inline')}

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
            : foveaConfirmed && manualGAMode
            ? 'crosshair'
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
          {imageAnalysis.gaRegions && foveaConfirmed && !gaConfirmed && (
            <>
              {manualGAMode ? (
                <p className="text-sm text-blue-600 font-semibold mt-2">
                  👆 Click on the image to set GA distance point
                </p>
              ) : imageAnalysis.selectedGARegionIndex === undefined && !imageAnalysis.isManualGAPoint ? (
                <p className="text-sm text-blue-600 font-semibold mt-2">
                  👆 Click a GA region to select, or use Manual for free-form point selection
                </p>
              ) : (
                <p className="text-sm text-gray-600 mt-2">
                  ✓ GA region selected
                </p>
              )}
              {isProcessingGA && (
                <p className="text-sm text-orange-600 font-semibold mt-2">
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

      {/* Expand Image button for fovea and GA precision work */}
      {canExpandImage && (
        <button
          onClick={() => setModalOpen(true)}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors"
        >
          {foveaConfirmed
            ? '🔍 Expand Image for Precise GA Selection'
            : '🔍 Expand Image for Precise Adjustment'}
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

            {renderManualGAButton('modal')}
            
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
                  : !foveaConfirmed && hoveredModalDiscZone
                  ? (hoveredModalDiscZone === 'body' ? 'move' : 'ns-resize')
                  : !foveaConfirmed && isHoveringModalFovea 
                  ? 'grab' 
                  : foveaConfirmed && manualGAMode
                  ? 'crosshair'
                  : foveaConfirmed
                  ? 'pointer'
                  : 'crosshair'
              }}
            />
            
            {/* Instructions */}
            <p className="text-white text-center mt-4 text-lg">
              {foveaConfirmed
                ? manualGAMode
                  ? 'Click to place a manual GA distance point'
                  : 'Click a GA region, or click any area to segment locally'
                : 'Drag fovea (green) or disc bracket handles (red) to adjust'}
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
