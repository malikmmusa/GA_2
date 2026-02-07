"""Optic Disc Detection Service - Refactored from run_inference.py"""
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple, Dict

# Import legacy modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.retfound_unet import RETFound_UNet
from utils.image_utils import get_split_indices_and_images


class DiscDetectorService:
    """
    Service for detecting optic disc using RETFound U-Net model.
    Preserves the "New Algorithm" (Contour/Energy method) from run_inference.py.
    """
    
    def __init__(
        self,
        model_path: str = "weights/best_disc_model.pth",
        img_size: int = 224,
        device: str = None
    ):
        """
        Initialize the disc detector service.
        
        Args:
            model_path: Path to trained model weights
            img_size: Input size for model (224x224)
            device: Device to run inference on (auto-detect if None)
        """
        self.img_size = img_size
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Define preprocessing transform
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def _load_model(self) -> RETFound_UNet:
        """Load the RETFound U-Net model with trained weights."""
        print(f"[DiscDetector] Loading model from {self.model_path}...")
        model = RETFound_UNet(
            img_size=self.img_size,
            weights_path=None,
            freeze_encoder=False
        )
        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"[DiscDetector] Model loaded on {self.device}")
        return model
    
    def detect_from_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect optic disc from a full composite OCT image.
        
        Args:
            image: BGR image (numpy array) - full composite with B-scan and En-face
        
        Returns:
            Dictionary containing:
                - disc_center_x: X coordinate in original image
                - disc_center_y: Y coordinate in original image
                - disc_top_y: Top Y coordinate of vertical line
                - disc_bottom_y: Bottom Y coordinate of vertical line
                - disc_height_pixels: Height of disc in pixels (1800 microns)
                - pixel_to_micron_ratio: Conversion factor (1800 / height)
                - en_face_split_x: X coordinate where en-face starts in original
        """
        # Step 1: Split the composite image
        _, en_face, metadata = get_split_indices_and_images(
            image,
            divider_safety_margin=10
        )
        
        # Convert to RGB
        en_face_rgb = cv2.cvtColor(en_face, cv2.COLOR_BGR2RGB)
        h_ef, w_ef = en_face_rgb.shape[:2]
        
        # Step 2: Preprocess
        augmented = self.transform(image=en_face_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        
        # Step 3: Inference
        with torch.no_grad():
            output = self.model(input_tensor)  # [1, 1, 224, 224]
            heatmap = output.cpu().squeeze().numpy()  # [224, 224]
        
        # Step 4: Extract Coordinates using "New Algorithm"
        coords = self._extract_disc_coordinates_new_algorithm(
            heatmap=heatmap,
            en_face_width=w_ef,
            en_face_height=h_ef,
            en_face_split_x=metadata['final_split_column']
        )
        
        return coords
    
    def _extract_disc_coordinates_new_algorithm(
        self,
        heatmap: np.ndarray,
        en_face_width: int,
        en_face_height: int,
        en_face_split_x: int
    ) -> Dict[str, float]:
        """
        Extract disc coordinates using an improved robust algorithm.
        
        UPDATED (2026-02-07): Fixed systematic over-segmentation by:
        1. Replacing Otsu with 95th percentile threshold (tighter cutoff)
        2. Using weighted centroid for X coordinate (more accurate than bbox center)
        3. Extracting actual pixel min/max Y (not bounding box extents)
        
        OLD ISSUES (confirmed via ground-truth validation on 52 images):
        - Otsu threshold was too permissive, including background gradient
        - Bounding box Y-extent was systematically ~134px taller than ground-truth
        - Mean height error: 175%, median: 35%, with 79% of images > 20% error
        
        NEW ALGORITHM:
        1. Apply 95th percentile threshold on heatmap (tighter than Otsu)
        2. Use connected component analysis to find largest cluster
        3. Extract weighted centroid for X, actual pixel min/max for Y
        4. Fallback: if no components, use argmax with percentile threshold
        
        Args:
            heatmap: Raw model output heatmap [224, 224]
            en_face_width: Width of en-face region in original image
            en_face_height: Height of en-face region in original image
            en_face_split_x: X offset where en-face starts in original image
        
        Returns:
            Dictionary with disc coordinates and metadata
        """
        # -------------------------------------------------------------------------
        # Improved Algorithm: Percentile-Based Threshold + Weighted Centroid
        # -------------------------------------------------------------------------
        
        hm_h, hm_w = heatmap.shape
        raw_max = heatmap.max()
        raw_min = heatmap.min()
        
        # Method 1: 98th percentile threshold (much tighter - only keep brightest ~2% of pixels)
        # The 95th percentile was still too permissive, leading to over-segmentation
        threshold_percentile = 98.0
        threshold_value = np.percentile(heatmap, threshold_percentile)
        
        # Create binary mask
        binary_mask = (heatmap > threshold_value).astype(np.uint8)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, 
            connectivity=8
        )
        
        if num_labels <= 1:
            # No components found (only background) - fallback to argmax with lower percentile
            threshold_value_fallback = np.percentile(heatmap, 90.0)
            binary_mask_fallback = (heatmap > threshold_value_fallback).astype(np.uint8)
            
            # If still no pixels, use argmax
            if np.sum(binary_mask_fallback) == 0:
                py_raw, px_raw = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                min_y_raw = py_raw
                max_y_raw = py_raw
                cx_raw = px_raw
                print("  [DiscDetector] WARNING: No components found, using argmax fallback")
            else:
                # Extract coordinates from fallback mask
                y_indices, x_indices = np.where(binary_mask_fallback > 0)
                min_y_raw = np.min(y_indices)
                max_y_raw = np.max(y_indices)
                
                # Weighted centroid for X
                weights = heatmap[binary_mask_fallback > 0]
                cx_raw = np.average(x_indices, weights=weights)
                
                print(f"  [DiscDetector] Using 90th percentile fallback: {len(y_indices)} pixels")
        else:
            # Find largest component (excluding background label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            largest_component_idx = np.argmax(areas) + 1  # +1 to account for skipping background
            
            # Extract mask for largest component
            component_mask = (labels == largest_component_idx).astype(np.uint8)
            
            # Get actual pixel coordinates (not bounding box)
            y_indices, x_indices = np.where(component_mask > 0)
            
            # Y-extent: actual min/max of pixels in component
            min_y_raw = np.min(y_indices)
            max_y_raw = np.max(y_indices)
            
            # X coordinate: weighted centroid (more accurate than bbox center)
            weights = heatmap[component_mask > 0]
            cx_raw = np.average(x_indices, weights=weights)
            
            # Debug logging
            num_pixels = len(y_indices)
            bbox_h = stats[largest_component_idx, cv2.CC_STAT_HEIGHT]
            print(f"  [DiscDetector] Largest component: {num_pixels} pixels, "
                  f"Y-extent={max_y_raw - min_y_raw} px (bbox was {bbox_h} px)")
        
        # 3. Project to En Face Dimensions with Sub-Pixel Correction
        # Add 0.5 to center the coordinate within the raw pixel grid
        scale_x = en_face_width / self.img_size
        scale_y = en_face_height / self.img_size
        
        pred_x_ef = (cx_raw + 0.5) * scale_x
        pred_min_y_ef = (min_y_raw + 0.5) * scale_y
        pred_max_y_ef = (max_y_raw + 0.5) * scale_y
        
        # Also calculate the single point for logging
        pred_y_ef = (pred_min_y_ef + pred_max_y_ef) / 2
        
        # 4. Project to Original Composite Size
        start_x_offset = en_face_split_x
        
        orig_cx = start_x_offset + pred_x_ef
        orig_min_y = pred_min_y_ef
        orig_max_y = pred_max_y_ef
        
        # For logging purposes
        pred_x_orig = orig_cx
        pred_y_orig = (orig_min_y + orig_max_y) / 2
        
        # Calculate disc height and pixel-to-micron ratio
        disc_height_pixels = orig_max_y - orig_min_y
        pixel_to_micron_ratio = 1800.0 / disc_height_pixels  # 1800 microns is standard disc diameter
        
        print(f"[DiscDetector] Disc detected at ({pred_x_orig:.1f}, {pred_y_orig:.1f})")
        print(f"[DiscDetector] Disc height: {disc_height_pixels:.1f} pixels = 1800 microns")
        print(f"[DiscDetector] Pixel-to-micron ratio: {pixel_to_micron_ratio:.3f}")
        
        return {
            'disc_center_x': float(orig_cx),
            'disc_center_y': float(pred_y_orig),
            'disc_top_y': float(orig_min_y),
            'disc_bottom_y': float(orig_max_y),
            'disc_height_pixels': float(disc_height_pixels),
            'pixel_to_micron_ratio': float(pixel_to_micron_ratio),
            'en_face_split_x': int(en_face_split_x)
        }
    
    def detect_from_path(self, image_path: str) -> Dict[str, float]:
        """
        Detect optic disc from an image file path.
        
        Args:
            image_path: Path to OCT image file
        
        Returns:
            Dictionary with disc detection results
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.detect_from_image(img)
