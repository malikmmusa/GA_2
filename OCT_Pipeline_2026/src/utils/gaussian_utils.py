"""
Utility functions for generating Gaussian heatmaps and handling coordinates.

Used in Stage 2 (Fovea) and Stage 3 (GA) for converting point annotations 
to 2D Gaussian distributions for training.
"""

import numpy as np
import torch


def generate_gaussian_heatmap(center_x, center_y, height, width, sigma=15):
    """
    Generate a 2D Gaussian heatmap centered at (center_x, center_y).
    
    Used for converting point annotations to regression targets.
    
    Args:
        center_x: X-coordinate of the center (in pixels)
        center_y: Y-coordinate of the center (in pixels)
        height: Height of the output heatmap
        width: Width of the output heatmap
        sigma: Standard deviation of the Gaussian (default: 15px for fovea)
        
    Returns:
        heatmap: 2D numpy array of shape (height, width) with values in [0, 1]
    """
    # Create coordinate grids
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    
    # Calculate Gaussian
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    
    # Normalize to [0, 1]
    if gaussian.max() > 0:
        gaussian = gaussian / gaussian.max()
    
    return gaussian


def generate_gaussian_mask(center_x, center_y, height, width, sigma=50, threshold=0.5):
    """
    Generate a binary mask from a Gaussian distribution.
    
    Used for Stage 3 (GA Segmentation) to create proxy masks from point labels.
    
    Args:
        center_x: X-coordinate of the center
        center_y: Y-coordinate of the center
        height: Height of the output mask
        width: Width of the output mask
        sigma: Standard deviation (default: 50px for GA, larger than fovea)
        threshold: Threshold for binarization (default: 0.5)
        
    Returns:
        mask: Binary mask (height, width) with values 0 or 1
    """
    gaussian = generate_gaussian_heatmap(center_x, center_y, height, width, sigma)
    mask = (gaussian > threshold).astype(np.float32)
    return mask


def heatmap_to_coordinates(heatmap):
    """
    Extract (x, y) coordinates from a predicted heatmap using argmax.
    
    Args:
        heatmap: 2D numpy array or torch tensor of shape (H, W)
        
    Returns:
        (x, y): Predicted coordinates
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Flatten and find argmax
    flat_idx = np.argmax(heatmap)
    height, width = heatmap.shape
    
    # Convert flat index to (y, x)
    y = flat_idx // width
    x = flat_idx % width
    
    return (int(x), int(y))


def refine_coordinates_weighted(heatmap, window_size=5):
    """
    Refine coordinate prediction using weighted average around the maximum.
    
    Provides sub-pixel accuracy by averaging nearby high-confidence pixels.
    
    Args:
        heatmap: 2D numpy array of shape (H, W)
        window_size: Size of the window around the maximum to consider
        
    Returns:
        (x, y): Refined coordinates (float)
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    height, width = heatmap.shape
    
    # Find the maximum point
    y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Define window bounds
    y_min = max(0, y_max - window_size // 2)
    y_max_bound = min(height, y_max + window_size // 2 + 1)
    x_min = max(0, x_max - window_size // 2)
    x_max_bound = min(width, x_max + window_size // 2 + 1)
    
    # Extract window
    window = heatmap[y_min:y_max_bound, x_min:x_max_bound]
    
    # Create coordinate grids for the window
    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max_bound),
        np.arange(x_min, x_max_bound),
        indexing='ij'
    )
    
    # Weighted average
    total_weight = np.sum(window)
    
    if total_weight > 0:
        x_refined = np.sum(x_coords * window) / total_weight
        y_refined = np.sum(y_coords * window) / total_weight
    else:
        x_refined = float(x_max)
        y_refined = float(y_max)
    
    return (float(x_refined), float(y_refined))


def apply_spatial_constraint(predicted_x, predicted_y, disc_x, disc_y, min_distance=50):
    """
    Apply anatomical constraint: Fovea is temporal (lateral) to the optic disc.
    
    If the predicted fovea is too close to the disc, adjust it.
    
    Args:
        predicted_x: Predicted fovea x-coordinate
        predicted_y: Predicted fovea y-coordinate
        disc_x: Known optic disc x-coordinate
        disc_y: Known optic disc y-coordinate
        min_distance: Minimum distance between fovea and disc (default: 50px)
        
    Returns:
        (x, y): Adjusted coordinates if necessary
    """
    # Calculate distance
    distance = np.sqrt((predicted_x - disc_x)**2 + (predicted_y - disc_y)**2)
    
    if distance < min_distance:
        # Push fovea away from disc to maintain minimum distance
        angle = np.arctan2(predicted_y - disc_y, predicted_x - disc_x)
        adjusted_x = disc_x + min_distance * np.cos(angle)
        adjusted_y = disc_y + min_distance * np.sin(angle)
        return (int(adjusted_x), int(adjusted_y))
    
    return (predicted_x, predicted_y)


def visualize_heatmap_overlay(image, heatmap, alpha=0.5):
    """
    Create a visualization of heatmap overlaid on the original image.
    
    Args:
        image: Original image (H, W, C) as numpy array
        heatmap: Heatmap (H, W) as numpy array
        alpha: Transparency of the overlay
        
    Returns:
        overlay: RGB image with heatmap overlay
    """
    import cv2
    
    # Normalize heatmap to [0, 255]
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap (e.g., jet)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap if necessary
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay
