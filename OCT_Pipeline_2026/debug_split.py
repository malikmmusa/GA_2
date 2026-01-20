"""
Diagnostic Script: Analyze Divider Detection for OCT Image 22899013.png

This script performs detailed analysis of why the divider detection failed
and outputs comprehensive metrics to understand the root cause.
"""

import cv2
import numpy as np
from pathlib import Path

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("⚠️  matplotlib not available - skipping plots")
    HAS_MATPLOTLIB = False

# Image to analyze
IMAGE_PATH = "data/raw/22899013.png"

def analyze_image_columns(img, region='center'):
    """
    Analyze column-wise metrics across the image.

    Args:
        img: Input image (H, W, C) in BGR format
        region: 'center', 'left', 'right', or 'full'

    Returns:
        Dictionary of metrics for each column
    """
    height, width, channels = img.shape

    # Define analysis region
    if region == 'center':
        start_col = width // 2 - width // 10  # Center 20%
        end_col = width // 2 + width // 10
    elif region == 'left':
        start_col = width // 4
        end_col = width // 2
    elif region == 'right':
        start_col = width // 2
        end_col = 3 * width // 4
    else:  # full
        start_col = 0
        end_col = width

    metrics = {
        'column': [],
        'mean_intensity': [],
        'variance': [],
        'std_dev': [],
        'color_range': [],  # max(B,G,R) - min(B,G,R)
        'blue_dominance': [],  # B - max(G,R)
        'is_colored': [],  # Boolean: is this a colored column?
        'mean_B': [],
        'mean_G': [],
        'mean_R': [],
    }

    for col in range(start_col, end_col):
        col_pixels = img[:, col, :]

        # Basic metrics
        mean_intensity = np.mean(col_pixels)
        variance = np.var(col_pixels)
        std_dev = np.std(col_pixels)

        # Color analysis
        mean_color = np.mean(col_pixels, axis=0)
        B, G, R = mean_color[0], mean_color[1], mean_color[2]
        color_range = max(B, G, R) - min(B, G, R)
        blue_dominance = B - max(G, R)

        # Colored check (using current algorithm logic)
        is_colored = False
        if color_range > 20:
            if B > 50 and (G < 10 or R < 80):
                is_colored = True
        if color_range > 10 and not is_colored:
            if B > 30 and B > R * 1.2 and B > G * 1.2:
                is_colored = True
        if color_range > 5 and not is_colored:
            if B > R and B > G and (B - R > 5 or B - G > 5):
                is_colored = True

        # Store metrics
        metrics['column'].append(col)
        metrics['mean_intensity'].append(mean_intensity)
        metrics['variance'].append(variance)
        metrics['std_dev'].append(std_dev)
        metrics['color_range'].append(color_range)
        metrics['blue_dominance'].append(blue_dominance)
        metrics['is_colored'].append(is_colored)
        metrics['mean_B'].append(B)
        metrics['mean_G'].append(G)
        metrics['mean_R'].append(R)

    return metrics


def analyze_hsv_characteristics(img):
    """
    Analyze HSV color space characteristics of the image.

    Returns:
        HSV analysis for divider detection
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channels = img.shape

    # Analyze center region columns
    start_col = width // 2 - width // 10
    end_col = width // 2 + width // 10

    hsv_metrics = {
        'column': [],
        'mean_hue': [],
        'mean_saturation': [],
        'mean_value': [],
        'high_sat_percentage': [],  # % of pixels with saturation > 50
    }

    for col in range(start_col, end_col):
        col_pixels = hsv[:, col, :]

        mean_h = np.mean(col_pixels[:, 0])
        mean_s = np.mean(col_pixels[:, 1])
        mean_v = np.mean(col_pixels[:, 2])

        # Percentage of highly saturated pixels
        high_sat_count = np.sum(col_pixels[:, 1] > 50)
        high_sat_pct = (high_sat_count / height) * 100

        hsv_metrics['column'].append(col)
        hsv_metrics['mean_hue'].append(mean_h)
        hsv_metrics['mean_saturation'].append(mean_s)
        hsv_metrics['mean_value'].append(mean_v)
        hsv_metrics['high_sat_percentage'].append(high_sat_pct)

    return hsv_metrics


def calculate_gradient_profile(img):
    """
    Calculate horizontal gradient magnitude profile.

    Returns:
        Gradient metrics for each column
    """
    # Convert to grayscale for gradient calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_magnitude = np.abs(sobelx)

    # Sum gradient magnitudes vertically for each column
    height, width = gray.shape
    start_col = width // 2 - width // 10
    end_col = width // 2 + width // 10

    gradient_profile = {
        'column': [],
        'gradient_sum': [],
        'gradient_mean': [],
    }

    for col in range(start_col, end_col):
        col_gradient = gradient_magnitude[:, col]
        gradient_profile['column'].append(col)
        gradient_profile['gradient_sum'].append(np.sum(col_gradient))
        gradient_profile['gradient_mean'].append(np.mean(col_gradient))

    return gradient_profile


def plot_diagnostics(img, metrics, hsv_metrics, gradient_metrics):
    """
    Create comprehensive diagnostic plots.
    """
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Skipping plots - matplotlib not available")
        return

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'Diagnostic Analysis: {IMAGE_PATH}', fontsize=16, fontweight='bold')

    columns = metrics['column']

    # Plot 1: Original Image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Composite Image')
    axes[0, 0].axvline(x=img.shape[1]//2, color='red', linestyle='--', label='Center')
    axes[0, 0].legend()

    # Plot 2: Variance Profile
    axes[0, 1].plot(columns, metrics['variance'], 'b-', linewidth=2)
    axes[0, 1].axhline(y=200, color='orange', linestyle='--', label='Old threshold (200)')
    axes[0, 1].axhline(y=500, color='red', linestyle='--', label='New threshold (500)')
    axes[0, 1].set_title('Column Variance Profile')
    axes[0, 1].set_xlabel('Column Index')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean Intensity
    axes[1, 0].plot(columns, metrics['mean_intensity'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=20, color='orange', linestyle='--', label='Old threshold (20)')
    axes[1, 0].axhline(y=40, color='red', linestyle='--', label='New threshold (40)')
    axes[1, 0].set_title('Column Mean Intensity')
    axes[1, 0].set_xlabel('Column Index')
    axes[1, 0].set_ylabel('Mean Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Color Range (max - min of BGR)
    axes[1, 1].plot(columns, metrics['color_range'], 'purple', linewidth=2)
    axes[1, 1].axhline(y=20, color='orange', linestyle='--', label='High threshold (20)')
    axes[1, 1].axhline(y=10, color='yellow', linestyle='--', label='Med threshold (10)')
    axes[1, 1].axhline(y=5, color='green', linestyle='--', label='Low threshold (5)')
    axes[1, 1].set_title('Color Range (BGR channel spread)')
    axes[1, 1].set_xlabel('Column Index')
    axes[1, 1].set_ylabel('Color Range')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: BGR Channels
    axes[2, 0].plot(columns, metrics['mean_B'], 'b-', label='Blue', linewidth=2)
    axes[2, 0].plot(columns, metrics['mean_G'], 'g-', label='Green', linewidth=2)
    axes[2, 0].plot(columns, metrics['mean_R'], 'r-', label='Red', linewidth=2)
    axes[2, 0].set_title('BGR Channel Means')
    axes[2, 0].set_xlabel('Column Index')
    axes[2, 0].set_ylabel('Channel Value')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Blue Dominance
    axes[2, 1].plot(columns, metrics['blue_dominance'], 'b-', linewidth=2)
    axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2, 1].fill_between(columns, 0, metrics['blue_dominance'],
                            where=np.array(metrics['blue_dominance']) > 0,
                            alpha=0.3, color='blue', label='Blue > R,G')
    axes[2, 1].set_title('Blue Dominance (B - max(G,R))')
    axes[2, 1].set_xlabel('Column Index')
    axes[2, 1].set_ylabel('Blue Dominance')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 7: HSV Saturation
    axes[3, 0].plot(hsv_metrics['column'], hsv_metrics['mean_saturation'],
                    'orange', linewidth=2, label='Mean Saturation')
    axes[3, 0].plot(hsv_metrics['column'], hsv_metrics['high_sat_percentage'],
                    'red', linewidth=2, label='High Sat % (>50)')
    axes[3, 0].set_title('HSV Saturation Profile')
    axes[3, 0].set_xlabel('Column Index')
    axes[3, 0].set_ylabel('Saturation / Percentage')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)

    # Plot 8: Gradient Profile
    axes[3, 1].plot(gradient_metrics['column'], gradient_metrics['gradient_mean'],
                    'darkgreen', linewidth=2)
    axes[3, 1].set_title('Horizontal Gradient Profile')
    axes[3, 1].set_xlabel('Column Index')
    axes[3, 1].set_ylabel('Mean Gradient Magnitude')
    axes[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagnostics_22899013.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to: diagnostics_22899013.png")


def print_summary_statistics(metrics, hsv_metrics, img):
    """
    Print detailed summary statistics.
    """
    height, width, _ = img.shape

    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY: 22899013.png")
    print("="*70)

    print(f"\nImage Dimensions: {width} x {height} pixels")
    print(f"Center Column: {width // 2}")
    print(f"Analysis Region: columns {min(metrics['column'])} to {max(metrics['column'])}")

    print("\n--- VARIANCE ANALYSIS ---")
    print(f"Min Variance: {min(metrics['variance']):.2f}")
    print(f"Max Variance: {max(metrics['variance']):.2f}")
    print(f"Mean Variance: {np.mean(metrics['variance']):.2f}")
    print(f"Columns with variance < 200: {sum(v < 200 for v in metrics['variance'])}")
    print(f"Columns with variance < 500: {sum(v < 500 for v in metrics['variance'])}")

    print("\n--- INTENSITY ANALYSIS ---")
    print(f"Min Intensity: {min(metrics['mean_intensity']):.2f}")
    print(f"Max Intensity: {max(metrics['mean_intensity']):.2f}")
    print(f"Mean Intensity: {np.mean(metrics['mean_intensity']):.2f}")

    print("\n--- COLOR ANALYSIS ---")
    print(f"Max Color Range: {max(metrics['color_range']):.2f}")
    print(f"Mean Color Range: {np.mean(metrics['color_range']):.2f}")
    print(f"Columns identified as colored: {sum(metrics['is_colored'])}")

    print("\n--- BLUE CHANNEL ANALYSIS ---")
    print(f"Max Blue Dominance: {max(metrics['blue_dominance']):.2f}")
    print(f"Columns with Blue > R,G: {sum(b > 0 for b in metrics['blue_dominance'])}")

    print("\n--- HSV ANALYSIS ---")
    print(f"Max Mean Saturation: {max(hsv_metrics['mean_saturation']):.2f}")
    print(f"Mean Mean Saturation: {np.mean(hsv_metrics['mean_saturation']):.2f}")
    print(f"Max High-Sat %: {max(hsv_metrics['high_sat_percentage']):.2f}%")

    # Find the divider columns
    colored_cols = [c for c, is_col in zip(metrics['column'], metrics['is_colored']) if is_col]
    if colored_cols:
        print(f"\n--- DETECTED DIVIDER REGION ---")
        print(f"Start Column: {min(colored_cols)}")
        print(f"End Column: {max(colored_cols)}")
        print(f"Width: {max(colored_cols) - min(colored_cols) + 1} pixels")

        # Sample a few columns
        print("\nSample Column Metrics from Divider:")
        sample_col = colored_cols[len(colored_cols)//2]
        idx = metrics['column'].index(sample_col)
        print(f"  Column {sample_col}:")
        print(f"    BGR: ({metrics['mean_B'][idx]:.1f}, {metrics['mean_G'][idx]:.1f}, {metrics['mean_R'][idx]:.1f})")
        print(f"    Variance: {metrics['variance'][idx]:.2f}")
        print(f"    Color Range: {metrics['color_range'][idx]:.2f}")
        print(f"    Blue Dominance: {metrics['blue_dominance'][idx]:.2f}")
    else:
        print("\n⚠️  WARNING: No colored columns detected in analysis region!")

    print("\n" + "="*70)


def main():
    """
    Run comprehensive diagnostic analysis.
    """
    print(f"\nLoading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print(f"❌ Error: Could not load image: {IMAGE_PATH}")
        return

    print("Running diagnostic analysis...")

    # Analyze metrics
    metrics = analyze_image_columns(img, region='center')
    hsv_metrics = analyze_hsv_characteristics(img)
    gradient_metrics = calculate_gradient_profile(img)

    # Print statistics
    print_summary_statistics(metrics, hsv_metrics, img)

    # Generate plots
    plot_diagnostics(img, metrics, hsv_metrics, gradient_metrics)

    print("\n✓ Diagnostic analysis complete!")


if __name__ == "__main__":
    main()
