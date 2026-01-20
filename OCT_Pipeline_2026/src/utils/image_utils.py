import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

def is_colored_divider_hsv(col_pixels):
    """
    Detect colored dividers using HSV color space analysis.
    """
    col_pixels_uint8 = col_pixels.astype(np.uint8)
    col_hsv = cv2.cvtColor(col_pixels_uint8.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    mean_h = np.mean(col_hsv[:, 0])
    mean_s = np.mean(col_hsv[:, 1])
    mean_v = np.mean(col_hsv[:, 2])

    high_sat_count = np.sum(col_hsv[:, 1] > 50)
    high_sat_percentage = (high_sat_count / len(col_hsv)) * 100

    if high_sat_percentage > 60 and mean_s > 40:
        if (80 <= mean_h <= 150) or (40 <= mean_h <= 80):
            return True

    if high_sat_percentage > 30 and mean_s > 25:
        if 85 <= mean_h <= 145:
            return True

    if mean_v < 80 and mean_s > 30 and high_sat_percentage > 20:
        if 70 <= mean_h <= 150:
            return True

    return False

def is_image_content(col_pixels):
    """
    Determine if a column contains actual image content.
    """
    mean_intensity = np.mean(col_pixels)
    if mean_intensity < 5:
        return False

    if is_colored_divider_hsv(col_pixels):
        return False

    mean_color = np.mean(col_pixels, axis=0)
    B, G, R = mean_color[0], mean_color[1], mean_color[2]

    if B > 50 and G < 10 and B > R * 1.5: return False
    if B > 50 and R < 80 and B > R * 1.3:
        if max(B,G,R) - min(B,G,R) > 20: return False
    if R < 80 and (B > 50 or G > 50):
        if np.std([B,G,R]) > 15: return False

    variance = np.var(col_pixels)
    if variance < 30:
        if mean_intensity < 15: return False

    color_std = np.std(mean_color)
    if color_std < 15:
        if mean_intensity > 40 and variance > 500: return True
        if mean_intensity > 30: return True
        return False

    if variance > 50:
        if G < 5 and B > 30: return False
        if mean_intensity > 25: return True

    return False

def detect_dividing_line(img, search_width=300):
    """
    Detect a vertical dividing line. Returns (divider_start, divider_end).
    """
    height, width, channels = img.shape
    center = width // 2
    search_start = max(0, center - search_width)
    search_end = min(width, center + search_width)

    bscan_content_end = None
    for col in range(center, search_start - 1, -1):
        if is_image_content(img[:, col, :]):
            bscan_content_end = col
            break

    enface_content_start = None
    for col in range(center, search_end):
        if is_image_content(img[:, col, :]):
            enface_content_start = col
            break

    if bscan_content_end is None or enface_content_start is None:
        return None, None

    divider_start = bscan_content_end + 1
    divider_end = enface_content_start
    
    # Refining the end of the divider
    content_start = divider_end
    true_divider_end = divider_end
    for col in range(content_start - 1, divider_start - 1, -1):
        col_pixels = img[:, col, :]
        mean_color = np.mean(col_pixels, axis=0)
        B, G, R = mean_color
        mean_intensity = np.mean(mean_color)
        if not (B < 20 and G < 20 and R < 20 and mean_intensity < 15):
            true_divider_end = col + 1
        else:
            break
    
    divider_end = true_divider_end
    
    divider_width = divider_end - divider_start
    if divider_width < 1 or divider_width > 400:
        return None, None

    return divider_start, divider_end

def detect_image_boundary(img, search_window=200):
    """
    Detect boundary based on content change (fallback).
    """
    height, width, channels = img.shape
    center = width // 2
    search_start = max(20, center - search_window)
    search_end = min(width - 20, center + search_window)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_magnitude = np.abs(sobelx)

    gradient_profile = []
    for col in range(search_start, search_end):
        col_gradient_sum = np.sum(gradient_magnitude[:, col])
        gradient_profile.append(col_gradient_sum)

    if len(gradient_profile) > 0:
        gradient_profile = np.array(gradient_profile)
        smoothed_gradient = gaussian_filter1d(gradient_profile, sigma=5)
        inverted_gradient = -smoothed_gradient
        valley_idx = np.argmax(inverted_gradient)
        valley_col = search_start + valley_idx
        valley_value = smoothed_gradient[valley_idx]
        mean_gradient = np.mean(smoothed_gradient)

        if valley_value < 0.4 * mean_gradient and mean_gradient > 100:
            return valley_col

    col_differences = []
    for col in range(search_start, search_end - 1):
        col_current = img[:, col, :].astype(np.float32)
        col_next = img[:, col + 1, :].astype(np.float32)
        diff = np.mean(np.abs(col_current - col_next))
        col_differences.append(diff)

    if len(col_differences) == 0:
        return None

    smoothed_diffs = gaussian_filter1d(col_differences, sigma=3)
    max_change_idx = np.argmax(smoothed_diffs)
    boundary_col = search_start + max_change_idx
    max_change = smoothed_diffs[max_change_idx]
    mean_change = np.mean(smoothed_diffs)

    if max_change > 3 * mean_change:
        return boundary_col

    return None

def trim_colored_edges(img_section):
    """
    Remove colored divider pixels from edges.
    """
    height, width, channels = img_section.shape
    left_trim = 0
    for col in range(width):
        col_pixels = img_section[:, col, :]
        if is_colored_divider_hsv(col_pixels):
            left_trim = col + 1
        else:
            # Fallback BGR check logic could go here, but keeping it simple for now
            # as HSV is robust.
            break

    right_trim = 0
    for col in range(width - 1, -1, -1):
        col_pixels = img_section[:, col, :]
        if is_colored_divider_hsv(col_pixels):
            right_trim = width - col
        else:
            break

    if left_trim > 0 or right_trim > 0:
        end_col = width - right_trim if right_trim > 0 else width
        trimmed = img_section[:, left_trim:end_col, :]
        return trimmed, left_trim, right_trim

    return img_section, 0, 0

def adaptive_edge_scrub(en_face, scan_width=300):
    """
    Scan left edge of en_face for artifacts.
    """
    height, width, _ = en_face.shape
    scan_limit = min(scan_width, width)
    left_region = np.ascontiguousarray(en_face[:, :scan_limit, :])
    hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
    last_colored_col = -1
    min_artifact_pixels = int(height * 0.10)

    for col in range(scan_limit):
        color_mask = (hsv[:, col, 1] > 20) & (hsv[:, col, 0] >= 70) & (hsv[:, col, 0] <= 160)
        if np.sum(color_mask) > min_artifact_pixels:
            last_colored_col = col

    scrub_offset = 0
    if last_colored_col >= 0:
        scrub_offset = last_colored_col + 5
        scrub_offset = min(scrub_offset, width - 1)
        en_face = en_face[:, scrub_offset:, :]

    return en_face, scrub_offset

def adaptive_edge_scrub_right(b_scan, scan_width=300):
    """
    Scan right edge of b_scan for artifacts.
    """
    height, width, _ = b_scan.shape
    scan_limit = min(scan_width, width)
    right_region = np.ascontiguousarray(b_scan[:, -scan_limit:, :])
    hsv = cv2.cvtColor(right_region, cv2.COLOR_BGR2HSV)
    first_colored_col_local = -1
    min_artifact_pixels = int(height * 0.10)
    
    for col in range(scan_limit):
        color_mask = (hsv[:, col, 1] > 20) & (hsv[:, col, 0] >= 70) & (hsv[:, col, 0] <= 160)
        if np.sum(color_mask) > min_artifact_pixels:
            first_colored_col_local = col
            break
            
    scrub_offset = 0
    if first_colored_col_local >= 0:
        cut_point = (width - scan_limit) + first_colored_col_local - 5
        cut_point = max(0, cut_point)
        b_scan = b_scan[:, :cut_point, :]
        scrub_offset = width - cut_point
        
    return b_scan, scrub_offset

def get_split_indices_and_images(img, divider_safety_margin=0):
    """
    Main entry point for splitting. Returns (b_scan, en_face, metadata).
    Metadata contains offsets needed to map coordinates back to original image.
    
    Args:
        img: BGR image
        divider_safety_margin: Pixels to skip around divider. Positive = shrink content.
                               Negative = include divider (AVOID THIS).
    """
    height, width, _ = img.shape
    
    # 1. Detect Divider
    divider_start, divider_end = detect_dividing_line(img)
    detection_method = 'unknown'
    actual_split_column = 0 # This is where 'en_face' technically begins before trimming
    
    if divider_start is not None:
        detection_method = 'divider'
        # With margin=0, we start exactly where content implies.
        # safe_divider_start is the END of B-scan (minus margin)
        # safe_divider_end is the START of En-Face (plus margin)
        safe_divider_start = max(0, divider_start - divider_safety_margin)
        safe_divider_end = min(width, divider_end + divider_safety_margin)
        
        b_scan_raw = img[:, :safe_divider_start, :]
        en_face_raw = img[:, safe_divider_end:, :]
        actual_split_column = safe_divider_end
        
    else:
        # 2. Boundary
        boundary_col = detect_image_boundary(img)
        if boundary_col is not None:
            detection_method = 'boundary'
            safe_boundary_start = max(0, boundary_col - divider_safety_margin)
            safe_boundary_end = min(width, boundary_col + divider_safety_margin)
            
            b_scan_raw = img[:, :safe_boundary_start, :]
            en_face_raw = img[:, safe_boundary_end:, :]
            actual_split_column = safe_boundary_end
        else:
            # 3. Fallback
            detection_method = 'fallback'
            mid_point = width // 2
            safe_split_start = max(0, mid_point - divider_safety_margin)
            safe_split_end = min(width, mid_point + divider_safety_margin)
            
            b_scan_raw = img[:, :safe_split_start, :]
            en_face_raw = img[:, safe_split_end:, :]
            actual_split_column = safe_split_end

    # 4. Cleanup (Trim & Scrub)
    b_scan, b_left_trim, b_right_trim = trim_colored_edges(b_scan_raw)
    en_face, e_left_trim, e_right_trim = trim_colored_edges(en_face_raw)
    
    en_face, scrub_offset = adaptive_edge_scrub(en_face)
    b_scan, b_scrub_offset = adaptive_edge_scrub_right(b_scan)

    # 5. Metadata for coordinate mapping
    # To map a coordinate X_rel in en_face back to X_abs in img:
    # X_abs = actual_split_column + e_left_trim + scrub_offset + X_rel
    
    final_split_column = actual_split_column + e_left_trim + scrub_offset
    
    metadata = {
        'detection_method': detection_method,
        'actual_split_column': actual_split_column, # The rough cut point
        'trim_left_offset': e_left_trim,            # Pixels removed by trim_colored_edges
        'scrub_offset': scrub_offset,               # Pixels removed by adaptive_scrub
        'final_split_column': final_split_column,   # The net offset
        'b_scan_width': b_scan.shape[1],
        'en_face_width': en_face.shape[1]
    }
    
    return b_scan, en_face, metadata
