import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def process_green_line_precise(fundus_img):
    """
    Detects the green scan line to determine the Y-coordinate.
    """
    h, w = fundus_img.shape[:2]
    hsv = cv2.cvtColor(fundus_img, cv2.COLOR_BGR2HSV) # Note: Input is BGR
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    green_points = cv2.findNonZero(mask)
    if green_points is None: 
        return None
    return int(np.mean(green_points[:, 0, 1]))

def find_fovea_dark_spot_refined(fundus_img, center_y, search_center_x=None, band_height=1):
    """
    Finds the fovea X-coordinate by analyzing intensity profiles.
    
    Args:
        fundus_img: The En-Face image (BGR or Gray).
        center_y: The Y-coordinate to search around.
        search_center_x: Optional X-coordinate hint to select the best peak. 
                         If None, uses the center of the search window.
        band_height: Height of the strip to analyze. 
                     Use 1 for precise lines (Green Line).
                     Use ~60 for estimated regions (Geometric Fallback).
    """
    if center_y is None: return None
    
    h, w = fundus_img.shape[:2]
    
    # Define vertical bounds for the band
    y_start = max(0, int(center_y - band_height // 2))
    y_end = min(h, int(center_y + band_height // 2 + 1))
    
    if y_end <= y_start: return None
    
    # Pre-process: Convert to Gray
    if len(fundus_img.shape) == 3:
        roi_gray = cv2.cvtColor(fundus_img[y_start:y_end, :], cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = fundus_img[y_start:y_end, :]
        
    # Collapse to 1D Profile
    # If band_height > 1, take the MEAN intensity to reduce noise/vessels
    if band_height > 1:
        line_profile = np.mean(roi_gray, axis=0)
    else:
        line_profile = roi_gray[0, :]

    # Define Search Area (Center 30% or surrounding the estimate)
    # If we have a specific X estimate, looking in a window around it is safer
    if search_center_x is not None:
        window_radius = int(w * 0.25) # Search +/- 25% of width around estimate
        start_search = max(0, int(search_center_x - window_radius))
        end_search = min(w, int(search_center_x + window_radius))
        ideal_peak_x = search_center_x - start_search # Relative to crop
    else:
        start_search = int(w * 0.35)
        end_search = int(w * 0.65)
        ideal_peak_x = (end_search - start_search) // 2

    if start_search >= end_search:
        start_search = 0
        end_search = w
        ideal_peak_x = w // 2

    center_profile = line_profile[start_search:end_search]

    # Smooth the signal
    smoothed = gaussian_filter1d(center_profile.astype(float), sigma=8)

    # Invert for peak finding (looking for valleys)
    inverted_profile = -smoothed
    peaks, _ = find_peaks(inverted_profile)

    if len(peaks) == 0:
        # Fallback to simple minimum
        return start_search + np.argmin(smoothed)

    # Pick the valley closest to the ideal location (Estimate or Center)
    best_peak_idx = peaks[np.argmin(np.abs(peaks - ideal_peak_x))]
    return start_search + best_peak_idx

def find_fovea_anatomy_aware(full_image, en_face_roi_x, est_x=None, est_y=None):
    """
    Attempts to find the fovea using anatomy-aware rules.
    1. Green Line Anchor (Precise Y) -> Scan Line X
    2. Geometric Fallback (Approx Y) -> Scan Band X -> Refine Y
    """
    # Extract En-Face region
    en_face_img = full_image[:, en_face_roi_x:, :]
    h, w = en_face_img.shape[:2]
    
    # Adjust est_x to be local to en-face image
    local_est_x = (est_x - en_face_roi_x) if est_x is not None else None
    
    green_y = process_green_line_precise(en_face_img)
    
    if green_y is not None:
        # STRATEGY A: Green Line Found (Precise)
        search_y = green_y
        # Use a tight band (or just the line)
        fovea_x_local = find_fovea_dark_spot_refined(en_face_img, search_y, local_est_x, band_height=1)
        final_y = green_y
        method = "Anatomy (Green Line)"
    elif est_y is not None:
        # STRATEGY B: Green Line Missing (Geometric Fallback)
        search_y = est_y
        # Use a WIDER band to account for Y-estimation error
        fovea_x_local = find_fovea_dark_spot_refined(en_face_img, search_y, local_est_x, band_height=60)
        
        if fovea_x_local is None: return None
        
        # Refine Y: Find darkest point in a vertical column at the found X
        # Look +/- 40 pixels around the estimated Y
        search_h_rad = 40
        y_search_start = max(0, int(est_y - search_h_rad))
        y_search_end = min(h, int(est_y + search_h_rad))
        
        if len(en_face_img.shape) == 3:
            col_gray = cv2.cvtColor(en_face_img[y_search_start:y_search_end, fovea_x_local:fovea_x_local+1], cv2.COLOR_BGR2GRAY)
        else:
            col_gray = en_face_img[y_search_start:y_search_end, fovea_x_local:fovea_x_local+1]
            
        smoothed_col = gaussian_filter1d(col_gray.flatten().astype(float), sigma=5)
        local_min_y = np.argmin(smoothed_col)
        final_y = y_search_start + local_min_y
        method = "Anatomy (Geometric Fallback)"
    else:
        return None

    if fovea_x_local is None:
        return None
        
    # Map back to global coordinates
    global_x = en_face_roi_x + fovea_x_local
    global_y = final_y
    
    return (global_x, global_y), method

def manual_fovea_adjustment(full_image, initial_coords, window_name="Fovea Adjustment"):
    """
    Open an interactive window to manually adjust the fovea location.
    Left click to place/move the point. Press 'Enter' to confirm, 'Esc' to cancel.
    """
    current_pos = list(initial_coords)
    confirmed = [False]
    
    # Resize for display if the image is too large for the screen
    h, w = full_image.shape[:2]
    max_disp_h = 1000
    max_disp_w = 1600
    scale = min(max_disp_h / h, max_disp_w / w, 1.0)
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pos[0] = int(x / scale)
            current_pos[1] = int(y / scale)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"\n[INTERACTIVE] Adjust Fovea for: {window_name}")
    print(" - LEFT CLICK to move the point.")
    print(" - Press ENTER to confirm.")
    print(" - Press ESC to cancel.")

    while True:
        # Create a display copy
        display_img = full_image.copy()
        
        # Draw the point (Green circle)
        cv2.circle(display_img, (int(current_pos[0]), int(current_pos[1])), 10, (0, 255, 0), -1)
        cv2.circle(display_img, (int(current_pos[0]), int(current_pos[1])), 12, (255, 255, 255), 2)
        
        # Add instructions on the image
        cv2.putText(display_img, "Click to move fovea. ENTER to confirm, ESC to cancel.", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Scale for display
        if scale < 1.0:
            display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 13: # Enter
            confirmed[0] = True
            break
        elif key == 27: # Esc
            break

    cv2.destroyWindow(window_name)
    return tuple(current_pos) if confirmed[0] else None

def find_fovea_robust(full_image, red_line_coords, en_face_roi_x, image_name="Image"):
    """
    Locate the fovea using anatomy-aware logic first, falling back to geometric estimation.
    Includes an interactive visual verification.
    """
    # Always compute Geometric Estimate first (needed for fallback)
    (x1, y1), (x2, y2) = red_line_coords
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    disc_x = (x1 + x2) / 2
    disc_y = (y1 + y2) / 2
    
    height, width = full_image.shape[:2]
    en_face_width = width - en_face_roi_x
    en_face_center_x = en_face_roi_x + (en_face_width / 2)
    
    if disc_x > en_face_center_x:
        eye_side = "OS"
        est_x = disc_x + (2.5 * L)
    else:
        eye_side = "OD"
        est_x = disc_x - (2.5 * L)
        
    est_y = disc_y + (0.15 * L)
    
    # Strategy 1 & 2: Anatomy-Aware (with Geometric Fallback built-in)
    anatomy_result = find_fovea_anatomy_aware(full_image, en_face_roi_x, est_x=est_x, est_y=est_y)
    
    if anatomy_result:
        initial_guess, method_name = anatomy_result
        print(f"  [Auto] {method_name} fovea found at {initial_guess}")
        method_label = method_name
    else:
        # Absolute Fallback (Should rarely happen)
        print("  [Auto] All anatomy methods failed. Using raw geometry.")
        initial_guess = (int(est_x), int(est_y))
        method_label = f"Raw Geometry ({eye_side})"
    
    # Step D: Interactive Verification & Adjustment
    return manual_fovea_adjustment(full_image, initial_guess, f"Fovea: {image_name} [{method_label}]")

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
