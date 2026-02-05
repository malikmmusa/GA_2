import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- SECTION 2: ANATOMY-AWARE PIPELINE ---

# --- 1. TOOLS FOR ANCHORS ---
def find_pink_line_anchor(fundus_img):
    hsv = cv2.cvtColor(fundus_img, cv2.COLOR_RGB2HSV)
    lower_pink = np.array([145, 100, 100])
    upper_pink = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def process_green_line_precise(fundus_img):
    h, w, _ = fundus_img.shape
    hsv = cv2.cvtColor(fundus_img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    green_points = cv2.findNonZero(mask)
    if green_points is None: 
        return h // 2
    return int(np.mean(green_points[:, 0, 1]))

def get_ground_truth_fovea(fundus_img):
    hsv = cv2.cvtColor(fundus_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) + \
           cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0: return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    return None

# --- 2. ANATOMY-AWARE FOVEA FINDER ---
def find_fovea_dark_spot_refined(fundus_img, green_y):
    if green_y is None: return None
    gray = cv2.cvtColor(fundus_img, cv2.COLOR_RGB2GRAY)
    line_profile = gray[green_y, :]

    # Define Search Area (Center 30%)
    w = len(line_profile)
    start_search = int(w * 0.35)
    end_search = int(w * 0.65)
    center_profile = line_profile[start_search:end_search]

    # Smooth the signal
    smoothed = gaussian_filter1d(center_profile.astype(float), sigma=8)

    # FIX: Don't just take min(). Find peaks (inverted) to get valleys.
    # We invert the signal because find_peaks looks for hills, not valleys.
    inverted_profile = -smoothed
    peaks, _ = find_peaks(inverted_profile)

    if len(peaks) == 0:
        # Fallback to simple minimum if no peaks found
        return start_search + np.argmin(smoothed)

    # Pick the valley closest to the CENTER of the search area
    # The fovea is usually dead center in the image.
    search_center = (end_search - start_search) // 2
    best_peak_idx = peaks[np.argmin(np.abs(peaks - search_center))]
    return start_search + best_peak_idx

# --- 3. CENTRALITY-BIASED GA SEGMENTER ---
def segment_macular_ga_kmeans(fundus_img, disc_contour):
    gray = cv2.cvtColor(fundus_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    image_center = (w // 2, h // 2)

    # Contrast Enhancement (Stronger for dark images)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Mask out Disc
    if disc_contour is not None:
        cv2.drawContours(enhanced, [disc_contour], -1, 0, -1)

    # K-Means
    pixel_values = enhanced.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    lesion_cluster_index = np.argmax(centers)

    labels = labels.flatten()
    lesion_mask = (labels == lesion_cluster_index).astype(np.uint8) * 255
    lesion_mask = lesion_mask.reshape(gray.shape)

    # Cleanup - Step A: Morphology (Larger Kernel)
    kernel_size = 15 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    candidates = []

    # Initial filtering (Size, Circularity, Border)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 1. Size Filter
        if area < 500: continue 

        # 2. Circularity Filter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8: continue 

        # 3. Location Filter
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if x <= 2 or y <= 2 or (x + w_box) >= (w - 2) or (y + h_box) >= (h - 2):
            continue

        candidates.append(cnt)
    
    if not candidates:
        return []

    # Step B: Sort by Area
    candidates.sort(key=cv2.contourArea, reverse=True)

    # Step C: Big Fish Rule (Relative Size)
    largest_area = cv2.contourArea(candidates[0])
    threshold_area = 0.2 * largest_area
    
    final_contours = [c for c in candidates if cv2.contourArea(c) >= threshold_area]

    # Step D: Limit to max 3
    return final_contours[:3]

# --- 4. MAIN PROCESSOR ---
def process_file_final_pipeline(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width = img.shape[1]
    
    # Assuming the fundus image is the right half if it's a dual image (common in datasets)
    # If standard images are used, this might need adjustment.
    # The original notebook had: fundus_img = img_rgb[:, width//2:]
    # We will keep it but add a check or make it optional if needed.
    # For now, sticking to the notebook logic as requested.
    fundus_img = img_rgb[:, width//2:]

    disc_cnt = find_pink_line_anchor(fundus_img)
    green_y = process_green_line_precise(fundus_img)
    fovea_x = find_fovea_dark_spot_refined(fundus_img, green_y)
    ga_contours = segment_macular_ga_kmeans(fundus_img, disc_cnt)

    vis_img = fundus_img.copy()
    dist_text = "N/A"

    if ga_contours and fovea_x is not None and green_y is not None:
        fovea_pt = (fovea_x, green_y)
        
        min_dist_overall = float('inf')
        closest_pt_overall = None
        
        # Iterate through ALL valid lesions to find the global minimum distance
        for cnt in ga_contours:
            ga_points = cnt[:, 0, :]
            dists = np.sqrt(np.sum((ga_points - fovea_pt)**2, axis=1))
            min_dist_local = np.min(dists)
            
            if min_dist_local < min_dist_overall:
                min_dist_overall = min_dist_local
                closest_idx = np.argmin(dists)
                closest_pt_overall = tuple(ga_points[closest_idx])

        if closest_pt_overall is not None:
            cv2.line(vis_img, fovea_pt, closest_pt_overall, (255, 255, 0), 2)
            dist_text = f"{min_dist_overall:.1f} px"
            
        # Draw ALL contours
        cv2.drawContours(vis_img, ga_contours, -1, (255, 255, 0), 2)
        cv2.drawMarker(vis_img, fovea_pt, (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2, markerSize=25)

    red_dot = get_ground_truth_fovea(fundus_img)
    if red_dot:
        cv2.circle(vis_img, red_dot, 5, (255, 0, 0), -1)

    return vis_img, dist_text

def main():
    # Use paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data/raw")
    output_dir = os.path.join(script_dir, "analyzed_results_v2")
    
    if not os.path.exists(input_dir):
        print(f"Creating input directory: {input_dir}")
        os.makedirs(input_dir)
        print(f"Please place your images in the '{input_dir}' folder and run the script again.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not files:
        print(f"No images found in '{input_dir}'. Please add some images.")
        return

    print(f"Found {len(files)} images to process.")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")
        
        res_img, dist = process_file_final_pipeline(file_path)

        if res_img is not None:
            save_path = os.path.join(output_dir, f"analyzed_{filename}")
            # Convert back to BGR for saving with OpenCV
            cv2.imwrite(save_path, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
            print(f"  -> Saved to {save_path} (Distance: {dist})")
        else:
            print(f"  -> Failed to process {filename}")

if __name__ == "__main__":
    main()
