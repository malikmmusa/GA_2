import cv2
import numpy as np

def analyze_result():
    img_path = "OCT_Pipeline_2026/data/inference_results/1954640_2_result.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return

    # Find Red pixels (B=0, G=0, R=255)
    # Note: OpenCV is BGR. So (0, 0, 255)
    # Allow some tolerance for anti-aliasing
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])
    
    mask = cv2.inRange(img, lower_red, upper_red)
    points = cv2.findNonZero(mask)
    
    if points is None:
        print("No red line found")
        return
        
    x_coords = points[:, 0, 0]
    y_coords = points[:, 0, 1]
    
    print(f"Red Line X range: {np.min(x_coords)} - {np.max(x_coords)}")
    print(f"Red Line Y range: {np.min(y_coords)} - {np.max(y_coords)}")
    
    # Check for Blue center point
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    mask_b = cv2.inRange(img, lower_blue, upper_blue)
    points_b = cv2.findNonZero(mask_b)
    
    if points_b is not None:
         xb = points_b[:, 0, 0]
         yb = points_b[:, 0, 1]
         print(f"Blue Center X range: {np.min(xb)} - {np.max(xb)}")
         print(f"Blue Center Y range: {np.min(yb)} - {np.max(yb)}")

if __name__ == "__main__":
    analyze_result()
