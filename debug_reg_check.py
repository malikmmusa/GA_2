import cv2
import numpy as np
from pathlib import Path

def debug_registration():
    base = Path("OCT_Pipeline_2026")
    clean_path = base / "data/processed/en_face/1954640.png"
    inpainted_path = base / "data/processed_inpainted/en_face/1954640.png"
    
    if not clean_path.exists() or not inpainted_path.exists():
        print("Files missing")
        return

    img_clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    img_inpainted = cv2.imread(str(inpainted_path), cv2.IMREAD_GRAYSCALE)
    
    print(f"Clean Shape: {img_clean.shape}")
    print(f"Inpainted Shape: {img_inpainted.shape}")
    
    # Template Matching
    res = cv2.matchTemplate(img_clean, img_inpainted, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"Max Correlation: {max_val}")
    print(f"Offset (x, y) of Inpainted INSIDE Clean: {max_loc}")
    
    dx, dy = max_loc
    h, w = img_inpainted.shape
    
    # Create composite
    # Clean is larger (presumably), so we place Inpainted into it
    composite = cv2.cvtColor(img_clean, cv2.COLOR_GRAY2BGR)
    
    # Draw Inpainted rectangle in Green
    # We can't easily alpha blend grayscale without converting.
    # Let's just create a blank overlay
    overlay = np.zeros_like(composite)
    overlay[dy:dy+h, dx:dx+w, 1] = img_inpainted # Green channel
    
    # Clean in Red channel
    composite[:,:,2] = img_clean # Red
    composite[:,:,0] = 0 # Blue
    
    # Combine (Green + Red = Yellow)
    # We need to handle the overlap area carefully
    # Where both exist: Yellow. Where only Clean: Red.
    
    # Better visualization: 
    # Base = Clean (Gray)
    # Rectangle = Inpainted (Green outline)
    debug_vis = cv2.cvtColor(img_clean, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(debug_vis, (dx, dy), (dx+w, dy+h), (0, 255, 0), 3)
    
    cv2.imwrite("debug_reg_1954640.png", debug_vis)
    print("Saved debug_reg_1954640.png")
    
    # Calculate Label Translation
    # Inpainted Label was x=2032
    # Clean Label = x_inpainted + dx
    # Let's verify this.
    # If Inpainted is at x=dx inside Clean.
    # Then a point (px, py) in Inpainted is at (dx+px, dy+py) in Clean.
    
    old_x = 2032
    new_x = old_x + dx
    print(f"Label Translation: {old_x} -> {new_x}")

if __name__ == "__main__":
    debug_registration()
