import numpy as np
import sys
import os

# Add src to path to import the model logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.retfound_unet import HeatmapGenerator, get_coordinates_from_heatmap

def test_coordinate_recovery():
    print("=== Testing Coordinate Recovery Logic ===")
    
    # Test parameters
    H, W = 224, 224
    sigma = 20
    test_points = [
        (112.0, 112.0), # Center
        (50.5, 50.5),   # Sub-pixel
        (10.0, 10.0),   # Corner
        (200.0, 150.0)  # Off-center
    ]
    
    max_error = 0.0
    
    for true_x, true_y in test_points:
        # 1. Generate Heatmap
        heatmap = HeatmapGenerator(true_x, true_y, H, W, sigma=sigma)
        
        # 2. Add slight noise (simulating model imperfection)
        noise = np.random.normal(0, 0.001, heatmap.shape)
        heatmap_noisy = heatmap + noise
        heatmap_noisy[heatmap_noisy < 0] = 0 # ReLU
        
        # 3. Recover Coordinates
        pred_x, pred_y = get_coordinates_from_heatmap(heatmap_noisy)
        
        # 4. Calculate Error
        dist = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        max_error = max(max_error, dist)
        
        print(f"True: ({true_x:5.1f}, {true_y:5.1f}) | Pred: ({pred_x:5.1f}, {pred_y:5.1f}) | Error: {dist:.4f} px")
        
    print(f"\nMax Error: {max_error:.4f} px")
    
    if max_error < 0.5:
        print("✅ PASSED: Coordinate recovery is sub-pixel accurate.")
    else:
        print("❌ FAILED: Coordinate recovery error is too high.")

if __name__ == "__main__":
    test_coordinate_recovery()
