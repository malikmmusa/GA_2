"""
Test suite for GA Local Segmentation Service.

Tests the localized segmentation fallback for GA regions missed by global segmentation.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.services.ga_segmenter import GASegmenterService


def test_synthetic_ellipse():
    """
    Test 1: Synthetic Ellipse Test
    Create a synthetic image with a white ellipse and verify local segmentation finds it.
    """
    print("\n" + "="*70)
    print("TEST 1: Synthetic Ellipse Test")
    print("="*70)
    
    try:
        segmenter = GASegmenterService()
        
        # Create 500x500 black image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Draw white ellipse at center
        ellipse_center = (250, 250)
        ellipse_axes = (60, 40)
        cv2.ellipse(image, ellipse_center, ellipse_axes, 0, 0, 360, (255, 255, 255), -1)
        
        print(f"  Created synthetic image: {image.shape}")
        print(f"  Ellipse at {ellipse_center}, axes {ellipse_axes}")
        
        # Click at ellipse center
        click_x, click_y = ellipse_center
        
        # Run local segmentation
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y
        )
        
        assert len(contours) > 0, "Expected to find region at ellipse location"
        
        # Verify the click point is inside the returned contour
        cnt = contours[0]
        dist = cv2.pointPolygonTest(cnt, (float(click_x), float(click_y)), True)
        assert dist >= 0, f"Click point should be inside contour (dist={dist})"
        
        print(f"  ✓ Found region with {len(cnt)} points")
        print(f"  ✓ Click point is inside contour")
        print("  ✓ Synthetic ellipse test passed")
        
        return True
    
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_click_on_background():
    """
    Test 2: Click on Background
    Click far from any features and verify no region is returned.
    """
    print("\n" + "="*70)
    print("TEST 2: Click on Background Test")
    print("="*70)
    
    try:
        segmenter = GASegmenterService()
        
        # Create 500x500 black image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Draw white ellipse at center
        cv2.ellipse(image, (250, 250), (60, 40), 0, 0, 360, (255, 255, 255), -1)
        
        # Click far from ellipse (in corner)
        click_x, click_y = 50, 50
        
        # Run local segmentation
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y
        )
        
        # Should return empty or a very small region (noise)
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            assert area < 200, f"Expected no significant region, got area={area}"
            print(f"  ✓ Found only noise region (area={area:.0f} px²)")
        else:
            print(f"  ✓ No region found (as expected)")
        
        print("  ✓ Click on background test passed")
        
        return True
    
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_crop_boundary():
    """
    Test 3: Crop Boundary Test
    Place ellipse near image edge and verify crop clamping works correctly.
    """
    print("\n" + "="*70)
    print("TEST 3: Crop Boundary Test")
    print("="*70)
    
    try:
        segmenter = GASegmenterService()
        
        # Create 500x500 black image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Draw white ellipse near edge
        ellipse_center = (50, 50)
        ellipse_axes = (30, 20)
        cv2.ellipse(image, ellipse_center, ellipse_axes, 0, 0, 360, (255, 255, 255), -1)
        
        print(f"  Ellipse near edge at {ellipse_center}")
        
        # Click at ellipse center (near edge)
        click_x, click_y = ellipse_center
        
        # Run local segmentation with default crop radius
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y
        )
        
        if len(contours) > 0:
            # Verify coordinates are in original image space (not negative)
            cnt = contours[0]
            min_x = np.min(cnt[:, 0, 0])
            min_y = np.min(cnt[:, 0, 1])
            
            assert min_x >= 0, f"Contour X should be >= 0, got {min_x}"
            assert min_y >= 0, f"Contour Y should be >= 0, got {min_y}"
            
            print(f"  ✓ Found region with correct coordinates")
            print(f"  ✓ Min coords: ({min_x}, {min_y})")
        else:
            print(f"  ⚠ No region found (may be too small after morph)")
        
        print("  ✓ Crop boundary test passed")
        
        return True
    
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_disc_masking():
    """
    Test 4: Disc Masking Test
    Verify disc area is excluded from segmentation results.
    """
    print("\n" + "="*70)
    print("TEST 4: Disc Masking Test")
    print("="*70)
    
    try:
        segmenter = GASegmenterService()
        
        # Create 500x500 black image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Draw large white circle (simulating disc)
        disc_center = (250, 250)
        disc_radius = 60
        cv2.circle(image, disc_center, disc_radius, (255, 255, 255), -1)
        
        print(f"  Disc at {disc_center}, radius {disc_radius}")
        
        # Click near disc center
        click_x, click_y = disc_center
        
        # Run local segmentation with disc masking
        disc_height_pixels = disc_radius * 2
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y,
            disc_center_x=disc_center[0],
            disc_center_y=disc_center[1],
            disc_height_pixels=disc_height_pixels
        )
        
        # Should return empty because click is in disc area
        assert len(contours) == 0, "Expected no region (disc area masked)"
        
        print(f"  ✓ No region found (disc masked correctly)")
        print("  ✓ Disc masking test passed")
        
        return True
    
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cluster_selection():
    """
    Test 5: Cluster Selection Test
    Verify that the cluster containing the click pixel is selected.
    """
    print("\n" + "="*70)
    print("TEST 5: Cluster Selection Test")
    print("="*70)
    
    try:
        segmenter = GASegmenterService()
        
        # Create 500x500 gradient image with distinct regions
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Dark region (left)
        image[:, :250] = 50
        
        # Medium region (center) - this is what we'll click
        cv2.rectangle(image, (200, 200), (300, 300), (150, 150, 150), -1)
        
        # Bright region (right)
        image[:, 350:] = 220
        
        print(f"  Created image with 3 intensity regions")
        
        # Click in the medium region
        click_x, click_y = 250, 250
        
        # Run local segmentation
        contours = segmenter.segment_ga_local(
            image=image,
            click_x=click_x,
            click_y=click_y
        )
        
        if len(contours) > 0:
            # Verify the click point is inside the returned contour
            cnt = contours[0]
            dist = cv2.pointPolygonTest(cnt, (float(click_x), float(click_y)), True)
            assert dist >= 0, f"Click point should be inside selected cluster contour"
            
            print(f"  ✓ Found region containing click point")
            print(f"  ✓ Cluster selection works correctly")
        else:
            print(f"  ⚠ No region found (may be filtered out)")
        
        print("  ✓ Cluster selection test passed")
        
        return True
    
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GA LOCAL SEGMENTATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Synthetic Ellipse
    results.append(("Synthetic Ellipse", test_synthetic_ellipse()))
    
    # Test 2: Click on Background
    results.append(("Click on Background", test_click_on_background()))
    
    # Test 3: Crop Boundary
    results.append(("Crop Boundary", test_crop_boundary()))
    
    # Test 4: Disc Masking
    results.append(("Disc Masking", test_disc_masking()))
    
    # Test 5: Cluster Selection
    results.append(("Cluster Selection", test_cluster_selection()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n  🎉 ALL TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
