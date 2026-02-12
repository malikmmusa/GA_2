"""Unit tests for image registration service."""
import pytest
import numpy as np
import cv2
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.services.image_registrar import ImageRegistrarService


class TestImageRegistrarService:
    """Test suite for ImageRegistrarService."""
    
    @pytest.fixture
    def service(self):
        """Create a registrar service instance for testing."""
        return ImageRegistrarService()
    
    @pytest.fixture
    def synthetic_image(self):
        """
        Create a synthetic test image with features.
        
        Returns a 300x300 grayscale image with some visual features.
        """
        img = np.zeros((300, 300), dtype=np.uint8)
        
        # Add some features (circles and lines to simulate vessels)
        cv2.circle(img, (100, 100), 20, 255, 2)
        cv2.circle(img, (200, 150), 15, 255, 2)
        cv2.circle(img, (150, 200), 25, 255, 2)
        cv2.line(img, (50, 50), (250, 250), 200, 2)
        cv2.line(img, (250, 50), (50, 250), 200, 2)
        cv2.circle(img, (150, 150), 30, 180, -1)
        
        return img
    
    def test_service_initialization(self, service):
        """Test that service initializes with correct parameters."""
        assert service.n_features == 5000
        assert service.ratio_test_threshold == 0.75
        assert service.ransac_threshold == 5.0
        assert service.min_inliers_success == 15
        assert service.min_inliers_low_confidence == 8
        assert service.orb is not None
        assert service.matcher is not None
    
    def test_enhance_vessels(self, service, synthetic_image):
        """Test CLAHE vessel enhancement."""
        enhanced = service._enhance_vessels(synthetic_image)
        
        assert enhanced.shape == synthetic_image.shape
        assert enhanced.dtype == np.uint8
        # Enhanced image should have different histogram than original
        assert not np.array_equal(enhanced, synthetic_image)
    
    def test_registration_identical_images(self, service, synthetic_image):
        """Test registration with identical images (should produce identity transform)."""
        # Create composite images (B-scan + En-face)
        composite_ref = np.hstack([synthetic_image, synthetic_image])
        composite_new = np.hstack([synthetic_image, synthetic_image])
        
        # Convert to BGR
        composite_ref = cv2.cvtColor(composite_ref, cv2.COLOR_GRAY2BGR)
        composite_new = cv2.cvtColor(composite_new, cv2.COLOR_GRAY2BGR)
        
        matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
            composite_ref,
            composite_new,
            en_face_split_x_ref=300,
            en_face_split_x_new=300
        )
        
        assert matrix is not None, "Registration should succeed for identical images"
        assert status == "success", f"Status should be success, got {status}"
        assert confidence >= 0.8, f"Confidence should be high for identical images, got {confidence}"
        assert num_matches > 0, "Should find matches"
        assert num_inliers > 0, "Should have inliers"
        
        # Transform should be approximately identity
        # Matrix is 2x3: [[a, b, tx], [c, d, ty]]
        # For identity: a≈1, b≈0, c≈0, d≈1, tx≈0, ty≈0
        assert abs(matrix[0, 0] - 1.0) < 0.1, "Scale X should be near 1"
        assert abs(matrix[1, 1] - 1.0) < 0.1, "Scale Y should be near 1"
        assert abs(matrix[0, 1]) < 0.1, "Rotation should be near 0"
        assert abs(matrix[1, 0]) < 0.1, "Rotation should be near 0"
        assert abs(matrix[0, 2]) < 10, "Translation X should be small"
        assert abs(matrix[1, 2]) < 10, "Translation Y should be small"
    
    def test_registration_translated_image(self, service, synthetic_image):
        """Test registration with translated image."""
        # Create a translated version
        translation_x = 20
        translation_y = 15
        M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        translated = cv2.warpAffine(synthetic_image, M_translate, (300, 300))
        
        # Create composite images
        composite_ref = np.hstack([synthetic_image, synthetic_image])
        composite_new = np.hstack([synthetic_image, translated])
        
        # Convert to BGR
        composite_ref = cv2.cvtColor(composite_ref, cv2.COLOR_GRAY2BGR)
        composite_new = cv2.cvtColor(composite_new, cv2.COLOR_GRAY2BGR)
        
        matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
            composite_ref,
            composite_new,
            en_face_split_x_ref=300,
            en_face_split_x_new=300
        )
        
        assert matrix is not None, "Registration should succeed for translated image"
        assert status in ["success", "low_confidence"], f"Status should be success or low_confidence, got {status}"
        assert num_matches > 0, "Should find matches"
        assert num_inliers > 0, "Should have inliers"
        
        # Check that detected translation is approximately correct
        detected_tx = matrix[0, 2]
        detected_ty = matrix[1, 2]
        assert abs(detected_tx - translation_x) < 5, f"Translation X detection error too large: expected {translation_x}, got {detected_tx}"
        assert abs(detected_ty - translation_y) < 5, f"Translation Y detection error too large: expected {translation_y}, got {detected_ty}"
    
    def test_registration_different_images(self, service):
        """Test registration with completely different images (should fail)."""
        # Create two different random images
        np.random.seed(42)
        img1 = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        
        # Create composite images
        composite_ref = np.hstack([img1, img1])
        composite_new = np.hstack([img2, img2])
        
        # Convert to BGR
        composite_ref = cv2.cvtColor(composite_ref, cv2.COLOR_GRAY2BGR)
        composite_new = cv2.cvtColor(composite_new, cv2.COLOR_GRAY2BGR)
        
        matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
            composite_ref,
            composite_new,
            en_face_split_x_ref=300,
            en_face_split_x_new=300
        )
        
        # Should fail or have low confidence
        assert status == "failed", f"Status should be failed for different images, got {status}"
        assert confidence < 0.4, f"Confidence should be low for different images, got {confidence}"
    
    def test_transform_landmarks(self, service):
        """Test landmark transformation with a known transform."""
        # Create a simple translation transform
        matrix = np.array([[1.0, 0.0, 10.0],
                          [0.0, 1.0, 20.0]], dtype=np.float32)
        
        # Original fovea position (in reference image original coordinates)
        fovea_x_ref = 500.0
        fovea_y_ref = 300.0
        en_face_split_x_ref = 400
        en_face_split_x_new = 400
        
        # Transform landmarks
        result = service.transform_landmarks(
            matrix,
            fovea_x_ref,
            fovea_y_ref,
            en_face_split_x_ref,
            en_face_split_x_new
        )
        
        assert 'transformed_fovea_x' in result
        assert 'transformed_fovea_y' in result
        
        # Expected: fovea_local_x = 500 - 400 = 100
        #           fovea_local_y = 300
        #           transformed_local = (100 + 10, 300 + 20) = (110, 320)
        #           transformed_original = (110 + 400, 320) = (510, 320)
        expected_x = 510.0
        expected_y = 320.0
        
        assert abs(result['transformed_fovea_x'] - expected_x) < 0.1, \
            f"Expected fovea_x={expected_x}, got {result['transformed_fovea_x']}"
        assert abs(result['transformed_fovea_y'] - expected_y) < 0.1, \
            f"Expected fovea_y={expected_y}, got {result['transformed_fovea_y']}"
    
    def test_transform_landmarks_with_disc(self, service):
        """Test landmark transformation including disc coordinates."""
        matrix = np.array([[1.0, 0.0, 5.0],
                          [0.0, 1.0, 10.0]], dtype=np.float32)
        
        fovea_x = 500.0
        fovea_y = 300.0
        disc_x = 600.0
        disc_y = 350.0
        en_face_split_x_ref = 400
        en_face_split_x_new = 400
        
        result = service.transform_landmarks(
            matrix,
            fovea_x,
            fovea_y,
            en_face_split_x_ref,
            en_face_split_x_new,
            disc_x,
            disc_y
        )
        
        assert result['transformed_disc_center_x'] is not None
        assert result['transformed_disc_center_y'] is not None
        
        # Expected disc: local = (600-400, 350) = (200, 350)
        #                transformed = (200+5, 350+10) = (205, 360)
        #                original = (205+400, 360) = (605, 360)
        expected_disc_x = 605.0
        expected_disc_y = 360.0
        
        assert abs(result['transformed_disc_center_x'] - expected_disc_x) < 0.1
        assert abs(result['transformed_disc_center_y'] - expected_disc_y) < 0.1
    
    def test_empty_image(self, service):
        """Test registration with empty images (should fail gracefully)."""
        # Create empty black images
        img = np.zeros((300, 300), dtype=np.uint8)
        composite_ref = np.hstack([img, img])
        composite_new = np.hstack([img, img])
        
        composite_ref = cv2.cvtColor(composite_ref, cv2.COLOR_GRAY2BGR)
        composite_new = cv2.cvtColor(composite_new, cv2.COLOR_GRAY2BGR)
        
        matrix, confidence, num_matches, num_inliers, status, message = service.register_images(
            composite_ref,
            composite_new,
            en_face_split_x_ref=300,
            en_face_split_x_new=300
        )
        
        # Should fail due to no features
        assert status == "failed"
        assert confidence == 0.0
        assert num_matches == 0
        assert num_inliers == 0
    
    def test_confidence_scoring(self, service):
        """Test confidence scoring logic."""
        # Test high confidence
        confidence, status, message = service._compute_confidence(100, 50)
        assert status == "success"
        assert confidence >= 0.8
        
        # Test low confidence
        confidence, status, message = service._compute_confidence(50, 10)
        assert status == "low_confidence"
        assert 0.4 <= confidence < 0.8
        
        # Test failed
        confidence, status, message = service._compute_confidence(50, 2)
        assert status == "failed"
        assert confidence < 0.4
        
        # Test no matches
        confidence, status, message = service._compute_confidence(0, 0)
        assert status == "failed"
        assert confidence == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
