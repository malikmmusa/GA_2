"""
RED phase — TDD tests for SAM2 integration into GASegmenterService (Task 3).

Expected failures:
  - Group 1: AttributeError / TypeError — 'use_sam' attribute/param not yet on GASegmenterService
  - Groups 2-4: AttributeError from @patch — SAMRefiner not yet imported in ga_segmenter.py

Do NOT modify ga_segmenter.py until GREEN phase is approved.
"""
import os
import sys
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.services.ga_segmenter import GASegmenterService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image(h: int = 200, w: int = 200) -> np.ndarray:
    """BGR image with a bright rectangle — K-means will find it as the brightest cluster."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[60:140, 60:140] = 200  # bright rectangle
    return img


def _make_mock_sam(available: bool = True) -> MagicMock:
    """Return a configured SAMRefiner mock instance."""
    mock = MagicMock()
    mock.available = available
    mock.refine_candidates.return_value = []
    return mock


# ---------------------------------------------------------------------------
# Group 1 — Constructor accepts `use_sam` parameter
# ---------------------------------------------------------------------------

class TestGASegmenterInitUseSam(unittest.TestCase):

    def test_init_default_use_sam_is_true(self):
        """GASegmenterService() stores use_sam=True by default."""
        seg = GASegmenterService()
        self.assertIs(seg.use_sam, True)

    def test_init_use_sam_false_stored(self):
        """GASegmenterService(use_sam=False) stores use_sam=False."""
        seg = GASegmenterService(use_sam=False)
        self.assertIs(seg.use_sam, False)


# ---------------------------------------------------------------------------
# Group 2 — SAMRefiner is called when use_sam=True and available
# ---------------------------------------------------------------------------

class TestGASegmenterSAMCalled(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_calls_sam_set_image_when_available(self, MockSAMRefiner):
        """set_image() is called once on the en-face BGR array when SAM is available."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_image()
        seg.segment_ga_regions(image)

        self.assertTrue(mock_sam.set_image.called)

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_calls_refine_candidates_with_boxes(self, MockSAMRefiner):
        """refine_candidates() is called with a list of numpy arrays (bounding boxes)."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_image()
        seg.segment_ga_regions(image)

        self.assertTrue(mock_sam.refine_candidates.called)
        call_args = mock_sam.refine_candidates.call_args
        boxes = call_args[0][0]  # first positional arg
        self.assertIsInstance(boxes, list)
        for box in boxes:
            self.assertIsInstance(box, np.ndarray)

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_uses_sam_contours_when_returned(self, MockSAMRefiner):
        """When SAM returns a contour, it is used instead of the K-means contour."""
        sam_contour = np.array(
            [[[70, 70]], [[70, 120]], [[120, 120]], [[120, 70]]], dtype=np.int32
        )
        mock_sam = _make_mock_sam(available=True)
        mock_sam.refine_candidates.return_value = [
            {
                "mask": np.ones((200, 200), dtype=bool),
                "iou": 0.9,
                "contour": sam_contour,
            }
        ]
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_image()
        result = seg.segment_ga_regions(image)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # At least one returned contour must match the SAM contour
        found = any(np.array_equal(cnt, sam_contour) for cnt in result)
        self.assertTrue(found, "Expected SAM contour to appear in segment_ga_regions output")


# ---------------------------------------------------------------------------
# Group 3 — Fallback when SAM unavailable or disabled
# ---------------------------------------------------------------------------

class TestGASegmenterSAMFallback(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_falls_back_to_kmeans_when_sam_unavailable(self, MockSAMRefiner):
        """When SAMRefiner.available is False, K-means contours are returned and refine_candidates is NOT called."""
        mock_sam = _make_mock_sam(available=False)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_image()
        result = seg.segment_ga_regions(image)

        self.assertIsInstance(result, list)
        # K-means should find the bright rectangle
        self.assertGreater(len(result), 0, "K-means fallback should return at least one contour")
        mock_sam.refine_candidates.assert_not_called()

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_does_not_call_sam_when_use_sam_false(self, MockSAMRefiner):
        """When use_sam=False, refine_candidates is never called even if SAM is available."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService(use_sam=False)
        image = _make_synthetic_image()
        seg.segment_ga_regions(image)

        mock_sam.refine_candidates.assert_not_called()


# ---------------------------------------------------------------------------
# Group 4 — API contract preserved
# ---------------------------------------------------------------------------

class TestGASegmenterAPIContract(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_returns_list_of_numpy_arrays(self, MockSAMRefiner):
        """Return type is always List[np.ndarray], even when SAM returns a valid contour."""
        sam_contour = np.array(
            [[[70, 70]], [[70, 120]], [[120, 120]], [[120, 70]]], dtype=np.int32
        )
        mock_sam = _make_mock_sam(available=True)
        mock_sam.refine_candidates.return_value = [
            {
                "mask": np.ones((200, 200), dtype=bool),
                "iou": 0.9,
                "contour": sam_contour,
            }
        ]
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_image()
        result = seg.segment_ga_regions(image)

        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, np.ndarray)


# ---------------------------------------------------------------------------
# Group 5 — RGB conversion before SAM set_image
# ---------------------------------------------------------------------------

class TestGASegmenterRGBConversion(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_regions_passes_rgb_to_sam_set_image(self, MockSAMRefiner):
        """set_image() is called with an RGB image (not BGR) — first channel != third for non-grey images."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        # Create image with R != B to distinguish RGB vs BGR
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[60:140, 60:140, 0] = 200  # B channel high in BGR
        img[60:140, 60:140, 2] = 50   # R channel low in BGR
        # After BGR→RGB: channel 0 = 50, channel 2 = 200

        seg = GASegmenterService()
        seg.segment_ga_regions(img)

        self.assertTrue(mock_sam.set_image.called)
        passed_image = mock_sam.set_image.call_args[0][0]
        # In the passed RGB image, channel 0 (R) should be 50, channel 2 (B) should be 200
        self.assertEqual(int(passed_image[100, 100, 0]), 50)   # R channel
        self.assertEqual(int(passed_image[100, 100, 2]), 200)  # B channel


if __name__ == "__main__":
    unittest.main()
