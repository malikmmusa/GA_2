"""
RED phase — TDD tests for SAM2 point-prompt integration into segment_ga_local() (Task 4).

Expected failures (RED):
  - Group 1 tests: set_image / refine_point not called — segment_ga_local() does not
    invoke self._sam at all yet. AssertionError on .called assertions and coordinate checks.
  - Group 2 tests: may partially pass (K-means fallback already exists), but must be
    confirmed running and producing correct outcomes.

Do NOT modify ga_segmenter.py until RED failures are confirmed.
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

def _make_synthetic_composite(h: int = 200, w: int = 400) -> np.ndarray:
    """Composite image: left half B-scan (dark), right half en-face with bright region."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[60:140, 210:290] = 200  # bright rect in right (en-face) half
    return img


def _make_mock_sam(available: bool = True) -> MagicMock:
    """Return a configured SAMRefiner mock — refine_point returns None by default."""
    mock = MagicMock()
    mock.available = available
    mock.refine_point.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Group 1 — SAM point prompt called when available
# ---------------------------------------------------------------------------

class TestSegmentGALocalSAMCalled(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_calls_sam_set_image_when_available(self, MockSAMRefiner):
        """set_image() is called on the en-face image when SAM is available."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_composite()
        seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        self.assertTrue(mock_sam.set_image.called,
                        "set_image should be called when SAM is available")

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_calls_refine_point_with_local_coords(self, MockSAMRefiner):
        """refine_point() is called with en-face-local coordinates (click_x - en_face_split_x)."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_composite()
        seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        self.assertTrue(mock_sam.refine_point.called,
                        "refine_point should be called when SAM is available")
        call_kwargs = mock_sam.refine_point.call_args
        # Extract the point — may be positional or keyword
        if call_kwargs[0]:
            point = call_kwargs[0][0]
        else:
            point = call_kwargs[1]["point"]
        # click_x_local = 250 - 200 = 50, click_y_local = 100
        self.assertEqual(point[0], 50,
                         f"Expected x=50 (250-200), got {point[0]}")
        self.assertEqual(point[1], 100,
                         f"Expected y=100, got {point[1]}")

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_returns_sam_contour_when_refine_point_succeeds(self, MockSAMRefiner):
        """When refine_point returns a contour, it is returned (shifted by en_face_split_x)."""
        sam_contour = np.array(
            [[[10, 60]], [[10, 140]], [[90, 140]], [[90, 60]]], dtype=np.int32
        )
        mock_sam = _make_mock_sam(available=True)
        mock_sam.refine_point.return_value = {
            "mask": np.ones((200, 200), dtype=bool),
            "iou": 0.85,
            "contour": sam_contour,
        }
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_composite()
        result = seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1, "Expected exactly 1 contour from SAM fast-path")
        # X coordinates should be shifted by en_face_split_x=200 (10+200=210, 90+200=290)
        x_coords = result[0][:, 0, 0]
        self.assertTrue(
            np.all(x_coords >= 200),
            f"X coords should be shifted by en_face_split_x=200, got {x_coords}"
        )

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_passes_rgb_image_to_sam(self, MockSAMRefiner):
        """set_image() receives an RGB array (not BGR) — channels are swapped."""
        mock_sam = _make_mock_sam(available=True)
        MockSAMRefiner.return_value = mock_sam

        # Create image where B != R so we can distinguish BGR from RGB
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        # en-face region (right half): B=200, G=0, R=50 in BGR
        image[60:140, 210:290, 0] = 200  # B channel
        image[60:140, 210:290, 2] = 50   # R channel

        seg = GASegmenterService()
        seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        self.assertTrue(mock_sam.set_image.called)
        passed = mock_sam.set_image.call_args[0][0]
        # After BGR→RGB conversion: channel 0 = R = 50, channel 2 = B = 200
        self.assertEqual(int(passed[100, 50, 0]), 50,
                         "Channel 0 of passed image should be R=50 (BGR→RGB swap)")
        self.assertEqual(int(passed[100, 50, 2]), 200,
                         "Channel 2 of passed image should be B=200 (BGR→RGB swap)")


# ---------------------------------------------------------------------------
# Group 2 — Fallback when SAM unavailable or returns None
# ---------------------------------------------------------------------------

class TestSegmentGALocalFallback(unittest.TestCase):

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_falls_back_to_kmeans_when_sam_unavailable(self, MockSAMRefiner):
        """When SAM is unavailable, refine_point is NOT called and K-means runs."""
        mock_sam = _make_mock_sam(available=False)
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_composite()
        result = seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        mock_sam.refine_point.assert_not_called()
        self.assertIsInstance(result, list, "K-means fallback must return a list")

    @patch("src.api.services.ga_segmenter.SAMRefiner")
    def test_segment_ga_local_falls_back_to_kmeans_when_refine_point_returns_none(self, MockSAMRefiner):
        """When refine_point returns None, K-means runs as fallback."""
        mock_sam = _make_mock_sam(available=True)
        mock_sam.refine_point.return_value = None  # SAM tried but no result
        MockSAMRefiner.return_value = mock_sam

        seg = GASegmenterService()
        image = _make_synthetic_composite()
        result = seg.segment_ga_local(image, click_x=250, click_y=100, en_face_split_x=200)

        self.assertTrue(mock_sam.refine_point.called,
                        "refine_point should be called (SAM was attempted)")
        self.assertIsInstance(result, list, "Result must be a list (K-means fallback ran)")


if __name__ == "__main__":
    unittest.main()
