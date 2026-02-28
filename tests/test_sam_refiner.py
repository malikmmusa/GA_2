"""
Tests for SAMRefiner.

Mock shapes match real SAM2 predictor output:
  masks  → (num_masks, H, W)  float32
  scores → (num_masks,)       float32  1-D numpy array
"""
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from src.api.services.sam_refiner import SAMRefiner  # noqa: E402  — fails until GREEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 50, 50

def _fake_mask() -> np.ndarray:
    """Real SAM2 returns (num_masks, H, W) float32."""
    return np.ones((1, H, W), dtype=np.float32)


def _fake_scores() -> np.ndarray:
    """Real SAM2 returns 1-D float32 array, one score per mask."""
    return np.array([0.9], dtype=np.float32)


def _fake_scores_low() -> np.ndarray:
    return np.array([0.5], dtype=np.float32)


def _make_available_refiner() -> SAMRefiner:
    """Return a SAMRefiner instance with availability forced True and a mock predictor."""
    refiner = SAMRefiner.__new__(SAMRefiner)
    refiner.available = True
    predictor = MagicMock()
    predictor.predict.return_value = (_fake_mask(), _fake_scores(), None)
    refiner.predictor = predictor
    return refiner


# ---------------------------------------------------------------------------
# Group 1 — Initialization / fallback
# ---------------------------------------------------------------------------

class TestSAMRefinerInit(unittest.TestCase):

    def test_init_missing_checkpoint_sets_unavailable(self):
        """When checkpoint_path does not exist, available is False and no exception raised."""
        refiner = SAMRefiner(checkpoint_path="/nonexistent/path/sam2.pt")
        self.assertFalse(refiner.available)

    @patch("src.api.services.sam_refiner.build_sam2")
    def test_init_load_failure_sets_unavailable(self, mock_build):
        """When build_sam2 raises, available is still False and no crash."""
        mock_build.side_effect = RuntimeError("CUDA OOM")
        # Use a file that does exist on disk so we pass the path-existence check
        real_path = __file__  # this file exists
        refiner = SAMRefiner(checkpoint_path=real_path)
        self.assertFalse(refiner.available)


# ---------------------------------------------------------------------------
# Group 2 — set_image
# ---------------------------------------------------------------------------

class TestSAMRefinerSetImage(unittest.TestCase):

    def test_set_image_raises_when_unavailable(self):
        """set_image() raises RuntimeError when available=False."""
        refiner = SAMRefiner(checkpoint_path="/nonexistent/path/sam2.pt")
        self.assertFalse(refiner.available)
        image = np.zeros((H, W, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            refiner.set_image(image)

    def test_set_image_calls_predictor_set_image(self):
        """When available, set_image() delegates to predictor.set_image()."""
        refiner = _make_available_refiner()
        image = np.zeros((H, W, 3), dtype=np.uint8)
        refiner.set_image(image)
        refiner.predictor.set_image.assert_called_once_with(image)


# ---------------------------------------------------------------------------
# Group 3 — refine_candidates (box prompt)
# ---------------------------------------------------------------------------

class TestSAMRefinerCandidates(unittest.TestCase):

    def test_refine_candidates_returns_empty_when_unavailable(self):
        """refine_candidates() returns [] without raising when available=False."""
        refiner = SAMRefiner(checkpoint_path="/nonexistent/path/sam2.pt")
        result = refiner.refine_candidates([np.array([0, 0, 10, 10])])
        self.assertEqual(result, [])

    def test_refine_candidates_filters_low_iou(self):
        """IoU=0.5 is filtered out with default min_iou=0.7 → returns []."""
        refiner = _make_available_refiner()
        refiner.predictor.predict.return_value = (_fake_mask(), _fake_scores_low(), None)
        boxes = [np.array([5, 5, 45, 45])]
        result = refiner.refine_candidates(boxes)
        self.assertEqual(result, [])

    def test_refine_candidates_keeps_high_iou(self):
        """IoU=0.9 is kept with default min_iou=0.7 → returns 1 item."""
        refiner = _make_available_refiner()
        boxes = [np.array([5, 5, 45, 45])]
        result = refiner.refine_candidates(boxes)
        self.assertEqual(len(result), 1)
        self.assertIn("mask", result[0])
        self.assertIn("iou", result[0])
        self.assertIn("contour", result[0])

    def test_refine_candidates_contour_is_numpy_array(self):
        """contour is a numpy array with shape (N, 1, 2) — OpenCV contour format."""
        refiner = _make_available_refiner()
        boxes = [np.array([5, 5, 45, 45])]
        result = refiner.refine_candidates(boxes)
        self.assertEqual(len(result), 1)
        contour = result[0]["contour"]
        self.assertIsInstance(contour, np.ndarray)
        self.assertEqual(contour.ndim, 3)
        self.assertEqual(contour.shape[1], 1)
        self.assertEqual(contour.shape[2], 2)

    def test_refine_candidates_handles_empty_predictor_output(self):
        """If predictor returns empty masks/scores, result is silently skipped."""
        refiner = _make_available_refiner()
        refiner.predictor.predict.return_value = (
            np.zeros((0, H, W), dtype=np.float32),
            np.array([], dtype=np.float32),
            None,
        )
        boxes = [np.array([5, 5, 45, 45])]
        # Must not raise IndexError
        result = refiner.refine_candidates(boxes)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Group 4 — refine_point (point prompt)
# ---------------------------------------------------------------------------

class TestSAMRefinerPoint(unittest.TestCase):

    def test_refine_point_returns_none_when_unavailable(self):
        """refine_point() returns None when available=False."""
        refiner = SAMRefiner(checkpoint_path="/nonexistent/path/sam2.pt")
        result = refiner.refine_point(point=(25, 25))
        self.assertIsNone(result)

    def test_refine_point_returns_dict_with_required_keys(self):
        """When available, refine_point() returns dict with mask, iou, contour."""
        refiner = _make_available_refiner()
        result = refiner.refine_point(point=(25, 25))
        self.assertIsNotNone(result)
        self.assertIn("mask", result)
        self.assertIn("iou", result)
        self.assertIn("contour", result)

    def test_refine_point_handles_empty_predictor_output(self):
        """If predictor returns empty masks/scores, returns None."""
        refiner = _make_available_refiner()
        refiner.predictor.predict.return_value = (
            np.zeros((0, H, W), dtype=np.float32),
            np.array([], dtype=np.float32),
            None,
        )
        # Must not raise IndexError
        result = refiner.refine_point(point=(25, 25))
        self.assertIsNone(result)


