"""Tests for autonomous GA measurement confidence / abstention."""

import numpy as np
import pytest

from src.api.services.ga_segmenter import GASegmenterService


@pytest.fixture
def segmenter():
    return GASegmenterService(use_sam=False)


def test_confidence_is_best_region_score(segmenter):
    assert segmenter.measurement_confidence([0.2, 0.9, 0.5]) == pytest.approx(0.9)


def test_no_regions_means_zero_confidence(segmenter):
    """An empty candidate list is the clearest possible signal not to report."""
    assert segmenter.measurement_confidence([]) == 0.0
    assert not segmenter.is_confident([])


def test_threshold_gates_reporting(segmenter):
    t = segmenter.MIN_CONFIDENT_SCORE
    assert segmenter.is_confident([t])
    assert segmenter.is_confident([t + 0.01])
    assert not segmenter.is_confident([t - 0.01])


def test_segment_returns_confidence_when_asked(segmenter):
    """return_confidence changes the return shape without changing the contours."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[60:140, 260:340] = 200  # one bright blob in the en-face half

    contours = segmenter.segment_ga_regions(image, en_face_split_x=200)
    result = segmenter.segment_ga_regions(image, en_face_split_x=200, return_confidence=True)

    assert isinstance(result, tuple) and len(result) == 2
    returned_contours, confidence = result
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert len(returned_contours) == len(contours)


def test_confidence_is_zero_on_blank_image(segmenter):
    """A blank image yields no supportable region, so nothing should be reported."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    contours, confidence = segmenter.segment_ga_regions(
        image, en_face_split_x=200, return_confidence=True)
    assert contours == [] or confidence == 0.0
    if not contours:
        assert confidence == 0.0
        assert not segmenter.is_confident([confidence])
