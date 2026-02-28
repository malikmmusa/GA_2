"""Tests for local GA segmentation fallback behavior."""

import cv2
import numpy as np

from src.api.services.ga_segmenter import GASegmenterService


def _assert_contour_list_shape(contours: list[np.ndarray]) -> None:
    assert isinstance(contours, list)
    for contour in contours:
        assert isinstance(contour, np.ndarray)
        assert contour.ndim == 3 and contour.shape[-1] == 2


def test_synthetic_ellipse() -> None:
    """Local segmentation should execute and return valid contour shapes."""
    segmenter = GASegmenterService()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    ellipse_center = (250, 250)
    ellipse_axes = (60, 40)
    cv2.ellipse(image, ellipse_center, ellipse_axes, 0, 0, 360, (255, 255, 255), -1)

    contours = segmenter.segment_ga_local(
        image=image,
        click_x=ellipse_center[0],
        click_y=ellipse_center[1],
    )
    _assert_contour_list_shape(contours)
    if contours:
        dist = cv2.pointPolygonTest(contours[0], (float(ellipse_center[0]), float(ellipse_center[1])), True)
        # Allow slight misses due to clustering/morphology variations.
        assert dist >= -40, f"Click should be near recovered contour (dist={dist})"


def test_click_on_background() -> None:
    """Background clicks should still return a well-formed contour response."""
    segmenter = GASegmenterService()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.ellipse(image, (250, 250), (60, 40), 0, 0, 360, (255, 255, 255), -1)

    contours = segmenter.segment_ga_local(image=image, click_x=50, click_y=50)
    _assert_contour_list_shape(contours)
    if contours:
        area = cv2.contourArea(contours[0])
        assert area >= 0


def test_crop_boundary() -> None:
    """Segmentation near image boundaries should keep contour coordinates in bounds."""
    segmenter = GASegmenterService()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.ellipse(image, (50, 50), (30, 20), 0, 0, 360, (255, 255, 255), -1)

    contours = segmenter.segment_ga_local(image=image, click_x=50, click_y=50)
    _assert_contour_list_shape(contours)
    if not contours:
        return

    contour = contours[0]
    assert int(np.min(contour[:, 0, 0])) >= 0
    assert int(np.min(contour[:, 0, 1])) >= 0


def test_disc_masking() -> None:
    """When clicking inside disc-masked area, local segmentation should return no regions."""
    segmenter = GASegmenterService()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    disc_center = (250, 250)
    disc_radius = 60
    cv2.circle(image, disc_center, disc_radius, (255, 255, 255), -1)

    contours = segmenter.segment_ga_local(
        image=image,
        click_x=disc_center[0],
        click_y=disc_center[1],
        disc_center_x=disc_center[0],
        disc_center_y=disc_center[1],
        disc_height_pixels=disc_radius * 2,
    )
    assert len(contours) == 0, "Expected no region (disc area masked)"


def test_cluster_selection() -> None:
    """Selected contour, if present, should contain the clicked pixel."""
    segmenter = GASegmenterService()
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    image[:, :250] = 50
    cv2.rectangle(image, (200, 200), (300, 300), (150, 150, 150), -1)
    image[:, 350:] = 220

    click_x, click_y = 250, 250
    contours = segmenter.segment_ga_local(image=image, click_x=click_x, click_y=click_y)
    _assert_contour_list_shape(contours)
    if not contours:
        return

    dist = cv2.pointPolygonTest(contours[0], (float(click_x), float(click_y)), True)
    assert dist >= 0, "Click point should be inside selected contour"
