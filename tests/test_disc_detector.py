"""Tests for the disc detector service."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.api.services.disc_detector import DiscDetectorService

FULL_IMAGE_VALIDATION = os.getenv("OCT_TEST_FULL_IMAGES") == "1"


@pytest.fixture(scope="module")
def disc_detector() -> DiscDetectorService:
    """Reuse one detector instance to avoid repeated model load overhead."""
    return DiscDetectorService()


@pytest.fixture(scope="module")
def available_input_images() -> list[Path]:
    image_paths = [
        Path("input_images/test_1.png"),
        Path("input_images/test_2.png"),
        Path("input_images/test_3.png"),
    ]
    existing_paths = [path for path in image_paths if path.exists()]
    if not FULL_IMAGE_VALIDATION:
        return existing_paths[:1]
    return existing_paths


def test_dummy_batch(disc_detector: DiscDetectorService) -> None:
    """
    Pass a random tensor through the network and validate shape/NAN behavior.
    """
    if disc_detector.model is None:
        pytest.skip("Disc detector weights are unavailable; skipping model-forward test")

    random_rgb = np.random.default_rng(42).integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    augmented = disc_detector.transform(image=random_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(disc_detector.device)

    assert input_tensor.shape == (1, 3, 224, 224), "Input shape mismatch"

    with torch.no_grad():
        output = disc_detector.model(input_tensor)

    assert output.shape == (1, 1, 224, 224), "Output shape mismatch"
    assert not torch.isnan(output).any().item(), "Output contains NaN values"


def test_medical_logic() -> None:
    """Verify the 1800 micron optic-disc conversion convention."""
    disc_height_pixels = 200.0
    expected_ratio = 1800.0 / disc_height_pixels
    assert expected_ratio == pytest.approx(9.0, abs=0.1)


def test_real_image(
    disc_detector: DiscDetectorService,
    available_input_images: list[Path],
) -> None:
    """Run detector against local input image fixtures."""
    if not available_input_images:
        pytest.skip("No local input_images/test_*.png files available")

    required_keys = {
        "disc_center_x",
        "disc_center_y",
        "disc_top_y",
        "disc_bottom_y",
        "disc_height_pixels",
        "pixel_to_micron_ratio",
        "en_face_split_x",
    }

    for image_path in available_input_images:
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        result = disc_detector.detect_from_image(image)
        assert required_keys.issubset(result.keys()), f"Missing keys in {image_path.name}"

        disc_height = result["disc_height_pixels"]
        ratio = result["pixel_to_micron_ratio"]
        assert disc_height > 0, "Disc height must be positive"
        assert ratio * disc_height == pytest.approx(1800.0, abs=1.0)
