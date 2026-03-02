"""Tests for SAM-enabled GA segmenter dependency wiring."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch


class TestGASegmentationRouteSam(unittest.TestCase):
    def test_get_ga_segmenter_passes_use_sam_true(self) -> None:
        """Dependency factory must instantiate GASegmenterService with use_sam=True."""
        import src.api.dependencies as dependencies

        dependencies.get_ga_segmenter.cache_clear()
        try:
            with patch("src.api.dependencies.GASegmenterService") as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                first = dependencies.get_ga_segmenter()
                second = dependencies.get_ga_segmenter()

                self.assertIs(first, mock_instance)
                self.assertIs(second, mock_instance)
                mock_service.assert_called_once_with(use_sam=True)
        finally:
            dependencies.get_ga_segmenter.cache_clear()

    def test_requirements_contains_sam2(self) -> None:
        """requirements.txt must declare the SAM-2 dependency."""
        req_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        contents = req_path.read_text(encoding="utf-8")
        self.assertIn("SAM-2", contents)
