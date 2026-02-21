"""Task 5: Verify get_ga_segmenter() explicitly passes use_sam=True to GASegmenterService."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import patch, MagicMock


class TestGASegmentationRouteSam(unittest.TestCase):

    def test_get_ga_segmenter_passes_use_sam_true(self):
        """get_ga_segmenter() must instantiate GASegmenterService with use_sam=True explicitly."""
        import src.api.routes.ga_segmentation as route_module

        # Reset the singleton so get_ga_segmenter() re-creates it
        route_module.ga_segmenter = None

        with patch("src.api.routes.ga_segmentation.GASegmenterService") as MockService:
            mock_instance = MagicMock()
            MockService.return_value = mock_instance

            from src.api.routes.ga_segmentation import get_ga_segmenter
            get_ga_segmenter()

            # Must have been called with use_sam=True as a keyword argument
            MockService.assert_called_once_with(use_sam=True)

    def test_requirements_contains_sam2(self):
        """requirements.txt must declare the SAM-2 dependency."""
        req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(req_path) as f:
            contents = f.read()
        self.assertIn("SAM-2", contents)


if __name__ == "__main__":
    unittest.main()
