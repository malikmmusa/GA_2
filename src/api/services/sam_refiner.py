import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch  # required for device detection (mps/cuda) and SAM2 internals

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Return the largest OpenCV contour (N,1,2) from a bool mask, or empty array."""
    uint8 = mask_2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 1, 2), dtype=np.int32)
    return max(contours, key=cv2.contourArea)


class SAMRefiner:
    def __init__(
        self,
        checkpoint_path: str = "weights/sam2.1_hiera_tiny.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    ):
        self.available = False
        self.predictor = None

        if not os.path.exists(checkpoint_path):
            logger.warning("SAM2 checkpoint not found at %s — SAMRefiner unavailable", checkpoint_path)
            return

        try:
            device = _select_device()
            model = build_sam2(model_cfg, checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.available = True
        except Exception as exc:
            logger.warning("SAM2 failed to load (%s) — SAMRefiner unavailable", exc)

    def set_image(self, image: np.ndarray) -> None:
        if not self.available:
            raise RuntimeError("SAMRefiner is not available — checkpoint missing or failed to load")
        self.predictor.set_image(image)

    def refine_candidates(
        self,
        boxes: List[np.ndarray],
        min_iou: float = 0.7,
    ) -> List[dict]:
        if not self.available:
            return []

        results = []
        for box in boxes:
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
            if len(masks) == 0:
                continue
            iou = float(scores[0])
            if iou < min_iou:
                continue
            mask_2d = masks[0] > 0.5
            results.append({"mask": mask_2d, "iou": iou, "contour": _extract_contour(mask_2d)})

        return results

    def refine_point(
        self,
        point: Tuple[int, int],
        labels: Optional[List[int]] = None,
    ) -> Optional[dict]:
        if not self.available:
            return None

        if labels is None:
            labels = [1]

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array(labels),
            multimask_output=False,
        )
        if len(masks) == 0:
            return None
        iou = float(scores[0])
        mask_2d = masks[0] > 0.5
        return {"mask": mask_2d, "iou": iou, "contour": _extract_contour(mask_2d)}
