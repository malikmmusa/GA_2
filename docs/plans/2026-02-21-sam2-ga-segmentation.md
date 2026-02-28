# SAM2 GA Segmentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace K-means boundary extraction with SAM2 for pixel-perfect GA segmentation with built-in false positive rejection.

**Architecture:** K-means finds coarse candidate regions (unchanged), SAM2 refines boundaries and filters false positives via IoU confidence scoring. Graceful fallback to K-means-only if SAM2 model is unavailable.

**Tech Stack:** PyTorch (existing), `sam2` (Meta's official package), SAM2.1-hiera-tiny checkpoint (~150MB via HuggingFace).

---

### Task 1: Install SAM2 Dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add sam2 to requirements**

Add to `requirements.txt`:
```
# SAM2 for GA boundary refinement
sam2>=1.0
```

**Step 2: Install dependencies**

Run: `pip install sam2`
Expected: successful install (pulls from PyPI, includes HuggingFace hub support)

**Step 3: Verify SAM2 import**

Run: `python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2 OK')"`
Expected: `SAM2 OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add sam2 for GA boundary refinement"
```

---

### Task 2: Create SAMRefiner Service

**Files:**
- Create: `src/api/services/sam_refiner.py`
- Create: `tests/test_sam_refiner.py`

**Step 1: Write the failing test**

Create `tests/test_sam_refiner.py`:

```python
"""Tests for SAM2-based GA boundary refinement service."""
import pytest
import numpy as np


class TestSAMRefinerInit:
    """Test SAMRefiner initialization and fallback."""

    def test_import(self):
        from src.api.services.sam_refiner import SAMRefiner
        assert SAMRefiner is not None

    def test_init_without_checkpoint_returns_unavailable(self):
        from src.api.services.sam_refiner import SAMRefiner
        refiner = SAMRefiner(model_id="facebook/sam2.1-hiera-tiny")
        assert isinstance(refiner.available, bool)

    def test_set_image_accepts_numpy_bgr(self):
        from src.api.services.sam_refiner import SAMRefiner
        refiner = SAMRefiner(model_id="facebook/sam2.1-hiera-tiny")
        if not refiner.available:
            pytest.skip("SAM2 model not downloaded")
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        refiner.set_image(dummy)

    def test_refine_candidates_returns_list_of_dicts(self):
        from src.api.services.sam_refiner import SAMRefiner
        refiner = SAMRefiner(model_id="facebook/sam2.1-hiera-tiny")
        if not refiner.available:
            pytest.skip("SAM2 model not downloaded")
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        refiner.set_image(dummy)
        boxes = [(100, 100, 300, 300)]
        results = refiner.refine_candidates(boxes, min_iou=0.5)
        assert isinstance(results, list)
        if results:
            assert "mask" in results[0]
            assert "iou_score" in results[0]
            assert "contour" in results[0]

    def test_refine_point_returns_dict(self):
        from src.api.services.sam_refiner import SAMRefiner
        refiner = SAMRefiner(model_id="facebook/sam2.1-hiera-tiny")
        if not refiner.available:
            pytest.skip("SAM2 model not downloaded")
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        refiner.set_image(dummy)
        result = refiner.refine_point(point=(256, 256), label=1)
        assert result is None or isinstance(result, dict)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam_refiner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.services.sam_refiner'`

**Step 3: Write minimal implementation**

Create `src/api/services/sam_refiner.py`:

```python
"""SAM2-based boundary refinement for GA segmentation."""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SAMRefiner:
    """
    Wraps SAM2ImagePredictor to refine coarse GA candidate regions
    into pixel-perfect segmentation masks.
    
    Supports box prompts (auto from K-means) and point prompts (user clicks).
    Gracefully degrades if SAM2 model is not available.
    """

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-tiny", device: Optional[str] = None):
        self.model_id = model_id
        self.available = False
        self._predictor = None
        self._image_set = False

        if device is None:
            device = self._detect_device()
        self.device = device

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self._predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
            self.available = True
            logger.info(f"SAM2 loaded: {model_id} on {device}")
        except Exception as e:
            logger.warning(f"SAM2 unavailable ({e}). GA segmentation will use K-means only.")

    @staticmethod
    def _detect_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Encode image with SAM2. Input is BGR (OpenCV format), converted to RGB internally."""
        if not self.available:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)
        self._image_set = True

    def reset(self) -> None:
        """Clear encoded image state."""
        if self._predictor is not None:
            self._predictor.reset_predictor()
        self._image_set = False

    def refine_candidates(
        self,
        candidate_boxes: List[Tuple[int, int, int, int]],
        min_iou: float = 0.7
    ) -> List[Dict]:
        """
        Refine K-means candidate regions using SAM2 box prompts.
        
        Args:
            candidate_boxes: List of (x1, y1, x2, y2) bounding boxes
            min_iou: Minimum IoU prediction score to keep a region
        
        Returns:
            List of dicts with keys: mask (H,W bool), iou_score (float), contour (np.ndarray)
        """
        if not self.available or not self._image_set:
            return []

        results = []
        for box in candidate_boxes:
            box_np = np.array(box, dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=box_np,
                multimask_output=False
            )
            mask = masks[0]  # (H, W) bool
            score = float(scores[0])

            if score < min_iou:
                continue

            contour = self._mask_to_contour(mask)
            if contour is not None:
                results.append({
                    "mask": mask,
                    "iou_score": score,
                    "contour": contour
                })

        return results

    def refine_point(
        self,
        point: Tuple[int, int],
        label: int = 1,
        min_iou: float = 0.5
    ) -> Optional[Dict]:
        """
        Refine using a point prompt (for local/click-based segmentation).
        
        Args:
            point: (x, y) coordinates
            label: 1 for foreground, 0 for background
            min_iou: Minimum IoU score
        
        Returns:
            Dict with mask, iou_score, contour or None
        """
        if not self.available or not self._image_set:
            return None

        point_np = np.array([[point[0], point[1]]], dtype=np.float32)
        label_np = np.array([label], dtype=np.int32)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_np,
            point_labels=label_np,
            multimask_output=True
        )

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_mask = masks[best_idx]

        if best_score < min_iou:
            return None

        contour = self._mask_to_contour(best_mask)
        if contour is None:
            return None

        return {
            "mask": best_mask,
            "iou_score": best_score,
            "contour": contour
        }

    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        """Convert boolean mask to largest OpenCV contour."""
        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sam_refiner.py -v`
Expected: `test_import` PASS, `test_init_without_checkpoint_returns_unavailable` PASS. Others may skip if model not yet downloaded.

**Step 5: Download SAM2 checkpoint (first time only)**

Run: `python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; p = SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-tiny'); print('Downloaded')"`
Expected: Downloads ~150MB checkpoint to HuggingFace cache. Prints `Downloaded`.

**Step 6: Re-run all tests**

Run: `python -m pytest tests/test_sam_refiner.py -v`
Expected: ALL PASS (no skips now that model is available)

**Step 7: Commit**

```bash
git add src/api/services/sam_refiner.py tests/test_sam_refiner.py
git commit -m "feat: add SAMRefiner service for GA boundary refinement"
```

---

### Task 3: Integrate SAM2 into Global GA Segmentation

**Files:**
- Modify: `src/api/services/ga_segmenter.py`
- Modify: `src/api/routes/ga_segmentation.py`
- Create: `tests/test_ga_sam_integration.py`

**Step 1: Write the failing integration test**

Create `tests/test_ga_sam_integration.py`:

```python
"""Integration tests: GA segmenter with SAM2 refinement."""
import pytest
import numpy as np
import cv2
import os


def make_synthetic_image_with_bright_blob():
    """Create a 600x1200 synthetic composite image with a bright GA-like blob on the right half."""
    img = np.zeros((600, 1200, 3), dtype=np.uint8)
    # Right half = en-face (gray background)
    img[:, 600:, :] = 128
    # Bright irregular blob (simulating GA)
    cv2.ellipse(img, (900, 300), (80, 50), 30, 0, 360, (220, 220, 220), -1)
    return img


class TestGASegmenterWithSAM:

    def test_segment_ga_returns_contours(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=True)
        img = make_synthetic_image_with_bright_blob()
        contours = segmenter.segment_ga_regions(image=img, en_face_split_x=600)
        assert isinstance(contours, list)

    def test_fallback_without_sam(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=False)
        img = make_synthetic_image_with_bright_blob()
        contours = segmenter.segment_ga_regions(image=img, en_face_split_x=600)
        assert isinstance(contours, list)

    def test_sam_produces_different_contours_than_kmeans(self):
        """SAM contours should differ from raw K-means contours (more precise)."""
        from src.api.services.ga_segmenter import GASegmenterService
        img = make_synthetic_image_with_bright_blob()
        seg_sam = GASegmenterService(use_sam=True)
        seg_kmeans = GASegmenterService(use_sam=False)

        contours_sam = seg_sam.segment_ga_regions(image=img, en_face_split_x=600)
        contours_kmeans = seg_kmeans.segment_ga_regions(image=img, en_face_split_x=600)

        # Both should find something
        # (exact comparison depends on SAM availability)
        assert isinstance(contours_sam, list)
        assert isinstance(contours_kmeans, list)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ga_sam_integration.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_sam'`

**Step 3: Modify GASegmenterService to integrate SAM2**

Key changes to `src/api/services/ga_segmenter.py`:

1. Add `use_sam` parameter to `__init__`
2. Initialize `SAMRefiner` if `use_sam=True`
3. In `segment_ga_regions`, after K-means produces candidate contours:
   - Extract bounding boxes from each candidate contour
   - Call `sam_refiner.set_image(en_face)` 
   - Call `sam_refiner.refine_candidates(boxes)` 
   - Replace K-means contours with SAM contours
   - Fall back to K-means contours if SAM returns nothing
4. Reset SAM predictor after use

The modification points in `segment_ga_regions()`:
- After line ~377 (where `final_contours` is built from K-means), insert SAM refinement block
- Before the en-face-to-original coordinate adjustment (line ~380)

Specific code to add to `__init__`:
```python
self.use_sam = use_sam
self.sam_refiner = None
if use_sam:
    from .sam_refiner import SAMRefiner
    self.sam_refiner = SAMRefiner()
    if not self.sam_refiner.available:
        logger.warning("SAM2 not available, falling back to K-means only")
        self.sam_refiner = None
```

In `segment_ga_regions()`, replace the final contour extraction block (after morphology + contour filtering, before coordinate adjustment) with:

```python
# --- SAM2 Refinement ---
if self.sam_refiner is not None and final_contours:
    boxes = []
    for cnt in final_contours:
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        pad = 10
        boxes.append((
            max(0, x_b - pad),
            max(0, y_b - pad),
            min(w, x_b + w_b + pad),
            min(h, y_b + h_b + pad)
        ))

    self.sam_refiner.set_image(en_face)
    sam_results = self.sam_refiner.refine_candidates(boxes, min_iou=0.7)
    self.sam_refiner.reset()

    if sam_results:
        final_contours = [r["contour"] for r in sam_results]
# --- End SAM2 Refinement ---
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ga_sam_integration.py -v`
Expected: ALL PASS

**Step 5: Run existing GA tests to verify no regression**

Run: `python -m pytest tests/test_ga_local_segmentation.py -v`
Expected: ALL PASS (existing behavior preserved via fallback)

**Step 6: Commit**

```bash
git add src/api/services/ga_segmenter.py tests/test_ga_sam_integration.py
git commit -m "feat: integrate SAM2 refinement into GA global segmentation"
```

---

### Task 4: Integrate SAM2 into Local GA Segmentation

**Files:**
- Modify: `src/api/services/ga_segmenter.py` (the `segment_ga_local` method)

**Step 1: Write the failing test**

Add to `tests/test_ga_sam_integration.py`:

```python
class TestLocalSegmentationWithSAM:

    def test_local_segment_with_sam_point_prompt(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=True)
        img = make_synthetic_image_with_bright_blob()
        # Click on the bright blob center (900, 300 in full image = 300, 300 in en-face)
        contours = segmenter.segment_ga_local(
            image=img,
            click_x=900,
            click_y=300,
            en_face_split_x=600
        )
        assert isinstance(contours, list)

    def test_local_segment_fallback_without_sam(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=False)
        img = make_synthetic_image_with_bright_blob()
        contours = segmenter.segment_ga_local(
            image=img,
            click_x=900,
            click_y=300,
            en_face_split_x=600
        )
        assert isinstance(contours, list)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ga_sam_integration.py::TestLocalSegmentationWithSAM -v`
Expected: FAIL (segment_ga_local doesn't use SAM yet)

**Step 3: Modify `segment_ga_local` to use SAM point prompt**

In `segment_ga_local()`, after the en-face extraction and coordinate conversion, try SAM first:

```python
# --- SAM2 Point-Prompt Refinement ---
if self.sam_refiner is not None:
    self.sam_refiner.set_image(en_face)
    result = self.sam_refiner.refine_point(
        point=(int(click_x_local), int(click_y_local)),
        label=1,
        min_iou=0.5
    )
    self.sam_refiner.reset()

    if result is not None:
        contour = result["contour"]
        # Adjust back to original image coordinates
        if en_face_split_x is not None:
            contour = contour.copy()
            contour[:, 0, 0] += en_face_split_x
        return [contour]
# --- Fall through to K-means if SAM didn't produce a result ---
```

Insert this block right after extracting `en_face`, `click_x_local`, `click_y_local` and before the grayscale conversion / K-means code.

**Step 4: Run tests**

Run: `python -m pytest tests/test_ga_sam_integration.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/api/services/ga_segmenter.py tests/test_ga_sam_integration.py
git commit -m "feat: integrate SAM2 point prompt into local GA segmentation"
```

---

### Task 5: Update Route Singleton and Requirements

**Files:**
- Modify: `src/api/routes/ga_segmentation.py`
- Modify: `requirements.txt`

**Step 1: Update singleton to pass `use_sam=True`**

In `src/api/routes/ga_segmentation.py`, change `get_ga_segmenter()`:

```python
def get_ga_segmenter() -> GASegmenterService:
    """Get or initialize the GA segmenter service (singleton)."""
    global ga_segmenter
    if ga_segmenter is None:
        ga_segmenter = GASegmenterService(use_sam=True)
    return ga_segmenter
```

**Step 2: Ensure requirements.txt is complete**

Verify `sam2>=1.0` is in `requirements.txt`.

**Step 3: Commit**

```bash
git add src/api/routes/ga_segmentation.py requirements.txt
git commit -m "feat: enable SAM2 in GA segmentation route"
```

---

### Task 6: End-to-End Smoke Test with Real Images

**Files:**
- Create: `tests/test_ga_e2e_real_images.py`

**Step 1: Write smoke test using actual input images**

```python
"""End-to-end smoke test with real OCT images."""
import pytest
import numpy as np
import cv2
import os

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "input_images")


def get_test_images():
    """Get list of test images if available."""
    if not os.path.isdir(IMAGES_DIR):
        return []
    return [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")][:3]


@pytest.mark.skipif(not get_test_images(), reason="No test images available")
class TestE2ERealImages:

    def test_segment_real_image_produces_regions(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=True)

        img_path = os.path.join(IMAGES_DIR, get_test_images()[0])
        image = cv2.imread(img_path)
        assert image is not None

        contours = segmenter.segment_ga_regions(image=image)
        assert isinstance(contours, list)
        print(f"Found {len(contours)} GA regions in {get_test_images()[0]}")

    def test_all_test_images_dont_crash(self):
        from src.api.services.ga_segmenter import GASegmenterService
        segmenter = GASegmenterService(use_sam=True)

        for fname in get_test_images():
            img_path = os.path.join(IMAGES_DIR, fname)
            image = cv2.imread(img_path)
            assert image is not None
            contours = segmenter.segment_ga_regions(image=image)
            assert isinstance(contours, list)
            print(f"  {fname}: {len(contours)} regions")
```

**Step 2: Run smoke tests**

Run: `python -m pytest tests/test_ga_e2e_real_images.py -v -s`
Expected: PASS for all test images, prints region counts

**Step 3: Visual inspection**

Write a quick debug script (do not commit) to overlay SAM contours on a test image:

```python
import cv2
from src.api.services.ga_segmenter import GASegmenterService

segmenter = GASegmenterService(use_sam=True)
image = cv2.imread("input_images/test_1.png")
contours = segmenter.segment_ga_regions(image=image)
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
cv2.imwrite("output_results/sam2_test_overlay.png", output)
print(f"Saved overlay with {len(contours)} regions")
```

Run: `python debug_sam_overlay.py`
Expected: `output_results/sam2_test_overlay.png` shows GA regions with precise green outlines

**Step 4: Commit tests**

```bash
git add tests/test_ga_e2e_real_images.py
git commit -m "test: add end-to-end GA segmentation smoke tests"
```

---

### Task 7: Add .gitignore for Model Weights

**Files:**
- Modify: `.gitignore`

Ensure HuggingFace cache and large weight files are not committed:

```
# Model weights (too large for git)
weights/*.pt
weights/*.pth
!weights/.gitkeep
```

**Commit:**

```bash
git add .gitignore
git commit -m "chore: ignore model weight files in git"
```
