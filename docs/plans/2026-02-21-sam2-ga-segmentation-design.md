# SAM2-Based GA Segmentation Design

## Problem

The current K-means clustering approach for Geographic Atrophy (GA) detection on NIR en-face OCT images has two critical limitations:

1. **False positives** — K-means highlights non-GA bright regions (peripapillary atrophy, drusen, artifacts)
2. **Imprecise boundaries** — Cluster boundaries are rough, not following true lesion edges

## Constraints

- No annotated training data available
- ~55 Heidelberg Spectralis composite images (B-scan + NIR en-face)
- Fully automatic detection (no user clicks for global segmentation)
- Must preserve existing API contract (`segment_ga_regions()` returns `List[np.ndarray]`)

## Solution: SAM2 with Auto-Prompting

Replace the boundary extraction step with Meta's SAM2 (Segment Anything Model 2) while keeping K-means as the coarse candidate detector.

### Pipeline

```
1. Extract en-face region + CLAHE enhancement (unchanged)
2. K-means clustering, 3 clusters (unchanged)
3. Select brightest cluster as candidate mask (unchanged)
4. Extract bounding boxes from K-means candidate contours (NEW)
5. SAM2 encoder: embed full en-face image once (~200ms) (NEW)
6. For each candidate box: SAM2 decoder predicts mask + IoU score (~20ms each) (NEW)
7. Filter by IoU score > 0.7 (kills false positives) (NEW)
8. Apply anatomical scoring (unchanged)
9. Return final contours
```

### Why SAM2

- Zero-shot: no training data needed
- Pixel-perfect boundaries: trained on 11M images / 1.1B masks
- Built-in confidence: IoU prediction score filters false positives
- Lightweight: SAM2-tiny is 38.9M params, ~150MB checkpoint
- Compatible: runs on MPS (Apple Silicon), CUDA, or CPU

### File Changes

**New file: `src/api/services/sam_refiner.py`**

SAM2 wrapper class:
- `__init__`: loads SAM2-tiny checkpoint, auto-detects device (MPS/CUDA/CPU)
- `set_image(image)`: runs SAM2 image encoder once per image
- `refine_candidates(boxes, min_iou)`: runs decoder per bounding box, returns masks + scores + contours
- `refine_point(point, labels)`: runs decoder with point prompt for local segmentation

**Modified file: `src/api/services/ga_segmenter.py`**

- Add `use_sam: bool = True` parameter
- After K-means candidate extraction, call `SAMRefiner.refine_candidates()`
- Extract contours from SAM masks instead of K-means masks
- Graceful fallback: if SAM unavailable, use current K-means contours
- Local segmentation: use SAM with point prompt (user click coordinates)

**New weight file: `weights/sam2.1_hiera_tiny.pt`** (~150MB)

Downloaded via `sam2` package or direct URL.

### Dependencies

- `sam2` (Meta's official SAM2 package)
- PyTorch (already installed for disc detection)

### Device Strategy

- Apple Silicon Mac: MPS acceleration
- NVIDIA GPU: CUDA
- Fallback: CPU

### Local Segmentation (`segment_ga_local`)

SAM2 with point prompt instead of box prompt:
- User click provides (x, y) point prompt
- SAM2 produces precise mask around clicked region
- No K-means needed for local mode

### Fallback Behavior

If SAM2 checkpoint is missing or fails to load:
- Log warning
- Fall back to current K-means-only pipeline
- No disruption to existing functionality

### Performance

- SAM2 encoder: ~200ms per image (run once)
- SAM2 decoder: ~20ms per candidate box
- Typical image (3-5 candidates): ~300-500ms total
- Current K-means-only: ~100ms
- Acceptable for clinical tool (not real-time, interactive)

### Success Criteria

1. Boundary precision: contours follow true GA lesion edges (visual inspection)
2. False positive rate: fewer non-GA regions highlighted
3. Graceful fallback: system works if SAM2 model not available
4. API compatibility: frontend unchanged
