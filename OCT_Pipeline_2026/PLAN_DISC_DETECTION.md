# Stage 1: Optic Disc Detection - Implementation Plan

## Overview

Implement the optic disc detection stage to establish a spatial anchor for fovea localization. The disc detector will use en_face images (right half of composites) to predict the optic disc center coordinates.

---

## Current State

| Component | Status |
|-----------|--------|
| Training script (`src/02_train_disc.py`) | Exists - needs updates |
| En face images | 52 images in `data/processed/en_face/` |
| Disc label CSVs | **Missing** - need to create |
| Inference integration | **Missing** - needs implementation |
| Spatial constraints | Utility exists, not integrated |

---

## Implementation Tasks

### Phase 1: Create Label Data

**Task 1.1: Create annotation tool**
- File: `src/utils/annotate_disc.py`
- Simple OpenCV tool to click on disc center in each en_face image
- Controls: click to mark, 's' to save, 'r' to redo, 'n' to skip (not visible), 'q' to quit

**Task 1.2: Annotation data format**
- Output: `data/csv/disc_labels.csv` (all 52 images in one file for K-fold)
- Format: `filename,disc_x,disc_y,visible,laterality`
  - `disc_x`, `disc_y`: Raw pixel coordinates in original image
  - `visible`: 1 if disc is visible, 0 if not (rare but possible)
  - `laterality`: OD (right eye) or OS (left eye) - **critical for spatial constraints**

**Task 1.3: Annotate images**
- Run annotation tool on all 52 en_face images
- Record laterality if known from filename/metadata
- Mark visibility flag for edge cases

### Phase 2: Update Training Script

**Task 2.1: Fix deprecated API**
- File: `src/02_train_disc.py` line 73
- Change: `models.resnet18(pretrained=pretrained)`
- To: `models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)`

**Task 2.2: Implement coordinate-aware augmentation**
- **DO NOT remove RandomHorizontalFlip** - we need augmentation with only 52 images
- Implement custom transform that flips coordinates when image is flipped:
  ```python
  if flipped:
      new_x = image_width - old_x
  ```
- This effectively doubles training data and prevents memorization

**Task 2.3: Use normalized coordinates**
- Coordinates normalized to [0, 1] range (already implemented in existing code)
- This ensures model is robust to input image resizing
- Denormalize only at inference time for visualization

**Task 2.4: Switch to SmoothL1Loss**
- Replace MSELoss with SmoothL1Loss for coordinate regression
- More robust to annotation outliers/noise
- Less sensitive to small errors in disc center labeling

**Task 2.5: Implement K-Fold Cross Validation**
- With only 52 images, standard 80/20 split (10 val images) is statistically fragile
- Use 5-fold cross validation:
  - Train 5 models on different splits
  - Average performance across folds for true accuracy estimate
  - Select best fold or ensemble for final model

**Task 2.6: Add pixel error metric**
- Add Euclidean distance calculation to validation
- Report both loss and pixel error during training
- Track: mean error, std error, max error, % within 10/20/50px

### Phase 3: Integrate into Inference Pipeline

**Task 3.1: Add disc model loading**
- File: `src/05_inference.py`
- Import `DiscDetectorCNN` class
- Add disc model path to `load_models()` function
- Handle case where disc model doesn't exist (graceful fallback)

**Task 3.2: Add disc preprocessing**
- Create `preprocess_for_disc()` function
- Input: BGR image → Output: RGB tensor (224x224, ImageNet normalized)
- Different from fovea/GA which use grayscale 256x256

**Task 3.3: Add disc prediction**
- Create `predict_disc()` function
- Denormalize [0,1] coordinates to original image size
- Return (disc_x, disc_y) in pixel coordinates

**Task 3.4: Integrate laterality-aware spatial constraints**
- **Critical**: Spatial constraint direction depends on eye laterality
  - **Right Eye (OD)**: Disc is nasal (right side), Fovea is temporal (left side)
  - **Left Eye (OS)**: Disc is nasal (left side), Fovea is temporal (right side)
- Options for laterality:
  1. Extract from image filename/metadata if available
  2. Infer from disc-fovea relative positions (less reliable)
  3. Accept as CLI argument `--laterality OD|OS`
  4. Skip directional constraint if unknown (use distance-only constraint)

**Task 3.5: Update visualization**
- Draw disc location on en_face image (blue circle)
- Add disc coordinates to output text
- Show fovea-disc distance
- Indicate laterality if known

**Task 3.6: Update CLI arguments**
- Add `--disc-model` argument (default: `models/disc_detector.pth`)
- Add `--laterality OD|OS` argument (optional)
- Add `--no-constraints` flag to disable spatial constraints

### Phase 4: Testing & Validation

**Task 4.1: K-Fold training**
- Run 5-fold cross validation
- Report per-fold and average metrics
- Expected training time: ~15-20 min per fold on MPS

**Task 4.2: Performance metrics**
- Mean/std pixel error across all folds
- Target: <30px error at 224x224 (good), <50px (acceptable)
- Report % within tolerance (10px, 20px, 50px)

**Task 4.3: Test inference pipeline**
- Run inference on held-out images
- Verify disc detection visualization
- Test spatial constraint effects with known laterality

**Task 4.4: Edge case testing**
- Test on images where disc may be partially visible
- Verify graceful handling when disc model unavailable

---

## File Changes Summary

| File | Action | Changes |
|------|--------|---------|
| `src/utils/annotate_disc.py` | **Create** | Manual annotation tool with visibility/laterality flags |
| `data/csv/disc_labels.csv` | **Create** | All 52 labels (for K-fold splitting) |
| `src/02_train_disc.py` | **Modify** | Fix API, coordinate-aware flip, SmoothL1Loss, K-fold |
| `src/05_inference.py` | **Modify** | Add disc detection, laterality-aware constraints, visualization |

---

## Architecture Decision: ResNet18 (Confirmed)

Keep the existing ResNet18 approach:

**Reasons:**
1. **Appropriate complexity**: Optic disc is a "gross anatomical feature" (large, high contrast) - deeper networks (ResNet50+) would be overkill and harder to train on small data
2. **Speed**: ResNet18 inference ~5ms vs U-Net ~15ms
3. **Model size**: ~45 MB vs ~66 MB for U-Net
4. **Proven**: Works well for single-point regression tasks

---

## Critical Implementation Details

### Coordinate-Aware Horizontal Flip

```python
class CoordinateAwareTransform:
    """Custom transform that flips coordinates with image."""

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, coords):
        # coords: (x, y) normalized to [0, 1]
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            coords = (1.0 - coords[0], coords[1])  # Flip x only
        return image, coords
```

### Laterality-Aware Spatial Constraint

```python
def apply_spatial_constraint(fovea_x, fovea_y, disc_x, disc_y,
                             laterality, min_distance=50):
    """
    Enforce: Fovea must be temporal to disc.

    Args:
        laterality: 'OD' (right eye) or 'OS' (left eye)
    """
    if laterality == 'OD':
        # Right eye: fovea should be LEFT of disc (smaller x)
        if fovea_x > disc_x:
            fovea_x = disc_x - min_distance
    elif laterality == 'OS':
        # Left eye: fovea should be RIGHT of disc (larger x)
        if fovea_x < disc_x:
            fovea_x = disc_x + min_distance

    # Also enforce minimum distance
    distance = np.sqrt((fovea_x - disc_x)**2 + (fovea_y - disc_y)**2)
    if distance < min_distance:
        # Push fovea away from disc
        ...

    return fovea_x, fovea_y
```

---

## Expected Outputs

After implementation:
- `models/disc_detector.pth` - Trained model weights (best fold or ensemble)
- `models/disc_detector_fold{1-5}.pth` - Individual fold models (optional)
- Updated inference with disc visualization and laterality-aware constraints
- Validation report with K-fold performance metrics

---

## Data Requirements Checklist

Before training can begin:
- [ ] All 52 en_face images annotated with disc center (x, y)
- [ ] Visibility flag recorded for each image
- [ ] Laterality (OD/OS) recorded if available from source data
- [ ] Labels saved to `data/csv/disc_labels.csv`
