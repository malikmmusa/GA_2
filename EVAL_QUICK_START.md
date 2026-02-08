# GA Segmentation Evaluation - Quick Start Guide

## Overview

The evaluation harness (`src/eval_ga_segmentation.py`) provides automated, quantitative testing of the GA segmentation algorithm using 52 ground-truth annotated images.

## Usage

### Basic Evaluation

```bash
# Activate virtual environment
cd /Users/musamalik/OCT_Project
source venv/bin/activate  # or: ./venv/bin/activate

# Run evaluation (saves debug images to evaluation_debug/)
python src/eval_ga_segmentation.py
```

### Options

```bash
# Custom debug directory
python src/eval_ga_segmentation.py --debug-dir my_evaluation

# Skip saving debug visualizations (faster)
python src/eval_ga_segmentation.py --no-debug

# Quiet mode (only show summary)
python src/eval_ga_segmentation.py --quiet

# Custom input/marked directories
python src/eval_ga_segmentation.py --input-dir custom_input --marked-dir custom_marked
```

## Output

### Console Output

Per-image results with status indicators:
- ✅ Hit within 20px
- ⚠️  Hit but >20px away
- ❌ Miss (no regions detected)

### Summary Metrics

- **Total/Valid/Detected images**: Dataset size and detection coverage
- **Miss rate**: % of images with no detections
- **Hit rates**: % of images where GT point is within 10/20/50 pixels of a detected region
- **Distance statistics**: Mean, median, std, min, max distance from GT to nearest contour

### Debug Visualizations

Saved to `evaluation_debug/` (or custom `--debug-dir`):

Each image shows:
- **Yellow contours**: Detected GA regions
- **Red circle**: Ground truth GA boundary point (from raw_marked peach line)
- **Green circle**: Nearest point on detected contours
- **Purple line**: Connecting line showing distance
- **Text overlay**: Distance in pixels (or "MISS" if no detection)

## Ground Truth Data

The `raw_marked/` folder contains 52 annotated images with:
- **Red dot**: Fovea location
- **Peach/yellow line** (#F4C5AD): Connects fovea to nearest GA boundary
- **Far endpoint of line**: The ground truth GA boundary point used for evaluation

If no line is present, it means the GA has engulfed the fovea.

## Interpreting Results

### Good Performance Indicators:
- Hit rate (≤20px) > 80%
- Mean distance < 30px
- Miss rate < 5%

### Current Baseline (Original Algorithm):
- Hit rate (≤20px): 3.8%
- Mean distance: 464.7px
- Miss rate: 0%

### Current Improved (Latest Version):
- Hit rate (≤20px): 3.8%
- Mean distance: 426.0px
- Miss rate: 0%

## Iterative Improvement Workflow

1. **Modify** `src/api/services/ga_segmenter.py`
2. **Run evaluation**: `python src/eval_ga_segmentation.py --debug-dir eval_v2 --quiet`
3. **Compare metrics** with previous runs
4. **Inspect debug images** for failure modes
5. **Repeat** until metrics improve

## Quick Test on Single Image

```python
from src.api.services.ga_segmenter import GASegmenterService
from src.utils.image_utils import get_split_indices_and_images
import cv2

img = cv2.imread('input_images/21896237.png')
b_scan, en_face, metadata = get_split_indices_and_images(img)
segmenter = GASegmenterService()
contours = segmenter.segment_ga_regions(img, en_face_split_x=metadata['final_split_column'])
print(f'Found {len(contours)} regions')
```

## Parameter Tuning Guide

Key parameters in `GASegmenterService.__init__()`:

```python
n_clusters=5                    # Number of K-means clusters (3-7 typical)
min_area=500                    # Minimum contour area in pixels
max_circularity=0.8            # Reject circular objects (0.7-0.9)
relative_area_threshold=0.15   # Keep regions >= this fraction of largest
max_regions=None               # Maximum regions to return (None = no limit)
disc_exclusion_multiplier=0.75 # Disc masking radius (0.6-0.9)
clahe_clip_limit=2.0           # CLAHE contrast (1.0-3.0)
```

Additional tuning in `segment_ga_regions()`:
- `ga_threshold = 0.3` (line ~285): GA likelihood threshold (0.3-0.7)
- Cluster selection logic: Currently selects all clusters above threshold

## Common Issues & Fixes

### Issue: Too many false positives
**Fix**: Increase `ga_threshold` from 0.3 to 0.5-0.6

### Issue: Over-merged blobs
**Fix**: 
- Reduce morphological kernel from 9 to 7
- Lower watershed activation threshold
- Increase `relative_area_threshold`

### Issue: Missing GA regions
**Fix**:
- Decrease `ga_threshold`
- Decrease `min_area`
- Increase `n_clusters` to 7

### Issue: Disc artifacts included
**Fix**: Increase `disc_exclusion_multiplier` from 0.75 to 0.85

## Files

- **Evaluation script**: `src/eval_ga_segmentation.py`
- **Segmenter service**: `src/api/services/ga_segmenter.py`
- **Ground truth images**: `raw_marked/*.png`
- **Test images**: `input_images/*.png`
- **Results summary**: `GA_SEGMENTATION_IMPROVEMENTS.md`
