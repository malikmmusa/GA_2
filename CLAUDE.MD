# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OCT Analysis Pipeline** - A modular computer vision pipeline for identifying anatomical landmarks and pathology in Optical Coherence Tomography (OCT) retinal scans.

- **Domain:** Ophthalmology / Retinal Imaging
- **User:** Physician & Developer (requires expert clinical context)
- **Status:** Production-ready with trained models (January 2026)
- **Project Location:** `/Users/musamalik/OCT_Project/OCT_Pipeline_2026/`

## Original Project Specification

**Goal:** Build a modular computer vision pipeline to identify anatomical landmarks and pathology in OCT scans, prioritizing clean, sequential architecture over complex end-to-end models.

### Input Data Format
- **Format:** Single JPG composites containing paired images side-by-side
  - **Left Half:** B-Scan (cross-sectional retinal view)
  - **Right Half:** En Face (top-down retinal map)
- **Labels:** CSV files with (x, y) pixel coordinates for three targets:
  1. Optic Disc (spatial anchor)
  2. Fovea (functional landmark)
  3. Geographic Atrophy / GA (pathology)

### Three-Stage Pipeline Architecture

#### Stage 0: Preprocessing (01_split_data.py)
- **Three-Strategy Intelligent Splitting:**
  1. **Dividing Line Detection:** Detects and removes colored vertical separators (e.g., purple/blue lines)
  2. **Content Boundary Detection:** Analyzes column-to-column content changes to find where B-scan ends and en face begins
  3. **Fallback 50% Split:** Only used if both detection methods fail (with warning)
- **Why Multiple Strategies:** B-scan and en face images may be different sizes - cannot assume 50% split
- Save left half → `data/processed/b_scans/`
- Save right half → `data/processed/en_face/`
- Maintain filename consistency across both datasets
- **Critical:** Dividing lines must be removed to prevent models from learning them as features

#### Stage 1: Optic Disc Detection (The Anchor)
- **Input:** En Face image (right half)
- **Model:** ResNet18 with regression head
- **Output:** (x, y) coordinates of optic disc center
- **Purpose:** Establishes spatial anchor - fovea is always temporal to disc
- **Status:** ⏸️ NOT TRAINED (no disc labels available)

#### Stage 2: Fovea Localization (The Landmark)
- **Input:** B-Scan image (left half) OR dual-stream (B-scan + En face)
- **Model:** U-Net with Gaussian Heatmap Regression
- **Training:** Point labels → 2D Gaussian heatmaps (σ=15px)
- **Loss:** MSE between predicted and target heatmaps
- **Inference:** argmax of predicted heatmap → (x, y) coordinates
- **Constraint:** Search space can be restricted to temporal side of disc
- **Status:** ✅ TRAINED (`models/fovea_detector.pth`, 66 MB, val loss: 0.260)

#### Stage 3: Geographic Atrophy Segmentation (The Pathology)
- **Input:** En Face image (right half)
- **Model:** U-Net for semantic segmentation
- **Challenge:** Labels are sparse points, but target is a region
- **Training Strategy (Weak Supervision):**
  - Generate "proxy masks" by creating Gaussian blobs (σ=50px, threshold=0.3) around labeled GA points
  - Model learns to identify texture of hyper-transmission defects (bright regions)
- **Loss:** Combined BCE (50%) + Dice (50%)
- **Output:** Probability map or binary mask of GA lesion
- **Status:** ✅ TRAINED (`models/ga_segmenter.pth`, 66 MB, Dice: 0.353)

## Current Implementation Status

### Completed Components
- ✅ Image preprocessing and splitting (52 images processed)
- ✅ Four U-Net architecture variants (`src/models/unet.py`)
- ✅ Gaussian utility functions for weak supervision (`src/utils/gaussian_utils.py`)
- ✅ Fovea detector training pipeline and trained model
- ✅ GA segmentation training pipeline and trained model
- ✅ End-to-end inference pipeline (`src/05_inference.py`)
- ✅ Comprehensive documentation suite

### Not Implemented
- ⏸️ Optic disc detection (Stage 1) - awaiting disc labels

### Dataset
- **Raw Data:** 52 composite images in `data/raw/`
- **Processed:** 52 B-scans + 52 en face images
- **Labels:** 42 training + 10 validation samples for fovea and GA
- **Missing:** No optic disc labels

## Technical Architecture

### Model Architectures (`src/models/unet.py`)

All models use U-Net architecture with the following structure:
- **Encoder:** 64 → 128 → 256 → 512 → 512 (with downsampling)
- **Decoder:** 512 → 256 → 128 → 64 → output (with upsampling + skip connections)
- **Building Blocks:** DoubleConv (Conv2d → BatchNorm → ReLU × 2)
- **Upsampling:** Bilinear by default (configurable to transposed convolution)

**Four Variants:**

1. **`UNet`** - Base architecture with flexible input/output channels
2. **`UNetWithSigmoid`** - For fovea heatmap regression (outputs values in [0,1])
3. **`UNetForSegmentation`** - For GA binary segmentation
4. **`DualStreamUNet`** - Concatenates B-scan + en face inputs (not used in current training)

### Weak Supervision Strategy

**The Key Innovation:** Converting sparse point annotations into training targets for spatial prediction tasks.

**For Fovea (Stage 2):**
```python
# Point (x, y) → 2D Gaussian heatmap with σ=15px
heatmap = generate_gaussian_heatmap(x, y, height, width, sigma=15)
# Small sigma for precise localization
# Model predicts heatmap, argmax gives coordinates
```

**For GA (Stage 3):**
```python
# Point (x, y) → Gaussian blob with σ=50px → binary mask at threshold=0.3
mask = generate_gaussian_mask(x, y, height, width, sigma=50, threshold=0.3)
# Large sigma creates broader region for lesion detection
# Model learns texture patterns within this region
```

**Why This Works:**
- Smooth targets prevent overfitting to exact pixel locations
- Provides gradient signal in neighborhood around target
- σ parameter controls precision vs. recall trade-off
- Enables training segmentation models without full mask annotations

### Utility Functions (`src/utils/gaussian_utils.py`)

**Core Functions:**
- `generate_gaussian_heatmap(center_x, center_y, height, width, sigma)` - Creates 2D Gaussian for heatmap regression
- `generate_gaussian_mask(center_x, center_y, height, width, sigma, threshold)` - Creates binary mask from Gaussian
- `heatmap_to_coordinates(heatmap)` - Extracts (x, y) via argmax
- `refine_coordinates_weighted(heatmap, x, y, window_size)` - Sub-pixel accuracy via weighted averaging
- `apply_spatial_constraint(fovea_x, fovea_y, disc_x, disc_y, min_distance)` - Enforces anatomical constraints
- `visualize_heatmap_overlay(image, heatmap, alpha)` - Debug visualization with JET colormap

## Clinical Context and Anatomical Constraints

### Anatomical Relationships (Critical for Model Design)
- **Optic Disc:** Bright circular region where optic nerve enters retina (nasal side)
- **Fovea:** Central pit of macula, responsible for sharp vision (temporal side)
- **Spatial Constraint:** Fovea is ALWAYS temporal to optic disc
- **Typical Distance:** 4-5mm between disc and fovea (~200-300 pixels)

### Geographic Atrophy (GA)
- Advanced form of age-related macular degeneration (dry AMD)
- Appears as bright hyper-transmission defects in en face view
- Indicates loss of retinal pigment epithelium (RPE) and photoreceptors
- **Clinical Significance:** Distance from fovea to nearest GA edge predicts vision loss

### Clinical Workflow (What the Pipeline Mimics)
1. Identify optic disc → provides orientation and spatial reference
2. Locate fovea → central landmark for measuring disease proximity
3. Segment GA regions → measure extent and distance from fovea

## Essential Commands

### Environment Setup
```bash
cd OCT_Pipeline_2026
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python verify_setup.py  # Verifies PyTorch, MPS/CUDA, dependencies
```

### Data Preprocessing
```bash
# Split composite images (run once on new data)
python src/01_split_data.py
# Input: data/raw/*.jpg
# Output: data/processed/b_scans/*.jpg and data/processed/en_face/*.jpg

# Three-strategy splitting:
# 1. Detects and removes colored dividing lines (variance analysis)
# 2. Detects content boundaries (handles different-sized images)
# 3. Falls back to 50% only if detection fails (with warning)
```

### Training Models

**Train Fovea Detector (Stage 2):**
```bash
python src/03_train_fovea.py
# Requires: data/csv/train_fovea_labels.csv, data/csv/val_fovea_labels.csv
# CSV format: filename,fovea_x,fovea_y
# Outputs: models/fovea_detector.pth
# Training time: ~15-18 minutes (20 epochs on MPS)
```

**Train GA Segmenter (Stage 3):**
```bash
python src/04_train_ga.py
# Requires: data/csv/train_ga_labels.csv, data/csv/val_ga_labels.csv
# CSV format: filename,ga_x,ga_y
# Outputs: models/ga_segmenter.pth
# Training time: ~15-18 minutes (20 epochs on MPS)
```

**Train Optic Disc Detector (Stage 1 - when labels available):**
```bash
python src/02_train_disc.py
# Requires: data/csv/train_disc_labels.csv, data/csv/val_disc_labels.csv
# CSV format: filename,disc_x,disc_y
# Outputs: models/disc_detector.pth
```

### Inference
```bash
# Run end-to-end inference on new composite image
python src/05_inference.py /path/to/composite_image.jpg

# Outputs (in data/inference_results/):
# - {name}_result.png - Side-by-side visualization with predictions
# - {name}_fovea_heatmap.png - Raw fovea heatmap
# - {name}_ga_mask.png - Binary GA segmentation
# - {name}_ga_prob.png - GA probability map
```

## Training Configuration

### Hyperparameters (Current)
- **Epochs:** 20 (quick training for testing)
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Optimizer:** Adam with ReduceLROnPlateau (patience=5, factor=0.5)
- **Image Size:** 256×256 (resized during training)
- **Device:** MPS (Apple Silicon M4) with automatic fallback to CUDA/CPU

### For Production Training
- Increase epochs to 100+ for better convergence
- Current 20-epoch models:
  - Fovea: ~117px error on 256×256 images (val loss: 0.260)
  - GA: Dice coefficient 0.353
- Expected with 100 epochs:
  - Fovea: ~10-20px error (sub-pixel with refinement)
  - GA: Dice > 0.6 (potentially > 0.75)

### Data Augmentation (Configured)
- Horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness/contrast ±0.2)

## Data Format Specifications

### Input Image Structure
```
Composite Image (side-by-side):
┌──────────────┬──────────────┐
│              │              │
│   B-Scan     │   En Face    │
│   (Left)     │   (Right)    │
│  Cross-sec   │   Top-down   │
│              │              │
└──────────────┴──────────────┘
```

### Label CSV Format

**Fovea Labels:**
```csv
filename,fovea_x,fovea_y
patient001.jpg,723,401
patient002.jpg,698,388
```

**GA Labels:**
```csv
filename,ga_x,ga_y
patient001.jpg,645,425
patient002.jpg,623,412
```

**Optic Disc Labels (when available):**
```csv
filename,disc_x,disc_y
patient001.jpg,512,384
patient002.jpg,498,392
```

**Coordinate System:**
- Origin (0, 0) at top-left
- X increases rightward
- Y increases downward
- Coordinates in original image pixel space (before 256×256 resize)

## Key Design Decisions

### Why Separate Stages Instead of End-to-End?
1. **Modularity:** Each stage can be trained, debugged, and evaluated independently
2. **Clinical Workflow:** Mimics how ophthalmologists actually analyze OCT scans
3. **Anatomical Constraints:** Can use disc location to constrain fovea search space
4. **Extensibility:** Easy to add new stages or swap architectures
5. **Data Efficiency:** Can train stages with different amounts of labeled data

### Why Different Gaussian Sigmas?
- **Fovea (σ=15px):** Small anatomical point requiring precise localization
- **GA (σ=50px):** Lesion region requiring broader coverage
- Empirically tuned for best performance on this dataset

### Why Heatmap Regression Instead of Direct Coordinate Regression?
- Heatmaps provide spatial context and gradient signal around target
- More robust to annotation noise than direct (x, y) regression
- Enables sub-pixel accuracy with weighted refinement
- Better generalization with limited training data

### Why U-Net for Both Tasks?
- Standard architecture for medical image analysis
- Skip connections preserve spatial information critical for localization
- Works well with limited data (52 training images)
- Proven effective for both regression and segmentation tasks

## GPU Acceleration

### Apple Silicon (MPS)
The codebase is optimized for Apple Silicon M-series chips:
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # 40-45 seconds per epoch (vs. minutes on CPU)
```

### Performance
- **MPS (M4 Mac):** ~40-45 seconds/epoch
- **CPU Fallback:** ~3-5 minutes/epoch
- **Full Training (20 epochs):** ~15-18 minutes on MPS

## Project Structure

```
OCT_Pipeline_2026/
├── data/
│   ├── raw/                    # Place composite JPG images here
│   ├── raw_marked/             # Manually annotated images for reference
│   ├── processed/
│   │   ├── b_scans/           # Auto-generated (Stage 0)
│   │   └── en_face/           # Auto-generated (Stage 0)
│   ├── csv/                   # Label files (see format above)
│   └── inference_results/     # Inference outputs
│
├── src/
│   ├── 00_extract_vector_labels.py  # Label extraction utility
│   ├── 01_split_data.py            # Stage 0: Preprocessing
│   ├── 02_train_disc.py            # Stage 1: Disc training
│   ├── 03_train_fovea.py           # Stage 2: Fovea training
│   ├── 04_train_ga.py              # Stage 3: GA training
│   ├── 05_inference.py             # End-to-end inference
│   ├── models/
│   │   └── unet.py                 # All U-Net variants
│   └── utils/
│       └── gaussian_utils.py       # Weak supervision utilities
│
├── models/                     # Trained model files (.pth)
│   ├── fovea_detector.pth     # 66 MB
│   └── ga_segmenter.pth       # 66 MB
│
├── README.md                   # User guide
├── PROJECT_SPECIFICATION.md    # Technical specifications
├── TRAINING_COMPLETE.md        # Training results
├── SUMMARY.md                  # Project overview
└── requirements.txt            # Python dependencies
```

## Working with This Codebase

### Before Modifying Training Code
1. Read existing dataset classes to understand data loading pipeline
2. Check `generate_gaussian_heatmap()` and `generate_gaussian_mask()` - these are critical for weak supervision
3. Understand that coordinates are scaled when images are resized to 256×256
4. Note that CSV filenames must match processed image filenames exactly

### Common Modifications

**Adjust Gaussian Parameters:**
```python
# In src/utils/gaussian_utils.py
sigma_fovea = 15  # Decrease for sharper, increase for broader
sigma_ga = 50     # Adjust based on typical lesion size
```

**Change Training Image Size:**
```python
# In dataset classes (03_train_fovea.py, 04_train_ga.py)
output_size = (256, 256)  # (width, height)
# Note: Coordinates must be scaled accordingly
```

**Modify Hyperparameters:**
```python
# In training scripts
num_epochs = 100      # More epochs for better convergence
batch_size = 8        # Decrease if GPU memory issues
learning_rate = 1e-4  # Adjust based on loss curves
```

### Debugging Training Issues

**Check Data Loading First:**
```python
# Add to training script after dataset creation
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Heatmap shape: {sample['heatmap'].shape}")
print(f"Coordinate: {sample['coords']}")
```

**Visualize Gaussian Targets:**
```python
from utils.gaussian_utils import visualize_heatmap_overlay
overlay = visualize_heatmap_overlay(image, heatmap, alpha=0.5)
plt.imshow(overlay)
plt.show()
```

**Monitor Training:**
- Training loss should decrease steadily
- Validation loss should follow training loss
- If loss plateaus early: increase learning rate or add augmentation
- If loss doesn't decrease: check data loading, coordinate scaling, or learning rate

### Adding New Features

**Implementing Spatial Constraints:**
```python
# After fovea prediction, constrain using disc location
from utils.gaussian_utils import apply_spatial_constraint
constrained_x, constrained_y = apply_spatial_constraint(
    fovea_x, fovea_y, disc_x, disc_y, min_distance=50
)
```

**Creating Dual-Stream Fovea Detector:**
```python
# Use DualStreamUNet in src/03_train_fovea.py
from models.unet import DualStreamUNet
model = DualStreamUNet(in_channels_1=1, in_channels_2=1, out_channels=1)
# Load both B-scan and en face in dataset
```

## Future Enhancements

### Immediate TODOs
- Acquire optic disc labels and train Stage 1
- Implement spatial constraints in inference pipeline (use disc to constrain fovea search)
- Extend training to 100+ epochs for production accuracy

### Potential Improvements
- Post-processing: Use anatomical constraints between stages
- Ensemble models for robustness
- Distance calculation: Find nearest GA boundary from fovea (clinical metric)
- Dual-stream architecture for fovea (combine B-scan + en face)
- Attention mechanisms for better localization

### Research Directions
- Multi-task learning (joint fovea + GA prediction)
- Semi-supervised learning with unlabeled OCT images
- 3D volume analysis (sequences of B-scans)
- Longitudinal tracking (measure GA growth over time)

## Troubleshooting

### GPU/MPS Issues
```python
# Check MPS availability
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```
- If OOM errors: reduce `batch_size` in training scripts
- Fallback to CPU: training will be slower but functional

### Training Issues
- **CSV filename mismatch:** Ensure CSV filenames match processed image names exactly
- **Coordinate out of bounds:** Check that coordinates are valid for image dimensions
- **Loss not decreasing:** Verify data loading, check if images are normalized correctly
- **Validation loss increases:** May need more data augmentation or regularization

### Inference Issues
- **Model not found:** Ensure `.pth` files are in `models/` directory
- **Wrong input format:** Input must be composite image (B-scan + en face side-by-side)
- **Coordinate mismatch:** Inference scales back to original image size automatically

## Critical Implementation Notes

1. **Intelligent Image Splitting (Preprocessing):** The splitting script uses a three-strategy approach to handle different composite image formats:
   - **Strategy 1 - Dividing Line Detection:** Uses variance analysis to detect uniform-color vertical separators (purple/blue lines). If found, completely removes them.
   - **Strategy 2 - Content Boundary Detection:** Analyzes column-to-column content changes to find where B-scan ends and en face begins. Critical because the two images may be different sizes.
   - **Strategy 3 - 50% Fallback:** Only used if both detection methods fail (with warning to user).
   - **Why This Matters:** Models can learn artificial dividers as features; B-scan and en face images are not always equal width; prevents inaccurate splits.

2. **Coordinate Scaling:** When images are resized to 256×256 for training, coordinates must be scaled proportionally. When predicting on new images, coordinates must be scaled back to original size.

3. **Weak Supervision Threshold:** GA masks use `threshold=0.3` (not 0.5) because we want broader coverage for the weak supervision proxy masks.

4. **Loss Function for GA:** Combined BCE + Dice (50/50) because BCE handles pixel-wise classification while Dice optimizes overlap (critical for segmentation).

5. **Filename Consistency:** Image splitting maintains exact filenames, so CSV labels work for both B-scans and en face images.

6. **MPS vs CUDA:** Code automatically detects and uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU. No manual configuration needed.

## Documentation Reference

- **README.md** - User-facing documentation and getting started guide
- **PROJECT_SPECIFICATION.md** - Detailed technical specifications
- **TRAINING_COMPLETE.md** - Training results and performance metrics
- **SUMMARY.md** - High-level project overview
- **QUICK_REFERENCE.md** - Command reference and quick start
- **INFERENCE_RESULTS.md** - Example inference outputs

---

**Last Updated:** January 2026
**Python Version:** 3.13.3
**PyTorch Version:** 2.0+ with MPS support
**Hardware:** Optimized for Apple Silicon M4, works on CUDA and CPU
