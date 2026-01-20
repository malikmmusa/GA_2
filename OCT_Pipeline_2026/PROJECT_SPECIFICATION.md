# Project Specification: Modular OCT Analysis Pipeline

## Goal
Build a modular computer vision pipeline to identify anatomical landmarks and pathology in Optical Coherence Tomography (OCT) scans. The project starts from scratch, prioritizing a clean, sequential architecture over complex end-to-end models.

**Domain:** Ophthalmology / Retinal Imaging  
**User Role:** Physician & Developer (Expert context required)

---

## 1. Data Structure & Preprocessing

### Input Data
- **Format:** Single JPG composites containing paired images side-by-side
  - **Left Half:** B-Scan (Cross-sectional view)
  - **Right Half:** En Face (Top-down retinal map)
- **Labels:** CSV file with $(x, y)$ coordinates for two targets: Fovea and Geographic Atrophy (GA)

### Task 1: The Splitter Script (src/01_split_data.py)

**Objective:** Ingest raw composite images and split them into two distinct datasets.

**Logic:**
1. Read image from `data/raw`
2. Detect the vertical divider (or split at exactly 50% width)
3. Save left half → `data/processed/b_scans/`
4. Save right half → `data/processed/en_face/`

**Critical:** Maintain filename consistency so ID `1234.jpg` exists in both folders.

---

## 2. The Three-Stage Pipeline

The system is divided into three independent modules to mimic clinical workflow.

### Stage 1: Optic Disc Detection (The Anchor)

- **Input:** En Face Image (Right half)
- **Model Architecture:** Lightweight CNN (e.g., ResNet18) or Standard CV (Hough Circle/Blob Detection)
- **Output:** $(x, y)$ coordinates of the Optic Disc center
- **Purpose:** Establishes a spatial anchor. The fovea is always temporal to the disc. This constraint reduces false positives in Stage 2.

### Stage 2: Fovea Localization (The Landmark)

- **Input:** Dual-Stream (B-Scan + En Face) OR Single-Stream B-Scan
- **Model Architecture:** U-Net with Heatmap Regression
- **Training Strategy:**
  - Convert point labels $(x, y)$ into 2D Gaussian Heatmaps ($\sigma \approx 15px$)
  - Model predicts the heatmap; loss is calculated via MSE (Mean Squared Error)
  - **Inference:** argmax of the predicted heatmap = $(x, y)$ location
- **Constraint:** If Stage 1 is available, restrict search space to the temporal side of the Optic Disc

### Stage 3: Geographic Atrophy (GA) Segmentation (The Pathology)

- **Input:** En Face Image
- **Model Architecture:** U-Net (Semantic Segmentation)
- **Challenge:** Labels are sparse points, but the target is a region
- **Strategy (Weak Supervision):**
  - During training, generate a "proxy mask" by creating a larger Gaussian blob ($\sigma \approx 50px$) around the labeled GA point
  - Model learns to identify the texture of hyper-transmission defects (bright regions) within that general area
- **Output:** A probability map/binary mask of the GA lesion

---

## 3. Technical Stack & Requirements

- **Language:** Python 3.9+
- **Framework:** PyTorch & TorchVision
- **Image Processing:** OpenCV (cv2) for splitting; Albumentations for training augmentations
- **Logging:** TensorBoard or simple CSV logs

### Directory Structure

```
/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── b_scans/
│   │   └── en_face/
│   └── csv/ (labels)
├── src/
│   ├── models/ (U-Net definitions)
│   ├── utils/ (Gaussian generation, coordinate handling)
│   ├── 01_split_data.py
│   ├── 02_train_disc.py
│   ├── 03_train_fovea.py
│   └── 04_train_ga.py
└── requirements.txt
```

---

## 4. Implementation Notes

### Key Design Principles
1. **Modularity:** Each stage is independent and can be trained/tested separately
2. **Clinical Workflow:** Pipeline mimics how a clinician would analyze these images
3. **Constraint-Based:** Each stage uses information from previous stages to improve accuracy
4. **Weak Supervision:** Handle sparse point annotations intelligently using Gaussian distributions

### Training Strategy
- **Stage 1:** Classification or coordinate regression for disc center
- **Stage 2:** Heatmap regression with Gaussian targets ($\sigma \approx 15px$)
- **Stage 3:** Semantic segmentation with pseudo-masks from Gaussian blobs ($\sigma \approx 50px$)

### Anatomical Constraints
- Fovea is always **temporal** (lateral) to the Optic Disc
- This spatial relationship can be used to filter false positives
- GA regions appear as bright hyper-transmission defects in en face images
