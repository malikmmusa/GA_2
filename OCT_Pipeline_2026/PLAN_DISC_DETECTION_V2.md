# Stage 1: Optic Disc Detection - The Perfect Plan (V2.1)

## Overview

Implement the optic disc detection stage with a focus on **automated label extraction** and **precise visualization**. This plan improves upon the original by eliminating manual annotation and aligning the output visualization with clinical expectations (vertical reference line).

**Key Constraint:** Use `data/processed_marked/en_face` for label extraction, as these contain the pre-split images with the disc markers.

---

## Phase 1: Automated Data Preparation

**Task 1.1: Marker Investigation**
- **Goal:** Identify the unique color and shape of the optic disc marker in `data/processed_marked/en_face/`.
- **Action:** Create `src/utils/analyze_markers.py` to detect non-red, non-peach dominant colors in the processed marked en-face images.
- **Hypothesis:** The marker is likely a distinct colored line (e.g., blue, green, or bright yellow) representing the disc's location.

**Task 1.2: Intelligent Label Extraction**
- **Action:** Create `src/00_extract_disc_labels.py` to:
  1. Input directory: `data/processed_marked/en_face/`
  2. Detect the disc marker (vertical line).
  3. Extract the **center coordinates (x, y)** of this line relative to the en-face image dimensions.
  4. Optionally extract the **height/vertical span**.
- **Output:** `data/csv/disc_labels.csv`
  - Columns: `filename,disc_x,disc_y,disc_height`
  - **Note:** Filenames must match the `data/processed/en_face` filenames (which they should, as they usually share the base ID).

**Task 1.3: Verification**
- **Action:** Generate debug images overlaying the extracted centroids on the marked images to ensure accuracy.

---

## Phase 2: Robust Model Training

**Task 2.1: Modernized Training Script**
- **File:** `src/02_train_disc.py`
- **Updates:**
  - Fix deprecated ResNet API: `weights='IMAGENET1K_V1'`.
  - **Loss Function:** `SmoothL1Loss` (robust to outliers).
  - **Metric:** Mean Pixel Error (Euclidean distance).
  - **Input Data:** Load images from `data/processed/en_face` (clean images) but use labels generated from `data/processed_marked/en_face`.

**Task 2.2: Coordinate-Aware Augmentation**
- **Critical:** Since we only have ~52 images, we *must* use augmentation.
- **Logic:** `RandomHorizontalFlip` must also flip the target `x` coordinate: `new_x = width - old_x`.

**Task 2.3: Cross-Validation Strategy**
- **Method:** 5-Fold Cross Validation.
- **Reasoning:** 52 images is too small for a static train/val split. K-Fold ensures every image is used for both training and validation.

---

## Phase 3: Inference & Visualization (User Requested)

**Task 3.1: Pipeline Integration**
- **File:** `src/05_inference.py`
- **Action:** Load the best trained disc model and run prediction on en_face images.

**Task 3.2: "Top to Bottom" Line Visualization**
- **Requirement:** "draw a line from the top to bottom of it [the disc]"
- **Implementation:**
  - If the model predicts height: Draw a vertical line segment from `y - height/2` to `y + height/2` at `x`.
  - If the model only predicts center: Draw a vertical line of a fixed standard length (e.g., 60-80px) or the full image height.
  - **Style:** Use a high-contrast standard color (e.g., Cyan).

**Task 3.3: Spatial Constraints**
- **Logic:** Use the detected Disc X position to constrain the Fovea search space (Fovea is always temporal to Disc).
- **Refinement:** Automatically detect eye laterality (OD/OS) if possible, or accept it as an input.

---

## Execution Roadmap

1.  **Analyze** `data/processed_marked/en_face` images to understand the "line".
2.  **Extract** labels automatically to `data/csv/disc_labels.csv`.
3.  **Train** the model (5-fold) using clean images in `data/processed/en_face` and the extracted labels.
4.  **Integrate** into inference with the specific line drawing.