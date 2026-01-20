# OCT Pipeline 2026: Modular Retinal Analysis System

A modular computer vision pipeline for identifying anatomical landmarks and pathology in Optical Coherence Tomography (OCT) scans.

**Domain:** Ophthalmology / Retinal Imaging  
**Architecture:** Sequential, clinically-motivated stages  
**Purpose:** Research & Clinical Decision Support

---

## 📋 Project Overview

This pipeline processes composite OCT images through three independent stages:

1. **Optic Disc Detection** - Spatial anchor using ResNet18
2. **Fovea Localization** - Landmark detection using U-Net with heatmap regression
3. **Geographic Atrophy Segmentation** - Pathology mapping using U-Net

Each stage can be trained and evaluated independently, mimicking clinical workflow.

---

## 📁 Directory Structure

```
OCT_Pipeline_2026/
├── data/
│   ├── raw/                    # Composite OCT images (JPG)
│   ├── processed/
│   │   ├── b_scans/            # Left half - B-Scans
│   │   └── en_face/            # Right half - En Face
│   └── csv/                    # Label files
│       ├── train_disc_labels.csv
│       ├── val_disc_labels.csv
│       ├── train_fovea_labels.csv
│       ├── val_fovea_labels.csv
│       ├── train_ga_labels.csv
│       └── val_ga_labels.csv
├── src/
│   ├── models/
│   │   └── unet.py            # U-Net architectures
│   ├── utils/
│   │   └── gaussian_utils.py  # Heatmap generation utilities
│   ├── 01_split_data.py       # Image preprocessing
│   ├── 02_train_disc.py       # Stage 1: Optic disc
│   ├── 03_train_fovea.py      # Stage 2: Fovea
│   └── 04_train_ga.py         # Stage 3: GA segmentation
├── notebooks/                  # Jupyter notebooks for analysis
├── models/                     # Saved model weights (created during training)
├── requirements.txt
├── README.md
└── PROJECT_SPECIFICATION.md   # Detailed technical specification
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd OCT_Pipeline_2026

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

**Step 1: Place composite images**
```bash
# Copy your composite OCT images to:
data/raw/
```

**Step 2: Create label CSV files**

Each CSV should have the following format:

```csv
filename,disc_x,disc_y
image001.jpg,512,384
image002.jpg,498,392
...
```

For fovea and GA labels, use appropriate column names (`fovea_x`, `fovea_y` or `ga_x`, `ga_y`).

Place CSV files in `data/csv/`:
- `train_disc_labels.csv` / `val_disc_labels.csv`
- `train_fovea_labels.csv` / `val_fovea_labels.csv`
- `train_ga_labels.csv` / `val_ga_labels.csv`

### 3. Run the Pipeline

**Stage 0: Split Composite Images**
```bash
python src/01_split_data.py
```
This splits each composite image into B-Scan (left) and En Face (right) halves.

**Stage 1: Train Optic Disc Detector**
```bash
python src/02_train_disc.py
```
Trains a ResNet18 model to detect optic disc coordinates.

**Stage 2: Train Fovea Detector**
```bash
python src/03_train_fovea.py
```
Trains a U-Net model with Gaussian heatmap regression.

**Stage 3: Train GA Segmentation**
```bash
python src/04_train_ga.py
```
Trains a U-Net model for geographic atrophy segmentation.

---

## 🧬 Technical Details

### Stage 1: Optic Disc Detection
- **Input:** En Face images (512×512 or native resolution)
- **Model:** ResNet18 backbone with regression head
- **Output:** (x, y) coordinates normalized to [0, 1]
- **Loss:** Mean Squared Error (MSE)
- **Purpose:** Provides spatial anchor for fovea detection

### Stage 2: Fovea Localization
- **Input:** B-Scan images (and optionally En Face)
- **Model:** U-Net with heatmap regression
- **Training:** 
  - Point labels → Gaussian heatmaps (σ ≈ 15px)
  - Loss: MSE between predicted and target heatmaps
- **Inference:** argmax of predicted heatmap
- **Constraint:** Fovea is temporal to optic disc

### Stage 3: GA Segmentation
- **Input:** En Face images
- **Model:** U-Net for semantic segmentation
- **Training (Weak Supervision):**
  - Point labels → Gaussian blobs (σ ≈ 50px) as proxy masks
  - Model learns to identify hyper-transmission defects
- **Output:** Binary mask or probability map
- **Loss:** Binary Cross-Entropy (BCE)

---

## 📊 Data Format

### Composite Images
- **Format:** JPG (or PNG)
- **Layout:** Side-by-side concatenation
  - Left 50%: B-Scan (cross-sectional)
  - Right 50%: En Face (top-down)

### Label CSV Format

**Optic Disc Labels:**
```csv
filename,disc_x,disc_y
patient001.jpg,512,384
patient002.jpg,498,392
```

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

---

## 🔬 Development & Experiments

### Using Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` for:
- Data exploration and visualization
- Model testing and debugging
- Results analysis
- Heatmap visualization

### Monitoring Training

Training logs are saved with TensorBoard support:

```bash
tensorboard --logdir=runs/
```

---

## 🎯 Clinical Context

### Anatomical Relationships
1. **Optic Disc (ONH):** Bright circular region where optic nerve enters the retina
2. **Fovea:** Central pit responsible for sharp vision, temporal to the disc
3. **Geographic Atrophy (GA):** Bright hyper-transmission defects indicating RPE/photoreceptor loss

### Spatial Constraints
- Fovea is always **temporal** (lateral) to the optic disc
- Typical fovea-disc distance: ~4-5 mm (~200-300 pixels depending on resolution)
- GA lesions appear as bright regions in en face OCT due to loss of overlying tissue

---

## 🛠️ Customization

### Adjusting Gaussian Parameters

Edit `src/utils/gaussian_utils.py`:

```python
# For fovea (sharper localization)
sigma_fovea = 15  # pixels

# For GA (broader region)
sigma_ga = 50  # pixels
```

### Model Architecture

Edit `src/models/unet.py` to:
- Change number of layers
- Adjust feature channels
- Switch between bilinear and transposed convolutions

### Training Hyperparameters

Each training script has configurable parameters:
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Initial learning rate
- Image augmentations (in transforms)

---

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{oct_pipeline_2026,
  title={Modular OCT Analysis Pipeline for Retinal Landmark Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/oct-pipeline}
}
```

---

## 📄 License

[Specify your license here]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or collaboration:
- Email: your.email@domain.com
- GitHub: @yourusername

---

## ⚠️ Disclaimer

This software is for research purposes only and is not intended for clinical use without proper validation and regulatory approval.
