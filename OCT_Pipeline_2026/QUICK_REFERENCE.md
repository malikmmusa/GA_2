# OCT Pipeline 2026 - Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Setup
```bash
cd OCT_Pipeline_2026
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pipeline Execution (In Order)

**Step 0: Split Data**
```bash
python src/01_split_data.py
```
- Input: `data/raw/*.jpg` (composite images)
- Output: `data/processed/b_scans/` and `data/processed/en_face/`

**Step 1: Train Optic Disc Detector**
```bash
python src/02_train_disc.py
```
- Input: En face images + `data/csv/train_disc_labels.csv`
- Output: `models/disc_detector.pth`

**Step 2: Train Fovea Detector**
```bash
python src/03_train_fovea.py
```
- Input: B-scan images + `data/csv/train_fovea_labels.csv`
- Output: `models/fovea_detector.pth`

**Step 3: Train GA Segmenter**
```bash
python src/04_train_ga.py
```
- Input: En face images + `data/csv/train_ga_labels.csv`
- Output: `models/ga_segmenter.pth`

---

## 📋 Required CSV Format

### Disc Labels
```csv
filename,disc_x,disc_y
patient001.jpg,512,384
patient002.jpg,498,392
```

### Fovea Labels
```csv
filename,fovea_x,fovea_y
patient001.jpg,723,401
patient002.jpg,698,388
```

### GA Labels
```csv
filename,ga_x,ga_y
patient001.jpg,645,425
patient002.jpg,623,412
```

Place these files in `data/csv/`:
- `train_disc_labels.csv` / `val_disc_labels.csv`
- `train_fovea_labels.csv` / `val_fovea_labels.csv`
- `train_ga_labels.csv` / `val_ga_labels.csv`

---

## 🧬 Key Parameters

### Stage 1: Optic Disc Detection
- **Model:** ResNet18
- **Input Size:** 224x224
- **Loss:** MSE
- **Default Epochs:** 50
- **Batch Size:** 16

### Stage 2: Fovea Localization
- **Model:** U-Net with Sigmoid
- **Input Size:** 256x256
- **Gaussian σ:** 15px (heatmap)
- **Loss:** MSE
- **Default Epochs:** 100
- **Batch Size:** 8

### Stage 3: GA Segmentation
- **Model:** U-Net (Segmentation)
- **Input Size:** 256x256
- **Gaussian σ:** 50px (proxy mask)
- **Loss:** BCE + Dice (50/50)
- **Default Epochs:** 100
- **Batch Size:** 8

---

## 🔧 Customization

### Change Gaussian Parameters
Edit `src/utils/gaussian_utils.py`:
```python
sigma_fovea = 15  # For fovea (sharper)
sigma_ga = 50     # For GA (broader)
```

### Adjust Training Hyperparameters
In each training script, modify:
```python
num_epochs = 100
batch_size = 8
learning_rate = 1e-4
```

### Change Image Size
In dataset classes, modify:
```python
output_size = (256, 256)  # (width, height)
```

---

## 📊 Monitoring Training

### TensorBoard (if implemented)
```bash
tensorboard --logdir=runs/
```

### Check Model Files
```bash
ls -lh models/
```

---

## 🧪 Testing & Visualization

### Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to `notebooks/01_exploration.ipynb`

### Quick Test
```python
from pathlib import Path
import torch
from models.unet import UNetWithSigmoid

model = UNetWithSigmoid(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('models/fovea_detector.pth'))
model.eval()
```

---

## 📁 Directory Structure Reminder

```
OCT_Pipeline_2026/
├── data/
│   ├── raw/                    # Your composite images here
│   ├── processed/
│   │   ├── b_scans/
│   │   └── en_face/
│   └── csv/                    # Label CSVs here
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── gaussian_utils.py
│   ├── 01_split_data.py
│   ├── 02_train_disc.py
│   ├── 03_train_fovea.py
│   └── 04_train_ga.py
├── notebooks/
│   └── 01_exploration.ipynb
├── models/                     # Generated during training
│   ├── disc_detector.pth
│   ├── fovea_detector.pth
│   └── ga_segmenter.pth
└── requirements.txt
```

---

## ⚠️ Common Issues

### Issue: "No images found"
**Solution:** Place JPG images in `data/raw/`

### Issue: "Cannot find CSV"
**Solution:** Create label CSVs in `data/csv/` with correct format

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size in training scripts

### Issue: "Module not found"
**Solution:** Check that you're running from project root and `src/` is in path

---

## 🎯 Expected Results

### After Stage 0:
- B-scans and en face images split successfully
- Same number of files in both directories

### After Stage 1:
- Disc detector model saved
- Validation loss < 0.01 (for normalized coordinates)

### After Stage 2:
- Fovea detector model saved
- Pixel error < 20px on validation set

### After Stage 3:
- GA segmenter model saved
- Dice coefficient > 0.6 on validation set

---

## 📝 Citation Template

```bibtex
@software{oct_pipeline_2026,
  title={Modular OCT Analysis Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/oct-pipeline}
}
```

---

## 📧 Need Help?

1. Check `PROJECT_SPECIFICATION.md` for detailed technical specs
2. Review `README.md` for comprehensive documentation
3. Open an issue on GitHub
4. Contact: your.email@domain.com
