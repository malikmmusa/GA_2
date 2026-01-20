# Scripts Directory

This directory contains preprocessing and utility scripts for the OCT Pipeline.

## Available Scripts

### 00_extract_vector_labels.py
Located in `src/00_extract_vector_labels.py`

**Purpose**: Extract fovea and GA coordinates from manually marked images

**Input**: Marked images in `data/raw_marked/`
- Fovea: Red dot
- GA: Peach/yellow line from fovea to GA edge

**Output**: CSV label files in `data/csv/`
- `train_fovea_labels.csv`
- `train_ga_labels.csv`
- (optional) validation splits

**Usage**:
```bash
python src/00_extract_vector_labels.py --split 0.2 --debug
```

See `data/raw_marked/README.md` for detailed marking conventions.

---

## Pipeline Workflow

1. **Pre-processing**: Extract labels from marked images
   ```bash
   python src/00_extract_vector_labels.py --split 0.2
   ```

2. **Stage 0**: Split composite images
   ```bash
   python src/01_split_data.py
   ```

3. **Stage 1**: Train optic disc detector
   ```bash
   python src/02_train_disc.py
   ```

4. **Stage 2**: Train fovea detector
   ```bash
   python src/03_train_fovea.py
   ```

5. **Stage 3**: Train GA segmenter
   ```bash
   python src/04_train_ga.py
   ```
