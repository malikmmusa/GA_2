# Marked OCT Images Directory

This directory contains manually annotated OCT images for automated label extraction.

## Marking Convention

### Fovea Detection
- **Mark with**: Red dot (RGB close to pure red)
- **Optional**: Blue outline around the red dot is acceptable
- **Purpose**: Identifies the central fovea location

### GA (Geographic Atrophy) Detection
- **Mark with**: Peach/yellow line (Hex: #F4C5AD, RGB: 244, 197, 173)
- **Line meaning**: Connects the fovea to the nearest edge of the GA lesion
- **Extraction logic**: The script finds the endpoint FARTHEST from the fovea as the GA target point
- **The endpoint closest to fovea is ignored** (it's just the ruler start point)

## Image Format
- Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF
- Any resolution (the script processes original resolution)

## Usage

After placing your marked images here, run:

```bash
# Basic usage (creates train_fovea_labels.csv and train_ga_labels.csv)
python src/00_extract_vector_labels.py

# With train/val split (20% validation)
python src/00_extract_vector_labels.py --split 0.2

# With debug visualization (saves annotated images to data/debug/)
python src/00_extract_vector_labels.py --debug

# Full example with all options
python src/00_extract_vector_labels.py \
    --input data/raw_marked \
    --output data/csv \
    --split 0.2 \
    --debug \
    --debug-dir data/debug
```

## Output

The script generates CSV files in `data/csv/`:
- `train_fovea_labels.csv`: filename, fovea_x, fovea_y
- `train_ga_labels.csv`: filename, ga_x, ga_y
- `val_fovea_labels.csv`: (if --split > 0)
- `val_ga_labels.csv`: (if --split > 0)

These CSV files are ready to use with the pipeline's training scripts (Stage 2 and Stage 3).

## Troubleshooting

If images fail to process, run with `--debug` to see visualization of what was detected:
- Green circle = Detected fovea
- Blue line = Detected peach line skeleton
- Cyan circles = Line endpoints
- Red circle = Selected GA target point

Common issues:
- **Red dot not detected**: Make sure the red color is bright and saturated
- **Peach line not detected**: Adjust color tolerance with `--peach-tolerance H S V`
- **Wrong endpoint selected**: Check that line clearly connects fovea to GA edge
