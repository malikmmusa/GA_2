#!/usr/bin/env python3
"""
Build fovea training labels in en-face crop coordinates.

Source of truth is the clinician's marking in `raw_marked/`, extracted by
`scripts/batch_bland_altman_validation.py` into the `gt_fovea_x/gt_fovea_y`
columns of `test_validation/bland_altman_data.csv`. Those coordinates are in
*full composite* image space, while `data/training/en_face/` holds right-hand
crops, so every point needs shifting by that image's split_x.

split_x is recovered by exact pixel alignment rather than assumed. 51 of 52
crops are flush-right slices (`split_x = full_width - crop_width`), but
22205914.png is not — its crop starts at 1845 and stops 6 px short of the right
edge. Assuming flush-right would put every label on that image 6 px off, so the
crop is located by searching for the offset that reproduces it exactly.

Usage:
  python scripts/build_fovea_labels.py
  python scripts/build_fovea_labels.py --verify      # check an existing file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DISC_CSV = PROJECT_ROOT / "data" / "training" / "disc_labels_v2.csv"
DEFAULT_GT_CSV = PROJECT_ROOT / "test_validation" / "bland_altman_data.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "input_images"
DEFAULT_ENFACE_DIR = PROJECT_ROOT / "data" / "training" / "en_face"
DEFAULT_OUT = PROJECT_ROOT / "data" / "training" / "fovea_labels_v1.csv"

SEARCH_RADIUS_PX = 40


def find_split_x(full: np.ndarray, crop: np.ndarray) -> Optional[int]:
    """Locate the crop inside the full image, returning its left edge.

    Tries the flush-right position first (true for all but one image), then
    searches nearby. Returns None if no offset reproduces the crop exactly.
    """
    crop_w = crop.shape[1]
    flush = full.shape[1] - crop_w
    candidates = [flush] + [
        flush + d
        for r in range(1, SEARCH_RADIUS_PX + 1)
        for d in (-r, r)
    ]
    for s in candidates:
        if s < 0 or s + crop_w > full.shape[1]:
            continue
        if np.array_equal(full[:, s:s + crop_w], crop):
            return int(s)
    return None


def build(args: argparse.Namespace) -> pd.DataFrame:
    disc = pd.read_csv(args.disc_csv)
    gt = pd.read_csv(args.gt_csv)
    gt["image_id"] = gt["image_id"].astype(str)

    rows, problems = [], []
    for _, r in disc.iterrows():
        filename = str(r["filename"])
        stem = Path(filename).stem

        full = cv2.imread(str(args.image_dir / filename))
        crop = cv2.imread(str(args.enface_dir / filename))
        if full is None or crop is None:
            problems.append(f"{filename}: image or en-face crop missing")
            continue

        if full.shape[0] != crop.shape[0]:
            problems.append(f"{filename}: crop height {crop.shape[0]} != full {full.shape[0]}")
            continue

        split_x = find_split_x(full, crop)
        if split_x is None:
            problems.append(f"{filename}: could not locate crop within full image")
            continue

        match = gt[gt["image_id"] == stem]
        if match.empty:
            problems.append(f"{filename}: no ground-truth row")
            continue

        fx = pd.to_numeric(match.iloc[0].get("gt_fovea_x"), errors="coerce")
        fy = pd.to_numeric(match.iloc[0].get("gt_fovea_y"), errors="coerce")
        if pd.isna(fx) or pd.isna(fy):
            problems.append(f"{filename}: ground-truth fovea is blank")
            continue

        local_x = float(fx) - split_x
        local_y = float(fy)
        if not (0 <= local_x < crop.shape[1] and 0 <= local_y < crop.shape[0]):
            problems.append(
                f"{filename}: fovea ({local_x:.0f}, {local_y:.0f}) outside crop "
                f"{crop.shape[1]}x{crop.shape[0]}"
            )
            continue

        rows.append({
            "filename": filename,
            "fovea_x": round(local_x, 2),
            "fovea_y": round(local_y, 2),
            "split_x": split_x,
            "flush_right": int(split_x == full.shape[1] - crop.shape[1]),
        })

    if problems:
        print(f"[warn] {len(problems)} image(s) excluded:")
        for p in problems:
            print(f"       {p}")

    return pd.DataFrame(rows)


def verify(path: Path, args: argparse.Namespace) -> int:
    """Re-derive labels and confirm the stored file still matches."""
    if not path.exists():
        print(f"[ERROR] No label file at {path}")
        return 1
    stored = pd.read_csv(path)
    fresh = build(args)
    merged = stored.merge(fresh, on="filename", suffixes=("_stored", "_fresh"))
    if len(merged) != len(stored):
        print(f"[FAIL] {len(stored) - len(merged)} stored row(s) no longer derivable")
        return 1
    drift = np.hypot(
        merged.fovea_x_stored - merged.fovea_x_fresh,
        merged.fovea_y_stored - merged.fovea_y_fresh,
    )
    if drift.max() > 0.01:
        print(f"[FAIL] stored labels drifted from source, max {drift.max():.2f} px")
        return 1
    print(f"[ok]   {len(stored)} labels reproduce from raw_marked exactly")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fovea labels in en-face coordinates.")
    parser.add_argument("--disc-csv", type=Path, default=DEFAULT_DISC_CSV)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--enface-dir", type=Path, default=DEFAULT_ENFACE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true", help="Check an existing file, write nothing")
    args = parser.parse_args()

    if args.verify:
        return verify(args.out, args)

    df = build(args)
    if df.empty:
        print("[ERROR] No labels produced")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} labels)")
    print(f"  non-flush-right crops: {int((df.flush_right == 0).sum())} "
          f"({', '.join(df[df.flush_right == 0].filename) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
