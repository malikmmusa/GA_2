#!/usr/bin/env python3
"""
Enrich test_validation/summary.csv with accuracy metric columns.

Computes in-process (no backend server):
  - prediction_detected: yes if GA region was segmented at the raw_marked GA target click
  - pred_ga_x, pred_ga_y: nearest point on segmented GA contour to fovea (backend definition)
  - gt_ga_x, gt_ga_y: from raw_marked yellow-line endpoint (already in summary or from landmarks)
  - point_error_px: Euclidean distance between pred and GT GA point (pixels)
  - distance_error_px: same as point_error_px
  - cyan_coverage_ratio: fraction of GUI en-face (left half of comparison image) that is cyan overlay

Usage:
  python scripts/enrich_summary_metrics.py
  python scripts/enrich_summary_metrics.py --summary test_validation/summary.csv --input-dir input_images --raw-dir raw_marked
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

# Allow imports from repo root and scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cv2
import numpy as np

import gui_accuracy_validation as val
from src.api.constants import DISC_DIAMETER_MICRONS
from src.api.services.calculator import DistanceCalculatorService
from src.api.services.disc_detector import DiscDetectorService
from src.api.services.ga_segmenter import GASegmenterService

# Default pixel-to-micron when disc not used (nearest_ga_point does not depend on this)
DEFAULT_PIXEL_TO_MICRON = DISC_DIAMETER_MICRONS / 200.0


def compute_cyan_coverage_ratio(comparison_path: Path) -> float:
    """Fraction of left-half (GUI en-face) pixels that are cyan overlay. Returns 0 if file missing."""
    if not comparison_path.exists():
        return float("nan")
    img = cv2.imread(str(comparison_path))
    if img is None:
        return float("nan")
    h, w = img.shape[:2]
    left = img[:, : w // 2]
    b, g, r = left[:, :, 0], left[:, :, 1], left[:, :, 2]
    cyan = (b < 120) & (g > 200) & (r > 200)
    n = left.size // 3
    if n == 0:
        return float("nan")
    return float(np.count_nonzero(cyan)) / n


def enrich_row(
    row: dict,
    meta: val.ImageMeta,
    segmenter: GASegmenterService,
    calculator: DistanceCalculatorService,
    output_dir: Path,
    disc_detector: Optional["DiscDetectorService"] = None,
) -> None:
    """Fill metric columns for one summary row in place."""
    gt_x, gt_y = meta.ga_target_xy if meta.ga_target_xy else (None, None)
    if gt_x is None or meta.fovea_xy is None:
        row["prediction_detected"] = "no"
        return

    row["gt_ga_x"] = f"{gt_x:.2f}"
    row["gt_ga_y"] = f"{gt_y:.2f}"

    image = cv2.imread(str(meta.input_path))
    if image is None:
        row["prediction_detected"] = "no"
        return

    # Detect disc to enable disc masking and proper crop radius in local seg
    disc_cx: Optional[float] = None
    disc_cy: Optional[float] = None
    disc_h: Optional[float] = None
    if disc_detector is not None:
        try:
            disc_info = disc_detector.detect_from_image(image, image_name=meta.filename)
            disc_cx = disc_info.get("disc_center_x")
            disc_cy = disc_info.get("disc_center_y")
            disc_h = disc_info.get("disc_height_pixels")
        except Exception as exc:
            pass  # disc detection failed; proceed without masking

    contours = segmenter.segment_ga_local(
        image=image,
        click_x=float(gt_x),
        click_y=float(gt_y),
        disc_center_x=disc_cx,
        disc_center_y=disc_cy,
        disc_height_pixels=disc_h,
        en_face_split_x=meta.split_x,
    )
    regions = segmenter.contours_to_json(contours)
    if not regions:
        row["prediction_detected"] = "no"
        return

    row["prediction_detected"] = "yes"
    result = calculator.calculate_fovea_to_ga_distance(
        fovea_x=meta.fovea_xy[0],
        fovea_y=meta.fovea_xy[1],
        ga_region=regions[0],
        pixel_to_micron_ratio=DEFAULT_PIXEL_TO_MICRON,
    )
    pred_x = result["nearest_ga_point_x"]
    pred_y = result["nearest_ga_point_y"]
    row["pred_ga_x"] = str(pred_x)
    row["pred_ga_y"] = str(pred_y)
    point_error = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    row["point_error_px"] = f"{point_error:.2f}"
    row["distance_error_px"] = f"{point_error:.2f}"

    out_file = row.get("output_file", "")
    if out_file:
        comparison_path = PROJECT_ROOT / out_file
        ratio = compute_cyan_coverage_ratio(comparison_path)
        row["cyan_coverage_ratio"] = f"{ratio:.4f}" if math.isfinite(ratio) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich summary.csv with accuracy metrics.")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "test_validation" / "summary.csv")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input_images")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "raw_marked")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_validation")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM2 for local segmentation (default: no)")
    parser.add_argument("--use-disc", action="store_true", help="Enable disc detection for masking in local seg (default: no)")
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Summary not found: {args.summary}")
        return 1

    fieldnames = list(val.SUMMARY_FIELDS)
    rows: list[dict[str, str]] = []
    with open(args.summary, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in fieldnames:
                row.setdefault(k, "")
            rows.append(row)

    image_meta = val.prepare_image_meta(args.input_dir, args.raw_dir)
    segmenter = GASegmenterService(use_sam=args.use_sam)
    calculator = DistanceCalculatorService()
    disc_detector: Optional[DiscDetectorService] = None
    if args.use_disc:
        try:
            disc_detector = DiscDetectorService()
            print("Disc detector initialised (enables disc masking in local seg).")
        except Exception as exc:
            print(f"Disc detector unavailable ({exc}); running without disc masking.")

    for i, row in enumerate(rows):
        filename = row.get("image_filename", "")
        meta = image_meta.get(filename)
        if meta is None:
            continue
        enrich_row(row, meta, segmenter, calculator, args.output_dir, disc_detector=disc_detector)
        if (i + 1) % 10 == 0:
            print(f"  Enriched {i + 1}/{len(rows)} rows ...")

    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.summary} with {len(fieldnames)} columns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
