#!/usr/bin/env python3
"""
Extract one row per candidate GA region, with features and a quality label.

Used to train and evaluate a region ranker. The oracle analysis showed the
segmenter already returns a near-correct region on most images (best-available
MAE 133 um, ICC 0.892, against 368 um / 0.176 actually achieved), so nearly all
remaining autonomous error is *choosing* among regions rather than finding them.

Features come from `GASegmenterService.region_features` and the candidate pool
from `passes_filters`, both shared with the serving path, so the training set
cannot describe a different world than inference sees.

Labels:
  dist_to_gt_px    - from the region's nearest-to-fovea point to the marked GA edge
  abs_err_um       - |measurement this region would produce - ground truth|
  is_correct       - dist_to_gt_px <= --positive-threshold

Foveal-involvement cases are skipped: with no GA ruler drawn there is no target
to rank against.

Usage:
  python scripts/build_ranker_dataset.py
  python scripts/build_ranker_dataset.py --cv2-seed 1 --out data/training/ranker_seed1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.disc_detector import DiscDetectorService  # noqa: E402
from src.api.services.fovea_detector import FoveaDetectorService  # noqa: E402
from src.api.services.ga_segmenter import GASegmenterService  # noqa: E402

DEFAULT_GT = PROJECT_ROOT / "test_validation" / "bland_altman_data.csv"
DEFAULT_IMAGES = PROJECT_ROOT / "input_images"
DEFAULT_OUT = PROJECT_ROOT / "data" / "training" / "ranker_dataset.csv"
DISC_DIAMETER_MICRONS = 1800.0


def extract_rows(args) -> pd.DataFrame:
    cv2.setRNGSeed(args.cv2_seed)
    disc_svc = DiscDetectorService()
    fovea_svc = FoveaDetectorService()
    # max_regions=None: the ranker must see the whole pool, not the top slice.
    segmenter = GASegmenterService(use_sam=False, max_regions=None)

    gt = pd.read_csv(args.gt_csv)
    rows = []
    skipped = {"foveal": 0, "no_image": 0, "no_gt": 0, "failed": 0, "no_regions": 0}

    for _, r in gt.iterrows():
        image_id = str(r["image_id"])
        if str(r.get("foveal_involvement", "")).lower() == "true":
            skipped["foveal"] += 1
            continue
        if pd.isna(r.get("gt_ga_x")) or pd.isna(r.get("gt_distance_um")):
            skipped["no_gt"] += 1
            continue

        path = args.image_dir / f"{image_id}.png"
        if not path.exists():
            skipped["no_image"] += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            skipped["no_image"] += 1
            continue

        try:
            disc = disc_svc.detect_from_image(image, path.name)
            split_x = int(disc["en_face_split_x"])
            fovea = fovea_svc.detect_fovea(
                image, disc["disc_center_x"], disc["disc_center_y"],
                disc["disc_height_pixels"], split_x, use_manual_adjustment=False)
        except Exception:
            skipped["failed"] += 1
            continue

        en_face = image[:, split_x:]
        gray = cv2.cvtColor(en_face, cv2.COLOR_BGR2GRAY)
        enhanced = segmenter._apply_clahe(gray)

        contours = segmenter.segment_ga_regions(
            image, en_face_split_x=split_x,
            disc_center_x=disc["disc_center_x"], disc_center_y=disc["disc_center_y"],
            disc_height_pixels=disc["disc_height_pixels"],
            fovea_x=fovea["fovea_x"], fovea_y=fovea["fovea_y"])
        if not contours:
            skipped["no_regions"] += 1
            continue

        um_per_px = DISC_DIAMETER_MICRONS / float(r["gt_disc_height_px"])
        fovea_pt = np.array([fovea["fovea_x"], fovea["fovea_y"]], dtype=np.float64)
        gt_ga = np.array([float(r["gt_ga_x"]), float(r["gt_ga_y"])], dtype=np.float64)
        gt_um = float(r["gt_distance_um"])

        for idx, contour in enumerate(contours):
            # Contours come back in full-composite coordinates; features are
            # computed in en-face space.
            local = contour.copy()
            local[:, 0, 0] -= split_x

            area = cv2.contourArea(local)
            passed, circularity = segmenter.passes_filters(local, area, gray.shape)
            if not passed:
                continue

            features = segmenter.region_features(
                local, area, circularity, enhanced, gray.shape, cluster_rank=0)

            points = contour.reshape(-1, 2).astype(np.float64)
            nearest_to_fovea = float(np.min(np.linalg.norm(points - fovea_pt, axis=1)))
            dist_to_gt = float(np.min(np.linalg.norm(points - gt_ga, axis=1)))
            pred_um = nearest_to_fovea * um_per_px

            row = {
                "image_id": image_id,
                "region_idx": idx,
                "served_rank": idx,
                "served_score": segmenter.score_region(features),
                "pred_um": pred_um,
                "gt_um": gt_um,
                "abs_err_um": abs(pred_um - gt_um),
                "dist_to_gt_px": dist_to_gt,
                "is_correct": int(dist_to_gt <= args.positive_threshold),
            }
            row.update(features)
            rows.append(row)

    print("Skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the GA region ranker dataset.")
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cv2-seed", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=60.0,
                        help="px from the marked GA edge to count a region correct")
    args = parser.parse_args()

    df = extract_rows(args)
    if df.empty:
        print("[ERROR] No rows extracted")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_img = df.image_id.nunique()
    with_pos = df.groupby("image_id").is_correct.max().sum()
    print(f"\nWrote {args.out}")
    print(f"  {len(df)} regions across {n_img} images")
    print(f"  {int(df.is_correct.sum())} positive ({df.is_correct.mean()*100:.1f}%)")
    print(f"  {int(with_pos)}/{n_img} images contain at least one correct region")
    return 0


if __name__ == "__main__":
    sys.exit(main())
