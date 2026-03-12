#!/usr/bin/env python3
"""
Produce a ranked worst-case table from test_validation/summary.csv.

Requires summary.csv to have metric columns populated by gui_accuracy_validation.py:
  - prediction_detected, pred_ga_x, pred_ga_y, gt_ga_x, gt_ga_y
  - point_error_px, distance_error_px, cyan_coverage_ratio

Usage:
  python scripts/rank_validation_errors.py
  python scripts/rank_validation_errors.py --summary test_validation/summary.csv --top 20 --out test_validation/worst_cases.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_float(s: str) -> float | None:
    if not s or not s.strip():
        return None
    try:
        return float(s.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank validation errors by point_error_px.")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "test_validation" / "summary.csv")
    parser.add_argument("--top", type=int, default=0, help="Limit to top N worst (0=all)")
    parser.add_argument("--out", type=Path, default=None, help="Write ranked table to CSV")
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Summary not found: {args.summary}", file=sys.stderr)
        print("Run gui_accuracy_validation.py first to populate metrics.", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with open(args.summary, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Filter rows with prediction_detected=yes and non-empty point_error_px
    scored: list[tuple[dict, float]] = []
    for row in rows:
        if row.get("prediction_detected") != "yes":
            continue
        err = parse_float(row.get("point_error_px", ""))
        if err is None:
            continue
        scored.append((row, err))

    if not scored:
        print("No rows with prediction_detected=yes and point_error_px.")
        print("Run gui_accuracy_validation.py to populate metrics.")
        return 0

    # Sort by point_error_px descending (worst first)
    scored.sort(key=lambda x: x[1], reverse=True)

    if args.top > 0:
        scored = scored[: args.top]

    # Print table
    print(f"\nWorst case GA errors (ranked by point_error_px, n={len(scored)}):\n")
    print(f"{'case_key':<12} {'image_filename':<22} {'point_error_px':>14} {'distance_error_px':>18} {'cyan_coverage':>14}")
    print("-" * 80)

    for row, _ in scored:
        case = row.get("case_key", "")
        img = row.get("image_filename", "")
        pt_err = row.get("point_error_px", "")
        dist_err = row.get("distance_error_px", "")
        cyan = row.get("cyan_coverage_ratio", "")
        print(f"{case:<12} {img:<22} {pt_err:>14} {dist_err:>18} {cyan:>14}")

    # Optional CSV output
    if args.out:
        out_fields = ["rank", "case_key", "image_filename", "point_error_px", "distance_error_px", "cyan_coverage_ratio", "output_file"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            for i, (row, _) in enumerate(scored, start=1):
                out_row = {"rank": i, **row}
                writer.writerow(out_row)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
