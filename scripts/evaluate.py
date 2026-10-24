#!/usr/bin/env python3
"""
Single source of truth for pipeline accuracy.

Why this exists
---------------
Accuracy numbers for this project have lived in several places at once —
``test_validation/summary.csv``, the Bland-Altman outputs, the HTML reports —
measured under different conditions and not labelled as such. The current
``summary.csv`` is an *assisted* run (GA clicked by hand, disc dragged to
ground truth) with a mean point error of ~7 px, which reads like excellent
accuracy but is really operator click precision. The autonomous pipeline over
the same images has MAE ~550 µm and r ~ -0.18.

This script always reports **both modes side by side**, so a number can't be
quoted without its condition attached, and it reports on the frozen holdout
separately from the development eyes.

  ASSISTED   = clinician anchors the GA point; pipeline does the rest.
  AUTONOMOUS = no human input. This is what ships.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --subset test          # frozen holdout only
  python scripts/evaluate.py --save-baseline test_validation/baseline.json
  python scripts/evaluate.py --baseline test_validation/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bland_altman_analysis import (  # noqa: E402
    bland_altman_stats,
    component_stats,
    get_autonomous_distance_pairs,
    get_distance_pairs,
    load_csv,
)
from scripts.make_splits import eye_id, load_split  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "test_validation" / "bland_altman_data.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "splits_v1.json"

# ICC bands in common clinical use (Koo & Li 2016).
ICC_BANDS = [(0.90, "excellent"), (0.75, "good"), (0.50, "moderate"), (0.0, "poor")]


def display_path(path: Path) -> str:
    """Project-relative path for display, tolerant of relative CLI arguments."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def icc_label(icc: float) -> str:
    for threshold, label in ICC_BANDS:
        if icc >= threshold:
            return label
    return "poor"


def subset_rows(rows: List[Dict], split: Optional[Dict], subset: str) -> List[Dict]:
    """Filter rows to 'all', the frozen holdout ('test'), or the CV eyes ('dev')."""
    if subset == "all" or split is None:
        return rows
    test_eyes = set(split["test"]["eyes"])
    if subset == "test":
        keep = test_eyes
    else:
        keep = {e for f in split["folds"] for e in f["eyes"]}
    return [r for r in rows if eye_id(r["image_id"]) in keep]


def mode_metrics(rows: List[Dict], mode: str) -> Optional[Dict]:
    """Compute Bland-Altman metrics for one measurement mode."""
    pair_fn = get_distance_pairs if mode == "assisted" else get_autonomous_distance_pairs
    gt, pred, ids = pair_fn(rows)
    if len(gt) < 3:
        return None
    stats_dict = bland_altman_stats(gt, pred)
    stats_dict["gt_mean"] = float(np.mean(gt))
    stats_dict["pred_mean"] = float(np.mean(pred))
    stats_dict["ids"] = ids
    return stats_dict


def fmt_delta(current: float, baseline: Optional[float], lower_is_better: bool) -> str:
    if baseline is None:
        return ""
    delta = current - baseline
    if abs(delta) < 1e-9:
        return "    (=)"
    improved = (delta < 0) if lower_is_better else (delta > 0)
    return f"  ({delta:+.3f} {'better' if improved else 'worse'})"


def print_mode(title: str, m: Optional[Dict], base: Optional[Dict]) -> None:
    print(f"\n  {title}")
    if m is None:
        print("    (insufficient paired measurements)")
        return
    b = base or {}
    print(f"    n                {m['n']}")
    print(f"    truth mean       {m['gt_mean']:8.0f} µm")
    print(f"    predicted mean   {m['pred_mean']:8.0f} µm")
    print(f"    MAE              {m['mae']:8.1f} µm{fmt_delta(m['mae'], b.get('mae'), True)}")
    print(f"    RMSE             {m['rmse']:8.1f} µm{fmt_delta(m['rmse'], b.get('rmse'), True)}")
    print(f"    bias             {m['bias']:+8.1f} µm")
    print(f"    95% LOA          [{m['loa_lo']:+.0f}, {m['loa_hi']:+.0f}] µm  (span {m['loa_hi'] - m['loa_lo']:.0f})")
    print(f"    Pearson r        {m['pearson_r']:+8.3f}{fmt_delta(m['pearson_r'], b.get('pearson_r'), False)}")
    print(f"    ICC(2,1)         {m['icc']:+8.3f}  [{icc_label(m['icc'])}]"
          f"{fmt_delta(m['icc'], b.get('icc'), False)}")
    print(f"    prop. bias slope {m['prop_bias_slope']:+8.2f}  (p={m['prop_bias_p']:.4f})")

    # A difference-vs-mean slope near -2 is the signature of an output that
    # barely varies with the truth, i.e. a near-constant prediction.
    if m["prop_bias_slope"] < -1.0 and m["prop_bias_p"] < 0.05:
        print("      ^ strongly negative slope: output is largely independent of the true value")


def print_components(comp: Dict, base: Optional[Dict]) -> None:
    print("\n  Component errors (pixels)")
    labels = {
        "disc_height_px": "disc height",
        "fovea_px": "fovea position",
        "ga_point_px": "GA edge point",
    }
    for key, label in labels.items():
        c = comp.get(key, {})
        if not c.get("n"):
            print(f"    {label:<16} (no data)")
            continue
        bmae = (base or {}).get(key, {}).get("mae")
        print(f"    {label:<16} n={c['n']:<3d} MAE={c['mae']:6.1f}  median={c['median']:6.1f}  "
              f"p90={c['p90']:6.1f}{fmt_delta(c['mae'], bmae, True)}")


def build_report(rows: List[Dict]) -> Dict:
    return {
        "n_rows": len(rows),
        "assisted": mode_metrics(rows, "assisted"),
        "autonomous": mode_metrics(rows, "autonomous"),
        "components": component_stats(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report autonomous and assisted accuracy side by side.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--subset", choices=["all", "dev", "test"], default="all",
                        help="'test' = frozen holdout only; 'dev' = CV eyes only")
    parser.add_argument("--save-baseline", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None, help="Compare against a saved baseline")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Data CSV not found: {args.data}")
        print("        Generate it with: python scripts/batch_bland_altman_validation.py")
        return 1

    rows = load_csv(args.data)

    split = None
    if args.split.exists():
        split = load_split(args.split)
    elif args.subset != "all":
        print(f"[ERROR] --subset {args.subset} needs a split file; {args.split} not found")
        return 1

    rows = subset_rows(rows, split, args.subset)
    if not rows:
        print(f"[ERROR] No rows in subset '{args.subset}'")
        return 1

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report = build_report(rows)

    subset_desc = {
        "all": "all eyes",
        "dev": "development eyes (CV folds)",
        "test": "FROZEN HOLDOUT eyes",
    }[args.subset]

    print("=" * 74)
    print(f"  Pipeline accuracy — {subset_desc}  ({len(rows)} images)")
    print(f"  data: {display_path(args.data)}")
    if baseline:
        print(f"  baseline: {display_path(args.baseline)}")
    print("=" * 74)

    print_mode(
        "AUTONOMOUS — no human input. This is what ships.",
        report["autonomous"], (baseline or {}).get("autonomous"),
    )
    print_mode(
        "ASSISTED — clinician anchors the GA point. Not a measure of the shipped pipeline.",
        report["assisted"], (baseline or {}).get("assisted"),
    )
    print_components(report["components"], (baseline or {}).get("components"))

    if args.subset == "all" and split is not None:
        print("\n  Note: 'all' mixes development and holdout eyes. Quote --subset test "
              "\n        for a number that reflects unseen data.")

    print()

    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        serialisable = json.loads(json.dumps(report, default=float))
        args.save_baseline.write_text(json.dumps(serialisable, indent=2) + "\n")
        print(f"Baseline saved to {args.save_baseline}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
