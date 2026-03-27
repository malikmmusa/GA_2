#!/usr/bin/env python3
"""
Predictive Validation Module — Abstract 2 of the Atrophy Advisor publication strategy.

Compares three GA progression rate strategies for predicting time-to-foveal-involvement:

  1. Fixed rate: 73 µm/year (Keenan et al.)
  2. Manual individualized rate: derived from physician spreadsheet data.
  3. DL-derived individualized rate: derived from Atrophy Advisor measurements.

Input CSV columns required:
    patient_id, baseline_date, baseline_distance_um, followup_date,
    followup_distance_um, observed_foveal_involvement_date (ISO YYYY-MM-DD or YYYY-MM),
    [optional] dl_baseline_distance_um, dl_followup_distance_um

Usage:
    PYTHONPATH=. venv/bin/python3 scripts/predictive_validation.py \\
        --csv path/to/progression_data.csv \\
        [--output-dir test_validation/predictive_plots]

If no CSV is provided, the script prints instructions and exits.

When N < 20 patients progressed to foveal involvement, the results are flagged
as exploratory (suitable for Abstract 1 secondary analysis rather than standalone abstract).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXED_RATE_UM_PER_YEAR = 73.0
DAYS_PER_YEAR = 365.25

RESULT_FIELDS = [
    "patient_id", "followup_date", "observed_date", "fu_dist_um",
    "manual_rate_um_yr", "dl_rate_um_yr",
    "pred_fixed_date", "pred_manual_date", "pred_dl_date",
    "err_fixed_yrs", "err_manual_yrs", "err_dl_yrs",
]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_date_flexible(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM (defaults to 1st of month)."""
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def years_between(d1: date, d2: date) -> float:
    return (d2 - d1).days / DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------


def compute_rate_um_per_year(
    baseline_distance_um: float,
    followup_distance_um: float,
    elapsed_years: float,
) -> Optional[float]:
    """Linear rate: positive = approaching fovea."""
    if elapsed_years <= 0:
        return None
    change = baseline_distance_um - followup_distance_um
    if change <= 0:
        return None  # Not progressing toward fovea
    return change / elapsed_years


def predict_involvement_date(
    followup_date: date,
    followup_distance_um: float,
    rate_um_per_year: float,
) -> Optional[date]:
    if rate_um_per_year <= 0:
        return None
    years_to_involvement = followup_distance_um / rate_um_per_year
    days_to_involvement = years_to_involvement * DAYS_PER_YEAR
    try:
        return followup_date + timedelta(days=days_to_involvement)
    except OverflowError:
        return None


def prediction_error_years(predicted: Optional[date], observed: date) -> Optional[float]:
    if predicted is None:
        return None
    return years_between(observed, predicted)  # positive = predicted later than observed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_progression_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _f(row: Dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _d(row: Dict, key: str) -> Optional[date]:
    return parse_date_flexible(row.get(key, ""))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def run_analysis(rows: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    For each patient row, compute predictions from all three methods.
    Returns (results_list, summary_stats_dict).
    """
    results = []
    for row in rows:
        pid = row.get("patient_id", row.get("image_id", "unknown"))
        bl_date = _d(row, "baseline_date")
        fu_date = _d(row, "followup_date")
        obs_date = _d(row, "observed_foveal_involvement_date")
        bl_dist = _f(row, "baseline_distance_um")
        fu_dist = _f(row, "followup_distance_um")
        dl_bl_dist = _f(row, "dl_baseline_distance_um")
        dl_fu_dist = _f(row, "dl_followup_distance_um")

        if obs_date is None:
            continue  # No observed date → skip (not known to have progressed)
        if fu_date is None or fu_dist is None:
            continue

        elapsed_years = years_between(bl_date, fu_date) if bl_date else None

        # 1. Fixed rate (73 µm/year from Keenan et al.)
        pred_fixed = predict_involvement_date(fu_date, fu_dist, FIXED_RATE_UM_PER_YEAR)
        err_fixed = prediction_error_years(pred_fixed, obs_date)

        # 2. Manual individualized rate
        manual_rate = None
        pred_manual = None
        err_manual = None
        if bl_dist is not None and elapsed_years and elapsed_years > 0:
            manual_rate = compute_rate_um_per_year(bl_dist, fu_dist, elapsed_years)
            if manual_rate is None:
                print(f"  NOTE {pid}: manual rate not computed (no progression or missing baseline)")
            else:
                pred_manual = predict_involvement_date(fu_date, fu_dist, manual_rate)
                err_manual = prediction_error_years(pred_manual, obs_date)

        # 3. DL-derived individualized rate
        dl_rate = None
        pred_dl = None
        err_dl = None
        if dl_bl_dist is not None and dl_fu_dist is not None and elapsed_years and elapsed_years > 0:
            dl_rate = compute_rate_um_per_year(dl_bl_dist, dl_fu_dist, elapsed_years)
            if dl_rate is None:
                print(f"  NOTE {pid}: DL rate not computed (no DL progression or missing DL baseline)")
            else:
                pred_dl = predict_involvement_date(fu_date, dl_fu_dist, dl_rate)
                err_dl = prediction_error_years(pred_dl, obs_date)

        results.append({
            "patient_id": pid,
            "followup_date": fu_date.isoformat() if fu_date else "",
            "observed_date": obs_date.isoformat(),
            "fu_dist_um": fu_dist,
            "manual_rate_um_yr": manual_rate,
            "dl_rate_um_yr": dl_rate,
            "pred_fixed_date": pred_fixed.isoformat() if pred_fixed else "unpredictable",
            "pred_manual_date": pred_manual.isoformat() if pred_manual else "unpredictable",
            "pred_dl_date": pred_dl.isoformat() if pred_dl else "unpredictable",
            "err_fixed_yrs": err_fixed,
            "err_manual_yrs": err_manual,
            "err_dl_yrs": err_dl,
        })

    # Summary statistics
    def _summarize(errs: List[float], label: str) -> Dict:
        if not errs:
            return {"n": 0, "label": label}
        arr = np.array(errs)
        return {
            "label": label,
            "n": len(arr),
            "mae_yrs": float(np.mean(np.abs(arr))),
            "rmse_yrs": float(math.sqrt(np.mean(arr ** 2))),
            "bias_yrs": float(np.mean(arr)),
            "sd_yrs": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median_abs_yrs": float(np.median(np.abs(arr))),
        }

    errs_fixed = [r["err_fixed_yrs"] for r in results if r["err_fixed_yrs"] is not None]
    errs_manual = [r["err_manual_yrs"] for r in results if r["err_manual_yrs"] is not None]
    errs_dl = [r["err_dl_yrs"] for r in results if r["err_dl_yrs"] is not None]

    summary = {
        "n_total": len(results),
        "fixed": _summarize(errs_fixed, "Fixed 73 µm/yr"),
        "manual": _summarize(errs_manual, "Manual individualized"),
        "dl": _summarize(errs_dl, "DL-derived individualized"),
        "power_note": (
            "EXPLORATORY: fewer than 20 progressors — fold into Abstract 1 secondary analysis"
            if len(results) < 20
            else f"n={len(results)} progressors — sufficient for standalone Abstract 2"
        ),
    }

    # Paired comparisons (manual vs DL) where both have errors
    paired_manual_dl = [(r["err_manual_yrs"], r["err_dl_yrs"]) for r in results
                        if r["err_manual_yrs"] is not None and r["err_dl_yrs"] is not None]
    if len(paired_manual_dl) >= 3:
        abs_manual = np.array([abs(a) for a, _ in paired_manual_dl])
        abs_dl = np.array([abs(b) for _, b in paired_manual_dl])
        t_stat, p_val = stats.wilcoxon(abs_manual, abs_dl) if len(paired_manual_dl) >= 6 else (None, None)
        summary["paired_comparison"] = {
            "n": len(paired_manual_dl),
            "wilcoxon_stat": float(t_stat) if t_stat is not None else None,
            "wilcoxon_p": float(p_val) if p_val is not None else None,
            "manual_mae": float(np.mean(abs_manual)),
            "dl_mae": float(np.mean(abs_dl)),
        }

    return results, summary


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_results_csv(results: List[Dict], out_path: Path) -> None:
    if not results:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Results CSV → {out_path.name}")


def write_summary_txt(summary: Dict, out_path: Path) -> None:
    lines = [
        "Predictive Validation Summary",
        "=" * 50,
        f"N progressors: {summary['n_total']}",
        f"Power note: {summary['power_note']}",
        "",
    ]
    for key in ("fixed", "manual", "dl"):
        s = summary[key]
        if s.get("n", 0) == 0:
            lines.append(f"{s.get('label', key)}: no data")
            continue
        lines += [
            f"{s['label']}:",
            f"  n={s['n']}",
            f"  MAE  = {s['mae_yrs']:.2f} yrs",
            f"  RMSE = {s['rmse_yrs']:.2f} yrs",
            f"  Bias = {s['bias_yrs']:+.2f} yrs (positive = predicted later)",
            f"  SD   = {s['sd_yrs']:.2f} yrs",
            f"  Med |err| = {s['median_abs_yrs']:.2f} yrs",
            "",
        ]

    if "paired_comparison" in summary:
        pc = summary["paired_comparison"]
        lines += [
            "Paired comparison (Manual vs DL):",
            f"  n={pc['n']}",
            f"  Manual MAE = {pc['manual_mae']:.2f} yrs",
            f"  DL MAE    = {pc['dl_mae']:.2f} yrs",
        ]
        if pc.get("wilcoxon_p") is not None:
            lines.append(f"  Wilcoxon p = {pc['wilcoxon_p']:.4f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary text → {out_path.name}")


def plot_mae_comparison(summary: Dict, out_path: Path) -> None:
    methods = []
    maes = []
    for key, color in [("fixed", "#E05C5C"), ("manual", "#4C9BE8"), ("dl", "#6CBF6C")]:
        s = summary[key]
        if s.get("n", 0) > 0:
            methods.append(s["label"])
            maes.append(s["mae_yrs"])

    if len(methods) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, maes, color=["#E05C5C", "#4C9BE8", "#6CBF6C"][:len(methods)],
                  width=0.5, edgecolor="white", zorder=3)
    ax.set_ylabel("MAE in predicted foveal involvement (years)", fontsize=11)
    ax.set_title("Prediction Accuracy: Three Rate Strategies", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f} yrs", ha="center", va="bottom", fontsize=10, fontweight="bold")

    note = f"n={summary['n_total']} progressors | {summary['power_note'][:60]}"
    ax.text(0.5, -0.15, note, transform=ax.transAxes, ha="center", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path.name}")


def plot_prediction_scatter(results: List[Dict], out_path: Path) -> None:
    """Predicted vs observed foveal involvement date (each method)."""
    methods = [
        ("err_fixed_yrs", "Fixed 73 µm/yr", "#E05C5C", "o"),
        ("err_manual_yrs", "Manual rate", "#4C9BE8", "s"),
        ("err_dl_yrs", "DL rate", "#6CBF6C", "^"),
    ]

    obs_dates = []
    for r in results:
        d = parse_date_flexible(r["observed_date"])
        if d:
            obs_dates.append((r, d.year + d.timetuple().tm_yday / 365.25))
        else:
            obs_dates.append((r, None))

    fig, ax = plt.subplots(figsize=(8, 6))

    for err_key, label, color, marker in methods:
        xs, ys = [], []
        for r, obs_yr in obs_dates:
            err = r.get(err_key)
            if err is not None and obs_yr is not None:
                xs.append(obs_yr)
                ys.append(obs_yr + err)  # predicted year = observed + error
        if xs:
            ax.scatter(xs, ys, s=50, alpha=0.7, color=color, marker=marker, label=label, zorder=3)

    if obs_dates:
        min_yr = min(d for _, d in obs_dates if d is not None) - 0.5
        max_yr = max(d for _, d in obs_dates if d is not None) + 0.5
        ax.plot([min_yr, max_yr], [min_yr, max_yr], "k--", lw=1, label="Perfect prediction")

    ax.set_xlabel("Observed foveal involvement (year)", fontsize=11)
    ax.set_ylabel("Predicted foveal involvement (year)", fontsize=11)
    ax.set_title("Predicted vs Observed Foveal Involvement Date", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predictive validation: three progression rate strategies.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Input CSV with patient progression data.")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "test_validation" / "predictive_plots")
    return p.parse_args()


TEMPLATE_CSV = """\
patient_id,baseline_date,baseline_distance_um,followup_date,followup_distance_um,observed_foveal_involvement_date,dl_baseline_distance_um,dl_followup_distance_um
P001,2022-01-15,850,2023-03-10,550,2025-06-01,900,580
P002,2021-06-01,620,2022-09-20,310,2024-04-15,650,330
"""


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_path = PROJECT_ROOT / "test_validation" / "progression_data_template.csv"

    if args.csv is None:
        print("No input CSV provided.")
        print()
        print("This script requires a CSV with patient progression data.")
        print("Required columns:")
        print("  patient_id                       — unique identifier")
        print("  baseline_date                    — YYYY-MM-DD")
        print("  baseline_distance_um             — manual GA-to-fovea at baseline")
        print("  followup_date                    — YYYY-MM-DD of most recent measurement")
        print("  followup_distance_um             — manual GA-to-fovea at followup")
        print("  observed_foveal_involvement_date — YYYY-MM-DD (only patients who progressed)")
        print("  dl_baseline_distance_um          — [optional] DL-measured baseline distance")
        print("  dl_followup_distance_um          — [optional] DL-measured followup distance")
        print()
        print(f"A template CSV has been written to: {template_path}")

        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(TEMPLATE_CSV, encoding="utf-8")

        print()
        print("Run with:")
        print(f"  PYTHONPATH=. venv/bin/python3 scripts/predictive_validation.py --csv {template_path}")
        return 0

    if not args.csv.exists():
        print(f"ERROR: File not found: {args.csv}")
        return 1

    print(f"Loading {args.csv} …")
    rows = load_progression_csv(args.csv)
    print(f"  {len(rows)} rows loaded")

    results, summary = run_analysis(rows)
    print(f"  {summary['n_total']} progressors with valid dates")
    print(f"  Power note: {summary['power_note']}")

    if summary["n_total"] == 0:
        print("No patients with observed foveal involvement date found. Cannot analyse.")
        return 0

    print("\nResults:")
    for key in ("fixed", "manual", "dl"):
        s = summary[key]
        if s.get("n", 0) > 0:
            print(f"  {s['label']:30s}  MAE={s['mae_yrs']:.2f} yr  RMSE={s['rmse_yrs']:.2f} yr  bias={s['bias_yrs']:+.2f} yr")

    # Outputs
    write_results_csv(results, args.output_dir / "predictive_results.csv")
    write_summary_txt(summary, args.output_dir / "predictive_summary.txt")
    plot_mae_comparison(summary, args.output_dir / "fig_predictive_mae.png")
    if results:
        plot_prediction_scatter(results, args.output_dir / "fig_predictive_scatter.png")

    print(f"\nOutputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
