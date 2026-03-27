#!/usr/bin/env python3
"""
Bland-Altman Analysis for Atrophy Advisor publication pipeline.

Reads bland_altman_data.csv (produced by batch_bland_altman_validation.py)
and generates:

  1. Primary Bland-Altman plot: DL vs GT distance (um) — bias, 95% LOA, CIs.
  2. Error decomposition: disc-calibration vs fovea-localization vs GA-boundary.
  3. Disc calibration sub-analysis: correction factor (first 25 images → apply to last 25).
  4. Component-level Bland-Altman: disc height (px), fovea position (px), GA point (px).
  5. Summary statistics table as CSV and LaTeX.

All figures saved at 300 DPI, journal-ready.

Usage:
    PYTHONPATH=. venv/bin/python3 scripts/bland_altman_analysis.py \\
        [--csv test_validation/bland_altman_data.csv] \\
        [--output-dir test_validation/bland_altman_plots]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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

DISC_DIAMETER_MICRONS = 1800.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv(csv_path: Path) -> List[Dict]:
    """Load and type-cast the CSV produced by batch_bland_altman_validation.py."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _f(row: Dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def get_distance_pairs(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (gt_dist_um, assisted_dist_um, image_ids) for non-foveal-involvement images
    that have valid distance measurements in both columns (Manual vs Assisted).
    """
    gt_vals, dl_vals, ids = [], [], []
    for r in rows:
        if r.get("foveal_involvement") == "true":
            continue
        gt = _f(r, "gt_distance_um")
        dl = _f(r, "dl_distance_um")
        if gt is None or dl is None:
            continue
        gt_vals.append(gt)
        dl_vals.append(dl)
        ids.append(r["image_id"])
    return np.array(gt_vals), np.array(dl_vals), ids


def get_autonomous_distance_pairs(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (gt_dist_um, auto_dist_um, image_ids) for non-foveal-involvement images
    that have valid distance measurements in both columns (Manual vs Autonomous).
    """
    gt_vals, auto_vals, ids = [], [], []
    for r in rows:
        if r.get("foveal_involvement") == "true":
            continue
        gt = _f(r, "gt_distance_um")
        auto = _f(r, "auto_distance_um")
        if gt is None or auto is None:
            continue
        gt_vals.append(gt)
        auto_vals.append(auto)
        ids.append(r["image_id"])
    return np.array(gt_vals), np.array(auto_vals), ids


# ---------------------------------------------------------------------------
# Bland-Altman statistics
# ---------------------------------------------------------------------------


def bland_altman_stats(gt: np.ndarray, dl: np.ndarray) -> Dict:
    """
    Compute full Bland-Altman statistics (DL − GT convention).

    Returns dict with:
        n, bias, sd, loa_lo, loa_hi,
        ci_bias_lo, ci_bias_hi,
        ci_loa_lo_lo, ci_loa_lo_hi, ci_loa_hi_lo, ci_loa_hi_hi,
        prop_bias_slope, prop_bias_intercept, prop_bias_p, prop_bias_r2,
        pearson_r, pearson_p, pearson_r_ci_lo, pearson_r_ci_hi,
        icc, icc_ci_lo, icc_ci_hi, mae, rmse
    """
    diffs = dl - gt
    means = (dl + gt) / 2.0
    n = len(diffs)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd

    # 95% CIs using t-distribution (Bland & Altman 1999 formula)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    se_bias = sd / math.sqrt(n)
    se_loa = math.sqrt(3 * sd ** 2 / n)

    ci_bias_lo = bias - t_crit * se_bias
    ci_bias_hi = bias + t_crit * se_bias
    ci_loa_lo_lo = loa_lo - t_crit * se_loa
    ci_loa_lo_hi = loa_lo + t_crit * se_loa
    ci_loa_hi_lo = loa_hi - t_crit * se_loa
    ci_loa_hi_hi = loa_hi + t_crit * se_loa

    # Proportional bias: regress diffs on means
    slope, intercept, r_val, p_val, _ = stats.linregress(means, diffs)

    # Pearson r with 95% CI via Fisher's z-transform
    pearson_r, pearson_p = stats.pearsonr(gt, dl)
    z = math.atanh(float(pearson_r))
    se_z = 1.0 / math.sqrt(max(n - 3, 1))
    pearson_r_ci_lo = float(math.tanh(z - 1.96 * se_z))
    pearson_r_ci_hi = float(math.tanh(z + 1.96 * se_z))

    # ICC(2,1) — two-way mixed, absolute agreement (McGraw & Wong 1996)
    # Based on one-way ANOVA components between the two-column matrix [gt, dl]
    data = np.column_stack([gt, dl])  # shape (n, 2)
    grand_mean = float(np.mean(data))
    ss_between = 2.0 * float(np.sum((np.mean(data, axis=1) - grand_mean) ** 2))
    ss_within = float(np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2))
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / n  # k=2 raters, within-subject error df = n*(k-1) = n
    # ICC(A,1) absolute agreement, single measurement:
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    icc = float(max(0.0, icc))

    # 95% CI for ICC via F distribution (Shrout & Fleiss method)
    f_lower = ms_between / ms_within / float(stats.f.ppf(0.975, dfn=n - 1, dfd=n))
    f_upper = ms_between / ms_within / float(stats.f.ppf(0.025, dfn=n - 1, dfd=n))
    icc_ci_lo = float(max(0.0, (f_lower - 1) / (f_lower + 1)))
    icc_ci_hi = float(min(1.0, (f_upper - 1) / (f_upper + 1)))

    mae = float(np.mean(np.abs(diffs)))
    rmse = float(math.sqrt(np.mean(diffs ** 2)))

    return {
        "n": n,
        "bias": bias,
        "sd": sd,
        "loa_lo": loa_lo,
        "loa_hi": loa_hi,
        "ci_bias_lo": ci_bias_lo,
        "ci_bias_hi": ci_bias_hi,
        "ci_loa_lo_lo": ci_loa_lo_lo,
        "ci_loa_lo_hi": ci_loa_lo_hi,
        "ci_loa_hi_lo": ci_loa_hi_lo,
        "ci_loa_hi_hi": ci_loa_hi_hi,
        "prop_bias_slope": float(slope),
        "prop_bias_intercept": float(intercept),
        "prop_bias_p": float(p_val),
        "prop_bias_r2": float(r_val ** 2),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_r_ci_lo": pearson_r_ci_lo,
        "pearson_r_ci_hi": pearson_r_ci_hi,
        "icc": icc,
        "icc_ci_lo": icc_ci_lo,
        "icc_ci_hi": icc_ci_hi,
        "mae": mae,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _setup_fig(title: str) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def plot_bland_altman(
    gt: np.ndarray,
    dl: np.ndarray,
    ids: List[str],
    ba: Dict,
    title: str,
    x_label: str = "Mean of GT and DL (µm)",
    y_label: str = "DL − GT (µm)",
    out_path: Optional[Path] = None,
) -> None:
    diffs = dl - gt
    means = (dl + gt) / 2.0

    fig, ax = _setup_fig(title)

    ax.scatter(means, diffs, s=40, alpha=0.7, color="#4C9BE8", zorder=3, label="Images")

    # Proportional bias trend line
    if ba["prop_bias_p"] < 0.10:
        x_fit = np.linspace(means.min(), means.max(), 200)
        y_fit = ba["prop_bias_slope"] * x_fit + ba["prop_bias_intercept"]
        ax.plot(x_fit, y_fit, "k--", lw=1.2, label=f"Trend (p={ba['prop_bias_p']:.3f})")

    # Bias line
    ax.axhline(ba["bias"], color="#E05C5C", lw=2, label=f"Bias {ba['bias']:+.1f} µm")
    ax.fill_between(
        [means.min() - 50, means.max() + 50],
        [ba["ci_bias_lo"]] * 2, [ba["ci_bias_hi"]] * 2,
        alpha=0.15, color="#E05C5C",
    )

    # LOA lines
    for loa, ci_lo, ci_hi, label in [
        (ba["loa_lo"], ba["ci_loa_lo_lo"], ba["ci_loa_lo_hi"], f"−1.96SD {ba['loa_lo']:.1f}"),
        (ba["loa_hi"], ba["ci_loa_hi_lo"], ba["ci_loa_hi_hi"], f"+1.96SD {ba['loa_hi']:.1f}"),
    ]:
        ax.axhline(loa, color="#6CBF6C", lw=1.8, linestyle="--", label=label)
        ax.fill_between(
            [means.min() - 50, means.max() + 50],
            [ci_lo] * 2, [ci_hi] * 2,
            alpha=0.12, color="#6CBF6C",
        )

    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")

    stats_text = (
        f"n={ba['n']}  MAE={ba['mae']:.1f} µm\n"
        f"Bias {ba['bias']:+.1f} µm  [95%CI {ba['ci_bias_lo']:.1f}, {ba['ci_bias_hi']:.1f}]\n"
        f"LoA [{ba['loa_lo']:.1f}, {ba['loa_hi']:.1f}] µm"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9),
    )

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path.name}")
    plt.close(fig)


def plot_scatter(
    gt: np.ndarray,
    dl: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Optional[Path] = None,
) -> None:
    fig, ax = _setup_fig(title)
    ax.scatter(gt, dl, s=40, alpha=0.7, color="#4C9BE8", zorder=3)

    lim_lo = min(gt.min(), dl.min()) - 50
    lim_hi = max(gt.max(), dl.max()) + 50
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="Identity")

    slope, intercept, r_val, p_val, _ = stats.linregress(gt, dl)
    x_fit = np.linspace(lim_lo, lim_hi, 200)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5,
            label=f"Regression (R²={r_val**2:.3f})")

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.legend(fontsize=9)

    text = f"R²={r_val**2:.3f}  p={p_val:.3g}\nPearson r={r_val:.3f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))
    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path.name}")
    plt.close(fig)


def plot_error_decomposition(
    disc_contrib: np.ndarray,
    fovea_contrib: np.ndarray,
    ga_contrib: np.ndarray,
    out_path: Optional[Path] = None,
) -> None:
    """Bar chart: mean absolute contribution of each error source."""
    labels = ["Disc\ncalibration", "Fovea\nlocalization", "GA boundary\nlocalization"]
    values = [float(np.mean(np.abs(disc_contrib))),
              float(np.mean(np.abs(fovea_contrib))),
              float(np.mean(np.abs(ga_contrib)))]
    colors = ["#E05C5C", "#4C9BE8", "#F0A030"]

    fig, ax = _setup_fig("Error Decomposition by Source")
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", zorder=3)
    ax.set_ylabel("Mean |contribution| (µm)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f} µm",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    total = sum(values)
    for bar, val in zip(bars, values):
        pct = 100 * val / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.0f}%",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path.name}")
    plt.close(fig)


def plot_calibration_comparison(
    uncalibrated_mae: float,
    calibrated_mae: float,
    correction_factor: float,
    n_train: int,
    n_val: int,
    out_path: Optional[Path] = None,
) -> None:
    fig, ax = _setup_fig("Disc Calibration Sub-analysis")
    bars = ax.bar(
        ["Uncalibrated DL\n(validation split)", "Calibrated DL\n(validation split)"],
        [uncalibrated_mae, calibrated_mae],
        color=["#E05C5C", "#6CBF6C"],
        width=0.45,
        edgecolor="white",
        zorder=3,
    )
    for bar, val in zip(bars, [uncalibrated_mae, calibrated_mae]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f} µm",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("MAE in distance (µm)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    improvement = uncalibrated_mae - calibrated_mae
    note = (
        f"Correction factor: {correction_factor:.3f} (from first {n_train} images)\n"
        f"Applied to {n_val} validation images  |  MAE reduction: {improvement:.1f} µm"
    )
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error decomposition computation
# ---------------------------------------------------------------------------


def compute_error_decomposition(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    For each non-foveal-involvement image with all required columns, compute:

    disc_contrib: error attributable to disc miscalibration
        = dl_distance_px * (1800/dl_disc_h) - dl_distance_px * (1800/gt_disc_h)

    fovea_contrib: error attributable to fovea localization
        = Euclidean(dl_fovea, dl_ga) * (1800/gt_disc_h)
          - Euclidean(gt_fovea, dl_ga) * (1800/gt_disc_h)
        i.e., how much the distance changes if we swap to GT fovea (with GT disc scale)

    ga_contrib: error attributable to GA localization
        = Euclidean(gt_fovea, dl_ga) * (1800/gt_disc_h)
          - Euclidean(gt_fovea, gt_ga) * (1800/gt_disc_h)
        = (gt_fovea→dl_ga) - (gt_fovea→gt_ga) in um

    Returns (disc_contrib, fovea_contrib, ga_contrib, image_ids).
    """
    disc_contribs, fovea_contribs, ga_contribs, ids = [], [], [], []

    for r in rows:
        if r.get("foveal_involvement") == "true":
            continue

        req = ["dl_distance_px", "dl_disc_height_px", "gt_disc_height_px",
               "dl_fovea_x", "dl_fovea_y", "gt_fovea_x", "gt_fovea_y",
               "dl_ga_x", "dl_ga_y", "gt_ga_x", "gt_ga_y"]
        vals = {k: _f(r, k) for k in req}
        if any(v is None for v in vals.values()):
            continue

        dl_dist_px = vals["dl_distance_px"]
        dl_disc_h = vals["dl_disc_height_px"]
        gt_disc_h = vals["gt_disc_height_px"]

        # Disc calibration contribution
        dl_dist_with_dl_disc = dl_dist_px * (DISC_DIAMETER_MICRONS / dl_disc_h)
        dl_dist_with_gt_disc = dl_dist_px * (DISC_DIAMETER_MICRONS / gt_disc_h)
        disc_contrib = dl_dist_with_dl_disc - dl_dist_with_gt_disc

        # Fovea localization contribution (using GT disc scale to isolate)
        dl_fovea = np.array([vals["dl_fovea_x"], vals["dl_fovea_y"]])
        gt_fovea = np.array([vals["gt_fovea_x"], vals["gt_fovea_y"]])
        dl_ga = np.array([vals["dl_ga_x"], vals["dl_ga_y"]])
        gt_ga = np.array([vals["gt_ga_x"], vals["gt_ga_y"]])

        scale = DISC_DIAMETER_MICRONS / gt_disc_h
        dist_dl_fovea_dl_ga = float(np.linalg.norm(dl_fovea - dl_ga)) * scale
        dist_gt_fovea_dl_ga = float(np.linalg.norm(gt_fovea - dl_ga)) * scale
        fovea_contrib = dist_dl_fovea_dl_ga - dist_gt_fovea_dl_ga

        # GA localization contribution
        dist_gt_fovea_gt_ga = float(np.linalg.norm(gt_fovea - gt_ga)) * scale
        ga_contrib = dist_gt_fovea_dl_ga - dist_gt_fovea_gt_ga

        disc_contribs.append(disc_contrib)
        fovea_contribs.append(fovea_contrib)
        ga_contribs.append(ga_contrib)
        ids.append(r["image_id"])

    return (
        np.array(disc_contribs),
        np.array(fovea_contribs),
        np.array(ga_contribs),
        ids,
    )


# ---------------------------------------------------------------------------
# Disc calibration sub-analysis
# ---------------------------------------------------------------------------


def disc_calibration_subanalysis(rows: List[Dict]) -> Dict:
    """
    Split non-foveal-involvement rows with valid distance + disc data into
    first 25 (training) and remaining (validation).

    Derive correction_factor = mean(gt_disc_h / dl_disc_h) from training split.
    Apply to validation: adjusted_dl_um = dl_dist_um * correction_factor.
    Report uncalibrated MAE vs calibrated MAE on validation split.
    """
    eligible = []
    for r in rows:
        if r.get("foveal_involvement") == "true":
            continue
        gt_um = _f(r, "gt_distance_um")
        dl_um = _f(r, "dl_distance_um")
        gt_disc = _f(r, "gt_disc_height_px")
        dl_disc = _f(r, "dl_disc_height_px")
        if None in (gt_um, dl_um, gt_disc, dl_disc) or dl_disc == 0:
            continue
        eligible.append({
            "image_id": r["image_id"],
            "gt_um": gt_um,
            "dl_um": dl_um,
            "gt_disc": gt_disc,
            "dl_disc": dl_disc,
        })

    n_train = min(25, len(eligible) // 2)
    train = eligible[:n_train]
    val = eligible[n_train:]

    if not train or not val:
        return {"error": "Insufficient data for calibration sub-analysis"}

    # Correction factor
    ratios = [e["gt_disc"] / e["dl_disc"] for e in train]
    cf = float(np.mean(ratios))

    # Uncalibrated MAE on validation split
    uncal_errors = [abs(e["dl_um"] - e["gt_um"]) for e in val]
    uncal_mae = float(np.mean(uncal_errors))

    # Calibrated: scale dl_um by correction factor
    # dl_distance_um = dl_distance_px * (1800 / dl_disc_h)
    # calibrated = dl_distance_px * (1800 / dl_disc_h) * cf
    #            = dl_um * cf
    cal_errors = [abs(e["dl_um"] * cf - e["gt_um"]) for e in val]
    cal_mae = float(np.mean(cal_errors))

    return {
        "n_train": n_train,
        "n_val": len(val),
        "correction_factor": cf,
        "uncalibrated_mae_um": uncal_mae,
        "calibrated_mae_um": cal_mae,
        "mae_improvement_um": uncal_mae - cal_mae,
        "mae_improvement_pct": 100 * (uncal_mae - cal_mae) / max(uncal_mae, 1e-9),
        "train_disc_ratios_mean": cf,
        "train_disc_ratios_sd": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Component-level stats (disc height, fovea, GA point)
# ---------------------------------------------------------------------------


def component_stats(rows: List[Dict]) -> Dict:
    disc_errors, fovea_errors, ga_errors = [], [], []

    for r in rows:
        gt_disc = _f(r, "gt_disc_height_px")
        dl_disc = _f(r, "dl_disc_height_px")
        if gt_disc is not None and dl_disc is not None:
            disc_errors.append(dl_disc - gt_disc)

        gt_fx, gt_fy = _f(r, "gt_fovea_x"), _f(r, "gt_fovea_y")
        dl_fx, dl_fy = _f(r, "dl_fovea_x"), _f(r, "dl_fovea_y")
        if None not in (gt_fx, gt_fy, dl_fx, dl_fy):
            fovea_errors.append(float(np.hypot(dl_fx - gt_fx, dl_fy - gt_fy)))

        if r.get("foveal_involvement") != "true":
            gt_gx, gt_gy = _f(r, "gt_ga_x"), _f(r, "gt_ga_y")
            dl_gx, dl_gy = _f(r, "dl_ga_x"), _f(r, "dl_ga_y")
            if None not in (gt_gx, gt_gy, dl_gx, dl_gy):
                ga_errors.append(float(np.hypot(dl_gx - gt_gx, dl_gy - gt_gy)))

    result: Dict = {}
    for name, errs in [("disc_height_px", disc_errors), ("fovea_px", fovea_errors), ("ga_point_px", ga_errors)]:
        if not errs:
            result[name] = {"n": 0}
            continue
        arr = np.array(errs)
        result[name] = {
            "n": len(arr),
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(math.sqrt(np.mean(arr ** 2))),
            "mean_signed": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(np.abs(arr))),
            "p90": float(np.percentile(np.abs(arr), 90)),
        }
    return result


# ---------------------------------------------------------------------------
# Summary CSV + LaTeX
# ---------------------------------------------------------------------------


def write_summary_csv(
    ba_assisted: Dict,
    calibration: Dict,
    components: Dict,
    decomp_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
    ba_autonomous: Optional[Dict] = None,
) -> None:
    disc_c, fovea_c, ga_c = decomp_arrays

    def _ba_block(label: str, ba: Dict) -> List:
        return [
            [f"--- {label} ---"],
            ["N (distance pairs)", ba.get("n", "")],
            ["Bias (method−Manual, µm)", f"{ba.get('bias', 0):.2f}"],
            ["SD of differences (µm)", f"{ba.get('sd', 0):.2f}"],
            ["95% LOA lower (µm)", f"{ba.get('loa_lo', 0):.2f}"],
            ["95% LOA upper (µm)", f"{ba.get('loa_hi', 0):.2f}"],
            ["95% CI bias lower (µm)", f"{ba.get('ci_bias_lo', 0):.2f}"],
            ["95% CI bias upper (µm)", f"{ba.get('ci_bias_hi', 0):.2f}"],
            ["MAE (µm)", f"{ba.get('mae', 0):.2f}"],
            ["RMSE (µm)", f"{ba.get('rmse', 0):.2f}"],
            ["Pearson r", f"{ba.get('pearson_r', 0):.3f}"],
            ["Pearson r 95% CI", f"[{ba.get('pearson_r_ci_lo', 0):.3f}, {ba.get('pearson_r_ci_hi', 0):.3f}]"],
            ["Pearson p-value", f"{ba.get('pearson_p', 1):.4f}"],
            ["ICC(2,1) absolute agreement", f"{ba.get('icc', 0):.3f}"],
            ["ICC 95% CI", f"[{ba.get('icc_ci_lo', 0):.3f}, {ba.get('icc_ci_hi', 0):.3f}]"],
            ["Proportional bias slope", f"{ba.get('prop_bias_slope', 0):.4f}"],
            ["Proportional bias p-value", f"{ba.get('prop_bias_p', 1):.4f}"],
            ["Proportional bias R²", f"{ba.get('prop_bias_r2', 0):.4f}"],
            [],
        ]

    rows: List = [["Metric", "Value"]]
    rows += _ba_block("Assisted (click-anchored) vs Manual", ba_assisted)
    if ba_autonomous is not None:
        rows += _ba_block("Autonomous (no-click) vs Manual", ba_autonomous)
    rows += [
        ["Error decomposition (mean |contribution|, µm)"],
        ["Disc calibration", f"{float(np.mean(np.abs(disc_c))):.2f}" if len(disc_c) else "N/A"],
        ["Fovea localization", f"{float(np.mean(np.abs(fovea_c))):.2f}" if len(fovea_c) else "N/A"],
        ["GA boundary", f"{float(np.mean(np.abs(ga_c))):.2f}" if len(ga_c) else "N/A"],
        [],
        ["Disc calibration sub-analysis"],
    ]
    if "error" not in calibration:
        rows += [
            ["N training", calibration.get("n_train", "")],
            ["N validation", calibration.get("n_val", "")],
            ["Correction factor", f"{calibration.get('correction_factor', ''):.4f}"],
            ["Uncalibrated MAE (µm)", f"{calibration.get('uncalibrated_mae_um', ''):.2f}"],
            ["Calibrated MAE (µm)", f"{calibration.get('calibrated_mae_um', ''):.2f}"],
            ["MAE improvement (µm)", f"{calibration.get('mae_improvement_um', ''):.2f}"],
            ["MAE improvement (%)", f"{calibration.get('mae_improvement_pct', ''):.1f}"],
        ]
    else:
        rows.append(["Error", calibration.get("error", "")])

    rows += [[], ["Component-level statistics"]]
    for comp, label in [("disc_height_px", "Disc height (px)"), ("fovea_px", "Fovea (px)"), ("ga_point_px", "GA point (px)")]:
        s = components.get(comp, {})
        rows.append([label, f"n={s.get('n','?')} MAE={s.get('mae','?'):.2f} RMSE={s.get('rmse','?'):.2f} median={s.get('median','?'):.2f}" if s.get("n") else "insufficient data"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  Summary CSV → {out_path.name}")


def write_latex_table(
    ba_assisted: Dict,
    calibration: Dict,
    components: Dict,
    decomp_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
    ba_autonomous: Optional[Dict] = None,
) -> None:
    disc_c, fovea_c, ga_c = decomp_arrays

    def _v(val, fmt=".1f"):
        try:
            return format(float(val), fmt)
        except (TypeError, ValueError):
            return "N/A"

    def _ba_rows(ba: Dict) -> List[str]:
        return [
            rf"N (distance pairs) & {ba.get('n','')} & images \\",
            rf"Bias (method$-$Manual) & {_v(ba.get('bias'))} & µm \\",
            rf"SD of differences & {_v(ba.get('sd'))} & µm \\",
            rf"95\% LoA & [{_v(ba.get('loa_lo'))},\ {_v(ba.get('loa_hi'))}] & µm \\",
            rf"MAE & {_v(ba.get('mae'))} & µm \\",
            rf"RMSE & {_v(ba.get('rmse'))} & µm \\",
            rf"Pearson r & {_v(ba.get('pearson_r'), '.3f')} [95\%CI {_v(ba.get('pearson_r_ci_lo'), '.3f')}, {_v(ba.get('pearson_r_ci_hi'), '.3f')}] & (p={_v(ba.get('pearson_p'),'.3f')}) \\",
            rf"ICC(2,1) & {_v(ba.get('icc'), '.3f')} [95\%CI {_v(ba.get('icc_ci_lo'), '.3f')}, {_v(ba.get('icc_ci_hi'), '.3f')}] & \\",
            rf"Proportional bias slope & {_v(ba.get('prop_bias_slope'), '.3f')} & (p={_v(ba.get('prop_bias_p'),'.3f')}) \\",
        ]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Agreement Between Automated and Manual GA-to-Fovea Distance Measurements}",
        r"\label{tab:bland-altman}",
        r"\begin{tabular}{lll}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\",
        r"\hline",
        r"\multicolumn{3}{l}{\textit{Assisted (clinician click + DL segmentation) vs Manual}} \\",
        r"\hline",
    ] + _ba_rows(ba_assisted)

    if ba_autonomous is not None:
        lines += [
            r"\hline",
            r"\multicolumn{3}{l}{\textit{Autonomous (fully automated, no click) vs Manual}} \\",
            r"\hline",
        ] + _ba_rows(ba_autonomous)

    lines += [
        r"\hline",
        r"\multicolumn{3}{l}{\textit{Error decomposition (mean |contribution|)}} \\",
        rf"Disc calibration & {_v(float(np.mean(np.abs(disc_c))) if len(disc_c) else float('nan'))} & µm \\",
        rf"Fovea localization & {_v(float(np.mean(np.abs(fovea_c))) if len(fovea_c) else float('nan'))} & µm \\",
        rf"GA boundary & {_v(float(np.mean(np.abs(ga_c))) if len(ga_c) else float('nan'))} & µm \\",
    ]

    if "error" not in calibration:
        lines += [
            r"\hline",
            r"\multicolumn{3}{l}{\textit{Disc calibration sub-analysis}} \\",
            rf"Correction factor & {_v(calibration.get('correction_factor'), '.3f')} & (train n={calibration.get('n_train','?')}) \\",
            rf"Uncalibrated MAE & {_v(calibration.get('uncalibrated_mae_um'))} & µm \\",
            rf"Calibrated MAE & {_v(calibration.get('calibrated_mae_um'))} & µm \\",
        ]

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX table → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bland-Altman analysis for Atrophy Advisor.")
    p.add_argument("--csv", type=Path, default=PROJECT_ROOT / "test_validation" / "bland_altman_data.csv")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_validation" / "bland_altman_plots")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run batch_bland_altman_validation.py first.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} …")
    rows = load_csv(args.csv)
    print(f"  {len(rows)} total rows")

    gt_dist, dl_dist, dist_ids = get_distance_pairs(rows)
    gt_dist_auto, auto_dist, auto_ids = get_autonomous_distance_pairs(rows)
    n_pairs = len(gt_dist)
    n_auto_pairs = len(gt_dist_auto)
    n_involvement = sum(1 for r in rows if r.get("foveal_involvement") == "true")
    print(f"  Non-foveal-involvement rows with assisted distance data: {n_pairs}")
    print(f"  Non-foveal-involvement rows with autonomous distance data: {n_auto_pairs}")
    print(f"  Foveal involvement cases: {n_involvement}")

    if n_pairs < 3:
        print("ERROR: Fewer than 3 valid distance pairs — cannot compute Bland-Altman.")
        return 1

    # ------------------------------------------------------------------
    # 1. Primary Bland-Altman: Assisted vs Manual
    # ------------------------------------------------------------------
    print("\n1. Primary Bland-Altman: Assisted vs Manual (distance µm)…")
    ba_assisted = bland_altman_stats(gt_dist, dl_dist)
    print(
        f"   n={ba_assisted['n']}  bias={ba_assisted['bias']:+.1f}  SD={ba_assisted['sd']:.1f}  "
        f"LoA=[{ba_assisted['loa_lo']:.1f}, {ba_assisted['loa_hi']:.1f}]  MAE={ba_assisted['mae']:.1f} µm"
    )
    print(
        f"   Pearson r={ba_assisted['pearson_r']:.3f} [95%CI {ba_assisted['pearson_r_ci_lo']:.3f}, {ba_assisted['pearson_r_ci_hi']:.3f}]  "
        f"ICC={ba_assisted['icc']:.3f} [95%CI {ba_assisted['icc_ci_lo']:.3f}, {ba_assisted['icc_ci_hi']:.3f}]"
    )

    plot_bland_altman(
        gt_dist, dl_dist, dist_ids, ba_assisted,
        title="Bland-Altman: Assisted vs Manual GA-to-Fovea Distance",
        x_label="Mean of Manual and Assisted (µm)",
        y_label="Assisted − Manual (µm)",
        out_path=args.output_dir / "fig1_bland_altman_assisted.png",
    )
    plot_scatter(
        gt_dist, dl_dist,
        title="Assisted vs Manual Distance (µm) — Scatter",
        x_label="Manual (GT) distance (µm)",
        y_label="Assisted DL distance (µm)",
        out_path=args.output_dir / "fig1b_scatter_assisted.png",
    )

    # ------------------------------------------------------------------
    # 1c. Autonomous vs Manual Bland-Altman
    # ------------------------------------------------------------------
    ba_autonomous: Optional[Dict] = None
    if n_auto_pairs >= 3:
        print(f"\n1c. Autonomous vs Manual Bland-Altman (n={n_auto_pairs})…")
        ba_autonomous = bland_altman_stats(gt_dist_auto, auto_dist)
        print(
            f"   n={ba_autonomous['n']}  bias={ba_autonomous['bias']:+.1f}  SD={ba_autonomous['sd']:.1f}  "
            f"LoA=[{ba_autonomous['loa_lo']:.1f}, {ba_autonomous['loa_hi']:.1f}]  MAE={ba_autonomous['mae']:.1f} µm"
        )
        print(
            f"   Pearson r={ba_autonomous['pearson_r']:.3f} [95%CI {ba_autonomous['pearson_r_ci_lo']:.3f}, {ba_autonomous['pearson_r_ci_hi']:.3f}]  "
            f"ICC={ba_autonomous['icc']:.3f} [95%CI {ba_autonomous['icc_ci_lo']:.3f}, {ba_autonomous['icc_ci_hi']:.3f}]"
        )
        plot_bland_altman(
            gt_dist_auto, auto_dist, auto_ids, ba_autonomous,
            title="Bland-Altman: Autonomous vs Manual GA-to-Fovea Distance",
            x_label="Mean of Manual and Autonomous (µm)",
            y_label="Autonomous − Manual (µm)",
            out_path=args.output_dir / "fig1c_bland_altman_autonomous.png",
        )
        plot_scatter(
            gt_dist_auto, auto_dist,
            title="Autonomous vs Manual Distance (µm) — Scatter",
            x_label="Manual (GT) distance (µm)",
            y_label="Autonomous DL distance (µm)",
            out_path=args.output_dir / "fig1d_scatter_autonomous.png",
        )
    else:
        print(f"\n1c. Autonomous vs Manual: only {n_auto_pairs} pairs — skipping (need ≥3).")
        print("    Re-run batch_bland_altman_validation.py to generate autonomous GA data.")

    # ------------------------------------------------------------------
    # 2. Error decomposition
    # ------------------------------------------------------------------
    print("\n2. Error decomposition…")
    disc_c, fovea_c, ga_c, decomp_ids = compute_error_decomposition(rows)
    if len(disc_c) > 0:
        print(
            f"   n={len(disc_c)}  disc={np.mean(np.abs(disc_c)):.1f} µm  "
            f"fovea={np.mean(np.abs(fovea_c)):.1f} µm  ga={np.mean(np.abs(ga_c)):.1f} µm"
        )
        plot_error_decomposition(
            disc_c, fovea_c, ga_c,
            out_path=args.output_dir / "fig2_error_decomposition.png",
        )
    else:
        print("   Insufficient data for decomposition (need dl_ga_x/y in CSV).")

    # ------------------------------------------------------------------
    # 3. Disc calibration sub-analysis
    # ------------------------------------------------------------------
    print("\n3. Disc calibration sub-analysis…")
    calibration = disc_calibration_subanalysis(rows)
    if "error" not in calibration:
        print(
            f"   CF={calibration['correction_factor']:.3f}  "
            f"uncal MAE={calibration['uncalibrated_mae_um']:.1f} µm → "
            f"cal MAE={calibration['calibrated_mae_um']:.1f} µm "
            f"(-{calibration['mae_improvement_pct']:.1f}%)"
        )
        plot_calibration_comparison(
            calibration["uncalibrated_mae_um"],
            calibration["calibrated_mae_um"],
            calibration["correction_factor"],
            calibration["n_train"],
            calibration["n_val"],
            out_path=args.output_dir / "fig3_disc_calibration.png",
        )
    else:
        print(f"   {calibration['error']}")

    # ------------------------------------------------------------------
    # 4. Component-level: disc height BA plot
    # ------------------------------------------------------------------
    print("\n4. Component-level analysis…")
    components = component_stats(rows)

    # Disc height BA
    gt_disc_vals, dl_disc_vals = [], []
    for r in rows:
        gt_d, dl_d = _f(r, "gt_disc_height_px"), _f(r, "dl_disc_height_px")
        if gt_d is not None and dl_d is not None:
            gt_disc_vals.append(gt_d)
            dl_disc_vals.append(dl_d)

    if len(gt_disc_vals) >= 3:
        gt_disc_arr = np.array(gt_disc_vals)
        dl_disc_arr = np.array(dl_disc_vals)
        ba_disc = bland_altman_stats(gt_disc_arr, dl_disc_arr)
        print(
            f"   Disc height: n={ba_disc['n']}  bias={ba_disc['bias']:+.1f} px  "
            f"MAE={ba_disc['mae']:.1f} px  ({100*ba_disc['bias']/np.mean(gt_disc_arr):+.1f}%)"
        )
        plot_bland_altman(
            gt_disc_arr, dl_disc_arr, [], ba_disc,
            title="Bland-Altman: DL vs GT Disc Height",
            x_label="Mean disc height (px)",
            y_label="DL − GT disc height (px)",
            out_path=args.output_dir / "fig4a_bland_altman_disc_height.png",
        )

    for comp, label in [("fovea_px", "Fovea position (px)"), ("ga_point_px", "GA nearest point (px)")]:
        s = components.get(comp, {})
        if s.get("n"):
            print(f"   {label}: n={s['n']}  MAE={s['mae']:.1f} px  median={s['median']:.1f} px")

    # ------------------------------------------------------------------
    # 5. Summary outputs
    # ------------------------------------------------------------------
    print("\n5. Writing summary outputs…")
    write_summary_csv(
        ba_assisted, calibration, components,
        (disc_c, fovea_c, ga_c),
        args.output_dir / "summary_statistics.csv",
        ba_autonomous=ba_autonomous,
    )
    write_latex_table(
        ba_assisted, calibration, components,
        (disc_c, fovea_c, ga_c),
        args.output_dir / "table1_bland_altman.tex",
        ba_autonomous=ba_autonomous,
    )

    print(f"\nAll outputs saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
