#!/usr/bin/env python3
"""
V1 vs V2 Disc/Fovea Comparison Tool.

Runs both the V1 disc model (best_disc_model.pth, heatmap-only) and the V2
disc model (best_disc_model_v2.pth, with height-regression head) on every
image in raw_marked/ that has a matching input_images/ counterpart, then
produces:

  - Per-image 3-panel comparison PNGs (V1 overlays | V2 overlays | metrics)
  - Standalone HTML report with embedded images and summary statistics
  - v1_v2_metrics.csv for further offline analysis

Usage:
    PYTHONPATH=. venv/bin/python3 scripts/v1_v2_comparison.py

Options:
    --input-dir   Path to input_images/    (default: <project_root>/input_images)
    --raw-dir     Path to raw_marked/      (default: <project_root>/raw_marked)
    --output-dir  Output directory         (default: <project_root>/test_validation/v1_v2_comparison)
    --limit N     Process only first N images (0 = all)
    --offset N    Skip first N images
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_bland_altman_validation import extract_gt_landmarks
from src.api.services.disc_detector import DiscDetectorService
from src.api.services.fovea_detector import FoveaDetectorService

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CSV_FIELDS = [
    "image_id",
    # Disc height
    "gt_disc_height_px",
    "v1_disc_height_px",
    "v2_disc_height_px",
    "v1_disc_height_err_px",
    "v1_disc_height_err_pct",
    "v2_disc_height_err_px",
    "v2_disc_height_err_pct",
    "v1_v2_disc_height_delta_px",
    # Disc center
    "v1_disc_center_x",
    "v1_disc_center_y",
    "v2_disc_center_x",
    "v2_disc_center_y",
    "v1_v2_disc_center_shift_px",
    # Pixel-to-micron ratio
    "v1_px_to_um",
    "v2_px_to_um",
    "v1_v2_px_to_um_pct_change",
    # Fovea position
    "gt_fovea_x",
    "gt_fovea_y",
    "v1_fovea_x",
    "v1_fovea_y",
    "v2_fovea_x",
    "v2_fovea_y",
    "v1_fovea_err_px",
    "v2_fovea_err_px",
    "v1_v2_fovea_shift_px",
    # Errors
    "v1_error",
    "v2_error",
]


# ---------------------------------------------------------------------------
# Single-model inference (disc + fovea only, no GA)
# ---------------------------------------------------------------------------


def run_inference(
    input_path: Path,
    disc_svc: DiscDetectorService,
    fovea_svc: FoveaDetectorService,
) -> Dict:
    """Run disc + fovea detection on one image. Returns dict of results."""
    result: Dict = {
        "disc_height_px": None,
        "disc_top_y": None,
        "disc_bottom_y": None,
        "disc_center_x": None,
        "disc_center_y": None,
        "px_to_um": None,
        "split_x": None,
        "fovea_x": None,
        "fovea_y": None,
        "error": None,
    }

    image = cv2.imread(str(input_path))
    if image is None:
        result["error"] = f"Cannot read {input_path}"
        return result

    try:
        disc = disc_svc.detect_from_image(image, image_name=input_path.name)
        result["disc_height_px"] = disc["disc_height_pixels"]
        result["disc_top_y"] = disc["disc_top_y"]
        result["disc_bottom_y"] = disc["disc_bottom_y"]
        result["disc_center_x"] = disc["disc_center_x"]
        result["disc_center_y"] = disc["disc_center_y"]
        result["px_to_um"] = disc["pixel_to_micron_ratio"]
        result["split_x"] = int(disc["en_face_split_x"])
    except Exception as exc:
        result["error"] = f"Disc detection failed: {exc}"
        return result

    try:
        fovea = fovea_svc.detect_fovea(
            image,
            disc["disc_center_x"],
            disc["disc_center_y"],
            disc["disc_height_pixels"],
            int(disc["en_face_split_x"]),
            use_manual_adjustment=False,
        )
        result["fovea_x"] = fovea["fovea_x"]
        result["fovea_y"] = fovea["fovea_y"]
    except Exception as exc:
        result["error"] = f"Fovea detection failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------


def _fmt(v: Optional[float], decimals: int = 2) -> str:
    return "" if v is None else f"{v:.{decimals}f}"


def build_metrics_row(
    image_id: str,
    gt: Dict,
    v1: Dict,
    v2: Dict,
) -> Dict[str, str]:
    row: Dict[str, str] = {f: "" for f in CSV_FIELDS}
    row["image_id"] = image_id
    row["v1_error"] = v1.get("error") or ""
    row["v2_error"] = v2.get("error") or ""

    # GT disc
    row["gt_disc_height_px"] = _fmt(gt.get("gt_disc_height_px"))
    row["gt_fovea_x"] = _fmt(gt.get("gt_fovea_x"))
    row["gt_fovea_y"] = _fmt(gt.get("gt_fovea_y"))

    # Disc height
    row["v1_disc_height_px"] = _fmt(v1.get("disc_height_px"))
    row["v2_disc_height_px"] = _fmt(v2.get("disc_height_px"))

    gt_h = gt.get("gt_disc_height_px")
    v1_h = v1.get("disc_height_px")
    v2_h = v2.get("disc_height_px")

    if gt_h and v1_h:
        e = v1_h - gt_h
        row["v1_disc_height_err_px"] = _fmt(e)
        row["v1_disc_height_err_pct"] = _fmt(100.0 * e / gt_h)
    if gt_h and v2_h:
        e = v2_h - gt_h
        row["v2_disc_height_err_px"] = _fmt(e)
        row["v2_disc_height_err_pct"] = _fmt(100.0 * e / gt_h)
    if v1_h and v2_h:
        row["v1_v2_disc_height_delta_px"] = _fmt(v2_h - v1_h)

    # Disc center
    row["v1_disc_center_x"] = _fmt(v1.get("disc_center_x"))
    row["v1_disc_center_y"] = _fmt(v1.get("disc_center_y"))
    row["v2_disc_center_x"] = _fmt(v2.get("disc_center_x"))
    row["v2_disc_center_y"] = _fmt(v2.get("disc_center_y"))
    if v1.get("disc_center_x") is not None and v2.get("disc_center_x") is not None:
        shift = float(np.hypot(
            v2["disc_center_x"] - v1["disc_center_x"],
            v2["disc_center_y"] - v1["disc_center_y"],
        ))
        row["v1_v2_disc_center_shift_px"] = _fmt(shift)

    # Pixel-to-micron ratio
    row["v1_px_to_um"] = _fmt(v1.get("px_to_um"), 4)
    row["v2_px_to_um"] = _fmt(v2.get("px_to_um"), 4)
    if v1.get("px_to_um") and v2.get("px_to_um"):
        pct = 100.0 * (v2["px_to_um"] - v1["px_to_um"]) / v1["px_to_um"]
        row["v1_v2_px_to_um_pct_change"] = _fmt(pct)

    # Fovea
    row["v1_fovea_x"] = _fmt(v1.get("fovea_x"))
    row["v1_fovea_y"] = _fmt(v1.get("fovea_y"))
    row["v2_fovea_x"] = _fmt(v2.get("fovea_x"))
    row["v2_fovea_y"] = _fmt(v2.get("fovea_y"))

    gt_fx, gt_fy = gt.get("gt_fovea_x"), gt.get("gt_fovea_y")
    v1_fx, v1_fy = v1.get("fovea_x"), v1.get("fovea_y")
    v2_fx, v2_fy = v2.get("fovea_x"), v2.get("fovea_y")

    if gt_fx is not None and v1_fx is not None:
        row["v1_fovea_err_px"] = _fmt(float(np.hypot(v1_fx - gt_fx, v1_fy - gt_fy)))
    if gt_fx is not None and v2_fx is not None:
        row["v2_fovea_err_px"] = _fmt(float(np.hypot(v2_fx - gt_fx, v2_fy - gt_fy)))
    if v1_fx is not None and v2_fx is not None:
        row["v1_v2_fovea_shift_px"] = _fmt(float(np.hypot(v2_fx - v1_fx, v2_fy - v1_fy)))

    return row


# ---------------------------------------------------------------------------
# Comparison panel PNG
# ---------------------------------------------------------------------------


def _draw_label(img: np.ndarray, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h = 36
    panel = cv2.copyMakeBorder(img, h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)
    return panel


def _resize_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / float(img.shape[0])
    tw = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (tw, target_h), interpolation=cv2.INTER_AREA)


def _draw_overlays(
    en_face: np.ndarray,
    split_x: int,
    result: Dict,
    gt: Optional[Dict],
    disc_line_color: Tuple[int, int, int],
    fovea_color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw disc line and fovea dot on an en-face crop. GT shown in muted orange/cyan."""
    out = en_face.copy()

    def ef(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x - split_x)), int(round(y)))

    # Model disc line
    if result.get("disc_center_x") is not None and result.get("disc_top_y") is not None:
        dx = int(round(result["disc_center_x"] - split_x))
        ty = int(round(result["disc_top_y"]))
        by = int(round(result["disc_bottom_y"]))
        cv2.line(out, (dx, ty), (dx, by), disc_line_color, 2)

    # Model fovea dot
    if result.get("fovea_x") is not None:
        fx, fy = ef(result["fovea_x"], result["fovea_y"])
        cv2.circle(out, (fx, fy), 6, fovea_color, -1)
        cv2.circle(out, (fx, fy), 8, tuple(max(0, c - 60) for c in fovea_color), 2)

    # GT disc line (orange)
    if gt and gt.get("gt_disc_x") is not None and gt.get("gt_disc_top_y") is not None:
        gdx = int(round(gt["gt_disc_x"] - split_x))
        gty = int(round(gt["gt_disc_top_y"]))
        gby = int(round(gt["gt_disc_bottom_y"]))
        cv2.line(out, (gdx, gty), (gdx, gby), (0, 140, 255), 2)

    # GT fovea (blue circle)
    if gt and gt.get("gt_fovea_x") is not None:
        gfx, gfy = ef(gt["gt_fovea_x"], gt["gt_fovea_y"])
        cv2.circle(out, (gfx, gfy), 6, (255, 80, 0), 2)

    return out


def build_comparison_panel(
    input_path: Path,
    gt: Dict,
    v1: Dict,
    v2: Dict,
    row: Dict[str, str],
) -> np.ndarray:
    """Build a 3-panel comparison image: V1 | V2 | metrics."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Cannot read {input_path}")

    split_x = gt["split_x"]
    ef_base = image[:, split_x:, :].copy()

    # Panel 1: V1 overlays (red disc, green fovea)
    ef_v1 = _draw_overlays(ef_base, split_x, v1, gt, (0, 0, 255), (0, 255, 0))
    # Panel 2: V2 overlays (magenta disc, cyan fovea)
    ef_v2 = _draw_overlays(ef_base, split_x, v2, gt, (255, 0, 200), (0, 255, 255))

    # Panel 3: metrics text
    h_ef = ef_base.shape[0]
    metrics_w = max(360, ef_base.shape[1] // 2)
    metrics_panel = np.full((h_ef, metrics_w, 3), 25, dtype=np.uint8)

    def _pct_color(pct_str: str) -> Tuple[int, int, int]:
        try:
            v = abs(float(pct_str))
            if v < 5:
                return (80, 220, 80)
            if v < 15:
                return (80, 220, 220)
            return (80, 80, 255)
        except (ValueError, TypeError):
            return (200, 200, 200)

    gt_h = row.get("gt_disc_height_px", "N/A") or "N/A"
    v1_h = row.get("v1_disc_height_px", "N/A") or "N/A"
    v2_h = row.get("v2_disc_height_px", "N/A") or "N/A"
    v1_herr = row.get("v1_disc_height_err_pct", "") or ""
    v2_herr = row.get("v2_disc_height_err_pct", "") or ""
    delta_h = row.get("v1_v2_disc_height_delta_px", "") or ""
    v1_px_um = row.get("v1_px_to_um", "N/A") or "N/A"
    v2_px_um = row.get("v2_px_to_um", "N/A") or "N/A"
    px_um_chg = row.get("v1_v2_px_to_um_pct_change", "") or ""
    center_shift = row.get("v1_v2_disc_center_shift_px", "") or ""
    fovea_shift = row.get("v1_v2_fovea_shift_px", "") or ""
    v1_ferr = row.get("v1_fovea_err_px", "") or ""
    v2_ferr = row.get("v2_fovea_err_px", "") or ""

    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"Image: {input_path.stem}", (240, 240, 240)),
        ("", (200, 200, 200)),
        ("-- DISC HEIGHT --", (180, 180, 100)),
        (f"  GT:  {gt_h} px", (200, 200, 200)),
        (f"  V1:  {v1_h} px  (err {v1_herr}%)", _pct_color(v1_herr)),
        (f"  V2:  {v2_h} px  (err {v2_herr}%)", _pct_color(v2_herr)),
        (f"  V2-V1 delta: {delta_h} px", (200, 200, 200)),
        ("", (200, 200, 200)),
        ("-- DISC CENTER SHIFT --", (180, 180, 100)),
        (f"  V1→V2: {center_shift} px", (200, 200, 200)),
        ("", (200, 200, 200)),
        ("-- PX-TO-MICRON RATIO --", (180, 180, 100)),
        (f"  V1:  {v1_px_um}", (200, 200, 200)),
        (f"  V2:  {v2_px_um}  ({px_um_chg}%)", _pct_color(px_um_chg)),
        ("", (200, 200, 200)),
        ("-- FOVEA --", (180, 180, 100)),
        (f"  V1 err vs GT: {v1_ferr} px", _pct_color(v1_ferr) if v1_ferr else (200, 200, 200)),
        (f"  V2 err vs GT: {v2_ferr} px", _pct_color(v2_ferr) if v2_ferr else (200, 200, 200)),
        (f"  V1→V2 shift:  {fovea_shift} px", (200, 200, 200)),
    ]

    v1_err_msg = v1.get("error") or ""
    v2_err_msg = v2.get("error") or ""
    if v1_err_msg:
        lines += [("", (200, 200, 200)), (f"V1 ERR: {v1_err_msg[:38]}", (80, 80, 255))]
    if v2_err_msg:
        lines += [("", (200, 200, 200)), (f"V2 ERR: {v2_err_msg[:38]}", (80, 80, 255))]

    y0 = 28
    for text, color in lines:
        if not text:
            y0 += 8
            continue
        cv2.putText(metrics_panel, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y0 += 20

    # Legend at bottom
    legend_y = h_ef - 110
    items = [
        ((0, 0, 255), "filled", "V1 disc (red)"),
        ((255, 0, 200), "filled", "V2 disc (magenta)"),
        ((0, 255, 0), "filled", "V1 fovea (green)"),
        ((0, 255, 255), "filled", "V2 fovea (cyan)"),
        ((0, 140, 255), "line", "GT disc (orange)"),
        ((255, 80, 0), "circle", "GT fovea (blue-orange)"),
    ]
    for color, kind, label in items:
        if legend_y >= h_ef - 8:
            break
        if kind == "filled":
            cv2.rectangle(metrics_panel, (8, legend_y - 6), (18, legend_y + 6), color, -1)
        elif kind == "line":
            cv2.line(metrics_panel, (8, legend_y), (18, legend_y), color, 2)
        else:
            cv2.circle(metrics_panel, (13, legend_y), 5, color, 2)
        cv2.putText(metrics_panel, label, (24, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
        legend_y += 18

    # Assemble panels at equal height
    target_h = max(ef_base.shape[0], 280)
    p1 = _draw_label(_resize_h(ef_v1, target_h), "V1  [red disc | green fovea | orange GT disc | blue-orange GT fovea]")
    p2 = _draw_label(_resize_h(ef_v2, target_h), "V2  [magenta disc | cyan fovea | orange GT disc | blue-orange GT fovea]")
    p3 = _draw_label(_resize_h(metrics_panel, target_h), "Metrics")

    h_max = max(p1.shape[0], p2.shape[0], p3.shape[0])
    panels = []
    for p in (p1, p2, p3):
        if p.shape[0] < h_max:
            pad = np.full((h_max - p.shape[0], p.shape[1], 3), 20, dtype=np.uint8)
            p = np.vstack([p, pad])
        panels.append(p)

    return np.hstack(panels)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _embed_png(png_path: Path) -> str:
    if not png_path.exists():
        return ""
    with open(png_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _row_color(row: Dict[str, str]) -> str:
    """Color row based on whether V2 improved, worsened, or is similar to V1 on disc height error."""
    v1_pct_str = row.get("v1_disc_height_err_pct", "")
    v2_pct_str = row.get("v2_disc_height_err_pct", "")
    try:
        v1_abs = abs(float(v1_pct_str))
        v2_abs = abs(float(v2_pct_str))
        diff = v1_abs - v2_abs  # positive = V2 improved
        if diff > 5:
            return "background:#0d2a0d;"   # V2 improved
        if diff < -5:
            return "background:#2a0d0d;"   # V2 regressed
        return "background:#1e1e1e;"
    except (ValueError, TypeError):
        return "background:#1e1e1e;"


def generate_html_report(
    rows: List[Dict[str, str]],
    comp_dir: Path,
    output_path: Path,
    v1_loaded: bool,
    v2_loaded: bool,
) -> None:
    """Generate a self-contained HTML V1 vs V2 comparison report."""

    header_cols = [
        "image_id",
        "gt_disc_height_px",
        "v1_disc_height_px",
        "v2_disc_height_px",
        "v1_disc_height_err_pct",
        "v2_disc_height_err_pct",
        "v1_v2_disc_height_delta_px",
        "v1_v2_disc_center_shift_px",
        "v1_px_to_um",
        "v2_px_to_um",
        "v1_v2_px_to_um_pct_change",
        "v1_fovea_err_px",
        "v2_fovea_err_px",
        "v1_v2_fovea_shift_px",
    ]

    def _col(row: Dict, key: str) -> str:
        return row.get(key, "") or "—"

    # Sort by absolute V2 disc height error (worst first)
    def _sort_key(r: Dict[str, str]) -> float:
        try:
            return abs(float(r.get("v2_disc_height_err_pct", "") or "0"))
        except ValueError:
            return 0.0

    rows_sorted = sorted(rows, key=_sort_key, reverse=True)

    # Summary statistics
    def _collect(field: str) -> List[float]:
        out = []
        for r in rows:
            v = r.get(field, "")
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                pass
        return out

    v1_h_errs = _collect("v1_disc_height_err_pct")
    v2_h_errs = _collect("v2_disc_height_err_pct")
    v1_f_errs = _collect("v1_fovea_err_px")
    v2_f_errs = _collect("v2_fovea_err_px")
    fovea_shifts = _collect("v1_v2_fovea_shift_px")
    center_shifts = _collect("v1_v2_disc_center_shift_px")

    def _stat(vals: List[float]) -> str:
        if not vals:
            return "N/A"
        return f"MAE={float(np.mean(np.abs(vals))):.1f}  bias={float(np.mean(vals)):+.1f}  median={float(np.median(vals)):+.1f}"

    n_improved = sum(
        1 for r in rows
        if _row_color(r) == "background:#0d2a0d;"
    )
    n_regressed = sum(
        1 for r in rows
        if _row_color(r) == "background:#2a0d0d;"
    )

    header_ths = "".join(
        f'<th onclick="sortTable({i})" style="cursor:pointer;padding:8px;background:#333;">{c}</th>'
        for i, c in enumerate(header_cols)
    )

    table_rows_html = ""
    for row in rows_sorted:
        color = _row_color(row)
        img_path = comp_dir / f"{row['image_id']}_comparison.png"
        img_src = _embed_png(img_path)
        img_tag = (
            f'<img src="{img_src}" style="max-width:1200px;width:100%;border:1px solid #444;">'
            if img_src else "<em>no image</em>"
        )
        cells = "".join(f"<td>{_col(row, c)}</td>" for c in header_cols)
        table_rows_html += f"""
        <tr style="{color}" onclick="toggleImg('{row['image_id']}')">
          {cells}
        </tr>
        <tr id="img_{row['image_id']}" style="display:none;">
          <td colspan="{len(header_cols)}">{img_tag}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>V1 vs V2 Disc/Fovea Comparison</title>
<style>
  body {{ background:#111; color:#ccc; font-family:monospace; font-size:13px; margin:20px; }}
  h1 {{ color:#fff; }}
  .summary {{ background:#1e1e1e; padding:14px; border-radius:6px; margin-bottom:20px; line-height:1.7; }}
  .summary b {{ color:#fff; }}
  table {{ border-collapse:collapse; width:100%; }}
  td, th {{ border:1px solid #333; padding:5px 8px; white-space:nowrap; }}
  tr:hover {{ filter:brightness(1.3); cursor:pointer; }}
  .legend {{ background:#1e1e1e; padding:8px; margin-bottom:12px; border-radius:4px; font-size:12px; }}
  .good {{ color:#6f6; }} .warn {{ color:#ff6; }} .bad {{ color:#f66; }}
  .v1tag {{ color:#f88; font-weight:bold; }} .v2tag {{ color:#8ff; font-weight:bold; }}
</style>
</head>
<body>
<h1>V1 vs V2 Disc / Fovea Comparison Report</h1>

<div class="summary">
  <b>Models</b><br>
  &nbsp;&nbsp;V1: best_disc_model.pth &nbsp;(heatmap-only)&nbsp;&nbsp;
  loaded: {'<span class="good">yes</span>' if v1_loaded else '<span class="bad">NO — fallback mode</span>'}<br>
  &nbsp;&nbsp;V2: best_disc_model_v2.pth (heatmap + height head)&nbsp;&nbsp;
  loaded: {'<span class="good">yes</span>' if v2_loaded else '<span class="bad">NO — fallback mode</span>'}<br>
  <br>
  <b>Dataset</b>: {len(rows)} images processed<br>
  &nbsp;&nbsp;V2 improved disc height error (&gt;5pp better): <span class="good">{n_improved}</span><br>
  &nbsp;&nbsp;V2 regressed disc height error (&gt;5pp worse): <span class="bad">{n_regressed}</span><br>
  <br>
  <b>Disc height error %</b><br>
  &nbsp;&nbsp;<span class="v1tag">V1</span> {_stat(v1_h_errs)}<br>
  &nbsp;&nbsp;<span class="v2tag">V2</span> {_stat(v2_h_errs)}<br>
  <br>
  <b>Fovea error vs GT (px)</b><br>
  &nbsp;&nbsp;<span class="v1tag">V1</span> {_stat(v1_f_errs)}<br>
  &nbsp;&nbsp;<span class="v2tag">V2</span> {_stat(v2_f_errs)}<br>
  <br>
  <b>V1→V2 shifts</b><br>
  &nbsp;&nbsp;Disc center: {_stat(center_shifts)}<br>
  &nbsp;&nbsp;Fovea:       {_stat(fovea_shifts)}<br>
</div>

<div class="legend">
  Row color:
  <span class="good">■ V2 improved &gt;5pp</span>&nbsp;&nbsp;
  <span style="color:#aaa">■ Similar (±5pp)</span>&nbsp;&nbsp;
  <span class="bad">■ V2 regressed &gt;5pp</span>&nbsp;&nbsp;|&nbsp;
  Click a row to toggle its comparison image.
</div>

<table id="mainTable">
<thead><tr>{header_ths}</tr></thead>
<tbody>{table_rows_html}</tbody>
</table>

<script>
function toggleImg(id) {{
  var row = document.getElementById('img_' + id);
  row.style.display = (row.style.display === 'none' || row.style.display === '') ? 'table-row' : 'none';
}}
function sortTable(n) {{
  var table = document.getElementById('mainTable');
  var rows = Array.from(table.tBodies[0].rows).filter(function(r) {{
    return !r.id || !r.id.startsWith('img_');
  }});
  var asc = table.dataset.sort == n && table.dataset.dir == 'asc';
  rows.sort(function(a, b) {{
    var va = a.cells[n] ? a.cells[n].textContent.trim() : '';
    var vb = b.cells[n] ? b.cells[n].textContent.trim() : '';
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  table.dataset.sort = n;
  table.dataset.dir = asc ? 'desc' : 'asc';
  var tbody = table.tBodies[0];
  rows.forEach(function(r) {{
    tbody.appendChild(r);
    var imgRow = document.getElementById('img_' + r.cells[0].textContent.trim());
    if (imgRow) tbody.appendChild(imgRow);
  }});
}}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1 vs V2 disc/fovea comparison.")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input_images")
    p.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "raw_marked")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "test_validation" / "v1_v2_comparison",
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir: Path = args.output_dir
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = PROJECT_ROOT / "weights"
    v1_path = str(weights_dir / "best_disc_model.pth")

    print("Loading V1 disc model…")
    disc_v1 = DiscDetectorService(model_path=v1_path, force_version="v1")
    v1_loaded = disc_v1.model is not None
    print(f"  V1 loaded: {v1_loaded} {'(has_height_head=' + str(disc_v1.has_height_head) + ')' if v1_loaded else '(fallback)'}")

    print("Loading V2 disc model…")
    disc_v2 = DiscDetectorService(model_path=v1_path, force_version="v2")
    v2_loaded = disc_v2.model is not None
    print(f"  V2 loaded: {v2_loaded} {'(has_height_head=' + str(disc_v2.has_height_head) + ')' if v2_loaded else '(fallback)'}")

    fovea_svc = FoveaDetectorService()

    # Enumerate valid pairs (raw_marked image must have a matching input_images counterpart)
    raw_images = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    valid_pairs = [
        (args.input_dir / p.name, p)
        for p in raw_images
        if (args.input_dir / p.name).exists()
    ]

    valid_pairs = valid_pairs[args.offset:]
    if args.limit > 0:
        valid_pairs = valid_pairs[: args.limit]

    print(f"Processing {len(valid_pairs)} image(s)…\n")

    all_rows: List[Dict[str, str]] = []
    errors: List[str] = []

    for idx, (input_path, raw_path) in enumerate(valid_pairs, start=1):
        image_id = input_path.stem
        print(f"[{idx}/{len(valid_pairs)}] {image_id}", end=" … ", flush=True)

        try:
            gt = extract_gt_landmarks(input_path, raw_path)
        except Exception as exc:
            msg = f"GT extraction failed for {image_id}: {exc}"
            print(f"ERROR: {msg}")
            errors.append(msg)
            continue

        try:
            v1_result = run_inference(input_path, disc_v1, fovea_svc)
        except Exception as exc:
            traceback.print_exc()
            v1_result = {k: None for k in [
                "disc_height_px", "disc_top_y", "disc_bottom_y",
                "disc_center_x", "disc_center_y", "px_to_um",
                "split_x", "fovea_x", "fovea_y",
            ]}
            v1_result["error"] = str(exc)

        try:
            v2_result = run_inference(input_path, disc_v2, fovea_svc)
        except Exception as exc:
            traceback.print_exc()
            v2_result = {k: None for k in [
                "disc_height_px", "disc_top_y", "disc_bottom_y",
                "disc_center_x", "disc_center_y", "px_to_um",
                "split_x", "fovea_x", "fovea_y",
            ]}
            v2_result["error"] = str(exc)

        row = build_metrics_row(image_id, gt, v1_result, v2_result)
        all_rows.append(row)

        # Build and save comparison panel
        try:
            panel = build_comparison_panel(input_path, gt, v1_result, v2_result, row)
            out_png = comp_dir / f"{image_id}_comparison.png"
            cv2.imwrite(str(out_png), panel)
        except Exception as exc:
            print(f"\n  WARNING: Panel generation failed: {exc}")

        # Status line
        parts = []
        if row.get("v1_disc_height_err_pct"):
            parts.append(f"V1_disc_err={row['v1_disc_height_err_pct']}%")
        if row.get("v2_disc_height_err_pct"):
            parts.append(f"V2_disc_err={row['v2_disc_height_err_pct']}%")
        if v1_result.get("error"):
            parts.append(f"V1_ERR={v1_result['error'][:25]}")
        if v2_result.get("error"):
            parts.append(f"V2_ERR={v2_result['error'][:25]}")
        print(", ".join(parts) if parts else "ok")

    # Write CSV
    csv_path = out_dir / "v1_v2_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV → {csv_path}")

    # Write HTML report
    html_path = out_dir / "v1_v2_comparison_report.html"
    generate_html_report(all_rows, comp_dir, html_path, v1_loaded, v2_loaded)

    if errors:
        print(f"\n{len(errors)} error(s) during processing:")
        for e in errors:
            print(f"  {e}")

    print(f"\nDone. {len(all_rows)} images processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
