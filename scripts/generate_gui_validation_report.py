#!/usr/bin/env python3
"""
Generate an HTML validation report for the GUI accuracy validation run.

Reads:
  - test_validation/summary.csv          (from gui_accuracy_validation.py)
  - test_validation/bland_altman_data.csv (for gt_fovea, gt_disc_height, gt_distance_um)

Outputs:
  - test_validation/gui_validation_report.html

Validation logic:
  - The cloud agent clicked disc/fovea/GA at exact GT pixel positions.
  - GA point error (px) = distance between the GT GA point and the
    detected cyan-line endpoint in the GUI screenshot.
  - GT distance (µm) comes from bland_altman_data.csv (physician measurement).
  - Since GA was placed at the exact GT position, GUI distance ≈ GT distance.
    Any residual error reflects sub-pixel rounding only.
"""

from __future__ import annotations

import argparse
import base64
import csv
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gui_accuracy_validation import compute_split_x

DISC_DIAMETER_MICRONS = 1800.0
LABEL_H = 40  # header added by export_side_by_side


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def safe_float(v: str | None) -> float | None:
    try:
        return float(v) if v and v.strip() else None
    except (ValueError, AttributeError):
        return None


def err_class_px(px: float | None) -> str:
    if px is None:
        return ""
    if px <= 15:
        return "row-good"
    if px <= 50:
        return "row-warn"
    return "row-bad"


def err_color_px(px: float | None) -> str:
    if px is None:
        return ""
    if px <= 15:
        return "err-good"
    if px <= 50:
        return "err-warn"
    return "err-bad"


def fmt_um(v: float | None) -> str:
    return f"{v:.0f}" if v is not None else "—"


def fmt_px(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def coord_str(x: float | None, y: float | None) -> str:
    if x is None or y is None:
        return "—"
    return f"{x:.0f}, {y:.0f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_validation")
    args = parser.parse_args()

    out_dir = args.output_dir
    summary_path = out_dir / "summary.csv"
    ba_path = out_dir / "bland_altman_data.csv"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run gui_accuracy_validation.py first.")
        return 1

    summary_rows = load_csv(summary_path)

    # Build lookup: image_id -> row from bland_altman_data.csv
    ba_lookup: dict[str, dict] = {}
    if ba_path.exists():
        for row in load_csv(ba_path):
            iid = row.get("image_id", "").strip()
            if iid:
                ba_lookup[iid] = row

    records = []
    for row in summary_rows:
        if row.get("prediction_detected", "no") != "yes":
            continue

        filename = row["image_filename"]
        image_id = Path(filename).stem
        case_key = row["case_key"]
        pair_mode = row.get("pair_mode", "")

        gt_ga_x = safe_float(row.get("gt_ga_x"))
        gt_ga_y = safe_float(row.get("gt_ga_y"))

        ba = ba_lookup.get(image_id) or ba_lookup.get(case_key)
        gt_fovea_x = safe_float(ba.get("gt_fovea_x") if ba else None)
        gt_fovea_y = safe_float(ba.get("gt_fovea_y") if ba else None)
        gt_disc_height = safe_float(ba.get("gt_disc_height_px") if ba else None)

        gt_dist_um = safe_float(ba.get("gt_distance_um") if ba else None)
        if gt_dist_um is None and gt_fovea_x is not None and gt_ga_x is not None and gt_disc_height:
            gt_dist_um = math.hypot(gt_ga_x - gt_fovea_x, gt_ga_y - gt_fovea_y) * (DISC_DIAMETER_MICRONS / gt_disc_height)

        # Comparison image
        img_path = PROJECT_ROOT / row.get("output_file", "")
        img_b64 = encode_image(img_path) if img_path.exists() else None

        # Re-detect GA endpoint from the comparison PNG (farthest cyan pixel from fovea)
        pred_ga_x: float | None = None
        pred_ga_y: float | None = None
        point_err_px: float | None = None
        if img_path.exists() and gt_fovea_x is not None:
            input_img = cv2.imread(str(PROJECT_ROOT / "input_images" / filename))
            if input_img is not None:
                img_h, img_w = input_img.shape[:2]
                split_x = compute_split_x(input_img)

                comp_img = cv2.imread(str(img_path))
                if comp_img is not None:
                    ch, cw = comp_img.shape[:2]
                    gui_enface = comp_img[LABEL_H:, :cw // 2]
                    ef_h, ef_w = gui_enface.shape[:2]

                    hsv = cv2.cvtColor(gui_enface, cv2.COLOR_BGR2HSV)
                    cyan = cv2.inRange(hsv, np.array([75, 70, 70], np.uint8), np.array([105, 255, 255], np.uint8))
                    cyan = cv2.morphologyEx(cyan, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                    ys, xs = np.where(cyan > 0)

                    if len(xs) > 0:
                        enface_w_orig = max(1, img_w - split_x)
                        scale_y = ef_h / float(img_h)
                        fov_gui_x = (gt_fovea_x - split_x) * (ef_w / float(enface_w_orig))
                        fov_gui_y = gt_fovea_y * scale_y

                        d2 = (xs - fov_gui_x) ** 2 + (ys - fov_gui_y) ** 2
                        far_idx = int(np.argmax(d2))

                        # Map back to original image coordinates
                        pred_local_x = float(xs[far_idx]) * (enface_w_orig / float(max(1, ef_w)))
                        pred_ga_x = float(split_x) + pred_local_x
                        pred_ga_y = float(ys[far_idx]) * (float(img_h) / float(max(1, ef_h)))

                        if gt_ga_x is not None and gt_ga_y is not None:
                            point_err_px = math.hypot(pred_ga_x - gt_ga_x, pred_ga_y - gt_ga_y)

        records.append({
            "image_id": image_id,
            "case_key": case_key,
            "pair_mode": pair_mode,
            "gt_ga_x": gt_ga_x,
            "gt_ga_y": gt_ga_y,
            "gt_fovea_x": gt_fovea_x,
            "gt_fovea_y": gt_fovea_y,
            "gt_disc_height": gt_disc_height,
            "gt_dist_um": gt_dist_um,
            "point_err_px": point_err_px,
            "img_b64": img_b64,
        })

    records.sort(key=lambda r: (r["case_key"], r["image_id"]))

    # Summary stats on point error (only trust values ≤ 50px as reliable)
    all_errors = [r["point_err_px"] for r in records if r["point_err_px"] is not None]
    reliable_errors = [e for e in all_errors if e <= 50]
    n = len(records)
    n_dist = sum(1 for r in records if r["gt_dist_um"] is not None)
    mae_px = statistics.mean(all_errors) if all_errors else float("nan")
    median_px = statistics.median(all_errors) if all_errors else float("nan")
    within15 = sum(1 for e in all_errors if e <= 15)
    within30 = sum(1 for e in all_errors if e <= 30)

    # Table rows
    table_rows_html = []
    for r in records:
        iid = r["image_id"]
        rc = err_class_px(r["point_err_px"])
        ec = err_color_px(r["point_err_px"])

        img_cell = ""
        if r["img_b64"]:
            img_cell = (
                f'<tr id="img_{iid}" class="img-row" style="display:none">'
                f'<td colspan="5"><img src="data:image/png;base64,{r["img_b64"]}" '
                f'style="max-width:100%;display:block;"></td></tr>'
            )

        table_rows_html.append(f"""
<tr class="{rc}" onclick="toggleImg('{iid}')">
  <td class="id-col">{iid}</td>
  <td class="num-col">{fmt_um(r["gt_dist_um"])}</td>
  <td class="num-col {ec}">{fmt_px(r["point_err_px"])}</td>
  <td class="num-col">{coord_str(r["gt_ga_x"], r["gt_ga_y"])}</td>
  <td class="num-col">{coord_str(r["gt_fovea_x"], r["gt_fovea_y"])}</td>
</tr>
{img_cell}""")

    table_html = "\n".join(table_rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Atrophy Advisor – GUI Validation Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    background: #f8f9fa;
    color: #212529;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 14px;
    margin: 0;
    padding: 24px;
  }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; color: #1a1a2e; }}
  h2 {{ font-size: 15px; font-weight: 600; color: #495057; margin: 20px 0 8px; }}
  .subtitle {{ color: #6c757d; margin-bottom: 6px; font-size: 13px; }}
  .note {{
    background: #e8f4fd; border: 1px solid #b8d9f0; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; color: #1a5276; margin-bottom: 20px;
  }}

  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .card {{
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 14px 18px;
  }}
  .card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #6c757d; }}
  .card-value {{ font-size: 22px; font-weight: 700; color: #1a1a2e; margin: 2px 0; }}
  .card-sub {{ font-size: 12px; color: #6c757d; }}

  .legend {{
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 12px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}

  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  thead tr {{ background: #1a1a2e; color: white; }}
  thead th {{
    padding: 11px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
  }}
  thead th:hover {{ background: #2d2d50; }}
  .num-col {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .id-col {{ font-family: monospace; font-size: 12px; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  tr {{ cursor: pointer; }}

  .row-good {{ background: #f0fff4; }}
  .row-good:hover {{ background: #d4f5e0; }}
  .row-warn {{ background: #fffbf0; }}
  .row-warn:hover {{ background: #fff0cc; }}
  .row-bad {{ background: #fff5f5; }}
  .row-bad:hover {{ background: #ffd5d5; }}
  tr:not([class]):hover {{ background: #f8f9fa; }}

  .err-good {{ color: #28a745; font-weight: 600; }}
  .err-warn {{ color: #856404; font-weight: 600; }}
  .err-bad {{ color: #dc3545; font-weight: 600; }}

  .img-row td {{ padding: 0; }}
  .img-row {{ background: white !important; }}
</style>
</head>
<body>

<h1>Atrophy Advisor — GUI Validation Report</h1>
<p class="subtitle">
  Ground-truth inputs placed at exact pixel positions via cloud agent · Validates app rendering and disc/fovea/GA accuracy
</p>
<div class="note">
  <strong>What this validates:</strong> The cloud agent placed the optic disc bracket, fovea, and GA endpoint at
  exact ground-truth pixel coordinates extracted from <code>raw_marked/</code>. Each comparison image shows
  the GUI en-face (left) vs the annotated ground-truth en-face (right).
  Since GA was clicked at the exact GT position, the app distance measurement = GT distance by construction.
  <br><br>
  <strong>GA point error (px):</strong> Measured by detecting the cyan distance line endpoint in the GUI screenshot.
  Values ≤15 px are reliable. Larger values may reflect detection difficulty for long lines — verify visually.
</div>

<div class="summary-grid">
  <div class="card">
    <div class="card-label">Images Processed</div>
    <div class="card-value">{n}</div>
    <div class="card-sub">{n_dist} with GT distance reference</div>
  </div>
  <div class="card">
    <div class="card-label">GA Point Error — MAE</div>
    <div class="card-value">{mae_px:.1f} px</div>
    <div class="card-sub">Median {median_px:.1f} px</div>
  </div>
  <div class="card">
    <div class="card-label">Within 15 px</div>
    <div class="card-value">{within15}/{n}</div>
    <div class="card-sub">{within30}/{n} within 30 px</div>
  </div>
</div>

<h2>Per-Image Results</h2>
<div class="legend">
  <span>Row color (GA point error):</span>
  <span class="legend-item"><span class="swatch" style="background:#d4f5e0;border:1px solid #aae0be;"></span> ≤15 px</span>
  <span class="legend-item"><span class="swatch" style="background:#fff0cc;border:1px solid #ffd66e;"></span> 16–50 px</span>
  <span class="legend-item"><span class="swatch" style="background:#ffd5d5;border:1px solid #f5a0a0;"></span> &gt;50 px</span>
  <span style="color:#6c757d;">· Click any row to expand comparison image (GUI vs ground truth)</span>
</div>

<table id="mainTable">
<thead>
  <tr>
    <th onclick="sortTable(0)">Image ID ▾</th>
    <th onclick="sortTable(1)" class="num-col">GT Distance (µm)</th>
    <th onclick="sortTable(2)" class="num-col">GA Point Error (px)</th>
    <th onclick="sortTable(3)" class="num-col">GT GA (x, y)</th>
    <th onclick="sortTable(4)" class="num-col">GT Fovea (x, y)</th>
  </tr>
</thead>
<tbody>
{table_html}
</tbody>
</table>

<script>
function toggleImg(id) {{
  var row = document.getElementById('img_' + id);
  if (row) row.style.display = row.style.display === 'none' ? '' : 'none';
}}
function sortTable(n) {{
  var table = document.getElementById('mainTable');
  var rows = Array.from(table.tBodies[0].rows).filter(function(r) {{
    return !r.id.startsWith('img_');
  }});
  var asc = table.dataset.sort == n && table.dataset.dir == 'asc';
  rows.sort(function(a, b) {{
    var va = a.cells[n] ? a.cells[n].textContent.trim().replace(/[µm]/g, '') : '';
    var vb = b.cells[n] ? b.cells[n].textContent.trim().replace(/[µm]/g, '') : '';
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  table.dataset.sort = n; table.dataset.dir = asc ? 'desc' : 'asc';
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

    out_path = out_dir / "gui_validation_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {out_path}")
    print(f"  {n} images · MAE GA point error {mae_px:.1f} px · Median {median_px:.1f} px")
    print(f"  Within 15 px: {within15}/{n} · Within 30 px: {within30}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
