#!/usr/bin/env python3
"""
Headless batch validation for Bland-Altman publication pipeline.

For each image in raw_marked/:
  1. Extract ground-truth landmarks (fovea, disc height, nearest GA point).
  2. Run DL model inference on the corresponding input image.
  3. Compute paired measurements and error metrics.
  4. Save 4-panel comparison PNG per image.
  5. Write bland_altman_data.csv (Bland-Altman-ready paired measurements).
  6. Generate validation_report.html (standalone, embeds all PNGs as base64).

Usage:
    PYTHONPATH=. venv/bin/python3 scripts/batch_bland_altman_validation.py

Options:
    --input-dir     Path to input_images/ (default: project_root/input_images)
    --raw-dir       Path to raw_marked/   (default: project_root/raw_marked)
    --output-dir    Output dir            (default: project_root/test_validation)
    --limit N       Process only first N images (0 = all)
    --offset N      Skip first N images
    --no-model      Skip DL disc model (use geometric fallback); useful for quick GT-only runs
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gui_accuracy_validation import (
    compute_split_x,
    detect_disc_from_red,
    detect_fovea_marker,
    detect_peach_line_endpoints,
)
from src.api.constants import DISC_DIAMETER_MICRONS
from src.api.services.calculator import DistanceCalculatorService
from src.api.services.disc_detector import DiscDetectorService
from src.api.services.fovea_detector import FoveaDetectorService
from src.api.services.ga_segmenter import GASegmenterService

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CSV_FIELDS = [
    "image_id",
    "foveal_involvement",
    "gt_fovea_x",
    "gt_fovea_y",
    "dl_fovea_x",
    "dl_fovea_y",
    "fovea_error_px",
    "gt_disc_height_px",
    "dl_disc_height_px",
    "disc_height_error_px",
    "disc_height_error_pct",
    "gt_ga_x",
    "gt_ga_y",
    "dl_ga_x",
    "dl_ga_y",
    "ga_point_error_px",
    "gt_distance_px",
    "dl_distance_px",
    "gt_distance_um",
    "dl_distance_um",
    "distance_error_um",
    "distance_error_pct",
    # Autonomous (no-click) GA measurement
    "auto_ga_x",
    "auto_ga_y",
    "auto_distance_px",
    "auto_distance_um",
    "auto_distance_error_um",
    "auto_distance_error_pct",
]


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------


def _diff_mask(input_bgr: np.ndarray, raw_bgr: np.ndarray, threshold: int = 35) -> np.ndarray:
    diff = cv2.absdiff(raw_bgr, input_bgr)
    return (np.max(diff, axis=2) > threshold).astype(np.uint8) * 255


def detect_fovea_no_line(
    raw_bgr: np.ndarray,
    diff_mask: np.ndarray,
    split_x: int,
) -> Optional[Tuple[float, float]]:
    """
    Detect fovea dot when no yellow ruler line is present (foveal involvement cases).
    Finds the most prominent compact red or blue blob in the enface diff region.
    """
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)

    def get_candidates(mask: np.ndarray, area_max: int) -> List[Tuple[float, float, float]]:
        masked = cv2.bitwise_and(mask, diff_mask)
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(masked, connectivity=8)
        out: List[Tuple[float, float, float]] = []
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if area < 8 or area > area_max:
                continue
            if max(w, h) / max(1, min(w, h)) > 6.0:
                continue
            cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
            if cx < split_x:
                continue
            out.append((cx, cy, float(area)))
        return out

    red1 = cv2.inRange(hsv, np.array([0, 70, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array([170, 70, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(red1, red2)
    candidates = get_candidates(red, 1200)

    if not candidates:
        blue = cv2.inRange(hsv, np.array([90, 60, 60], dtype=np.uint8), np.array([145, 255, 255], dtype=np.uint8))
        candidates = get_candidates(blue, 1600)

    if not candidates:
        return None
    best = max(candidates, key=lambda t: t[2])
    return best[0], best[1]


def extract_gt_landmarks(
    input_path: Path,
    raw_path: Path,
) -> Dict:
    """
    Extract all ground-truth landmarks from a raw_marked image.

    Returns a dict with keys:
        split_x, foveal_involvement,
        gt_fovea_x, gt_fovea_y (None if undetected),
        gt_disc_height_px, gt_disc_top_y, gt_disc_bottom_y, gt_disc_x (None if undetected),
        gt_ga_x, gt_ga_y (None if no yellow line),
        gt_distance_px, gt_distance_um (None if not applicable).
    """
    input_bgr = cv2.imread(str(input_path))
    if input_bgr is None:
        raise ValueError(f"Cannot read {input_path}")

    raw_bgr = cv2.imread(str(raw_path))
    if raw_bgr is None or raw_bgr.shape[:2] != input_bgr.shape[:2]:
        raise ValueError(f"Cannot read raw_marked image or size mismatch: {raw_path}")

    split_x = compute_split_x(input_bgr)
    dmask = _diff_mask(input_bgr, raw_bgr)

    result: Dict = {
        "split_x": split_x,
        "foveal_involvement": False,
        "gt_fovea_x": None,
        "gt_fovea_y": None,
        "gt_disc_height_px": None,
        "gt_disc_top_y": None,
        "gt_disc_bottom_y": None,
        "gt_disc_x": None,
        "gt_ga_x": None,
        "gt_ga_y": None,
        "gt_distance_px": None,
        "gt_distance_um": None,
    }

    # Disc detection (always attempted, regardless of yellow line)
    disc = detect_disc_from_red(raw_bgr, dmask)
    if disc is not None:
        disc_x, disc_top, disc_bottom = disc
        result["gt_disc_x"] = disc_x
        result["gt_disc_top_y"] = disc_top
        result["gt_disc_bottom_y"] = disc_bottom
        result["gt_disc_height_px"] = disc_bottom - disc_top

    # Try yellow line (GA ruler)
    line_endpoints = detect_peach_line_endpoints(raw_bgr, dmask, split_x)

    if line_endpoints is None:
        # No yellow line → foveal involvement
        result["foveal_involvement"] = True
        # Still try to detect fovea dot
        fovea = detect_fovea_no_line(raw_bgr, dmask, split_x)
        if fovea is not None:
            result["gt_fovea_x"], result["gt_fovea_y"] = fovea
        return result

    # With yellow line: detect fovea from dot closest to line endpoints
    fovea = detect_fovea_marker(raw_bgr, dmask, line_endpoints, split_x)
    if fovea is not None:
        result["gt_fovea_x"], result["gt_fovea_y"] = fovea

    # GA target: endpoint of yellow line farthest from fovea
    p1, p2 = line_endpoints
    if fovea is not None:
        f = np.array(fovea, dtype=np.float32)
        d1 = float(np.linalg.norm(p1 - f))
        d2 = float(np.linalg.norm(p2 - f))
        far = p1 if d1 > d2 else p2
    else:
        # No fovea dot: use the rightmost endpoint as GA target guess
        far = p1 if float(p1[0]) > float(p2[0]) else p2

    result["gt_ga_x"] = float(far[0])
    result["gt_ga_y"] = float(far[1])

    # Compute GT distance
    if (
        result["gt_fovea_x"] is not None
        and result["gt_disc_height_px"] is not None
        and result["gt_disc_height_px"] > 0
    ):
        dx = result["gt_ga_x"] - result["gt_fovea_x"]
        dy = result["gt_ga_y"] - result["gt_fovea_y"]
        result["gt_distance_px"] = float(np.hypot(dx, dy))
        result["gt_distance_um"] = result["gt_distance_px"] * (DISC_DIAMETER_MICRONS / result["gt_disc_height_px"])

    return result


# ---------------------------------------------------------------------------
# DL inference
# ---------------------------------------------------------------------------


def run_dl_inference(
    input_path: Path,
    disc_svc: DiscDetectorService,
    fovea_svc: FoveaDetectorService,
    ga_svc: GASegmenterService,
    dist_svc: DistanceCalculatorService,
    gt_ga_click: Optional[Tuple[float, float]] = None,
) -> Dict:
    """
    Run the full DL pipeline on an input image.

    Disc and fovea detection are fully automatic.

    For non-foveal-involvement cases (gt_ga_click is not None), two GA paths run:
    - Assisted: `segment_ga_local()` anchored to the GT GA click point (simulates
      the clinical workflow where a clinician clicks the GA they want to measure).
    - Autonomous: `segment_ga_regions()` with no click — fully automated segmentation
      using disc/fovea anatomy cues only. Finds the nearest GA boundary point to the
      DL-detected fovea across all returned regions.

    Returns a dict of DL predictions; values may be None on failure.
    """
    result: Dict = {
        "dl_fovea_x": None,
        "dl_fovea_y": None,
        "dl_disc_height_px": None,
        "dl_disc_top_y": None,
        "dl_disc_bottom_y": None,
        "dl_disc_center_x": None,
        "dl_disc_center_y": None,
        "dl_pixel_to_micron": None,
        "dl_split_x": None,
        # Assisted (click-anchored) GA
        "dl_ga_x": None,
        "dl_ga_y": None,
        "dl_distance_px": None,
        "dl_distance_um": None,
        "dl_ga_contours": None,
        "dl_error": None,
        # Autonomous (no-click) GA
        "auto_ga_x": None,
        "auto_ga_y": None,
        "auto_distance_px": None,
        "auto_distance_um": None,
        "auto_ga_contours": None,
        "auto_error": None,
    }

    image = cv2.imread(str(input_path))
    if image is None:
        result["dl_error"] = f"Cannot read {input_path}"
        return result

    image_name = input_path.name

    try:
        disc = disc_svc.detect_from_image(image, image_name)
        result["dl_disc_height_px"] = disc["disc_height_pixels"]
        result["dl_disc_top_y"] = disc["disc_top_y"]
        result["dl_disc_bottom_y"] = disc["disc_bottom_y"]
        result["dl_disc_center_x"] = disc["disc_center_x"]
        result["dl_disc_center_y"] = disc["disc_center_y"]
        result["dl_pixel_to_micron"] = disc["pixel_to_micron_ratio"]
        result["dl_split_x"] = int(disc["en_face_split_x"])
    except Exception as exc:
        result["dl_error"] = f"Disc detection failed: {exc}"
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
        result["dl_fovea_x"] = fovea["fovea_x"]
        result["dl_fovea_y"] = fovea["fovea_y"]
    except Exception as exc:
        result["dl_error"] = f"Fovea detection failed: {exc}"
        return result

    if gt_ga_click is None:
        # Foveal involvement — no GA target to segment; distance is N/A for both paths
        return result

    # ------------------------------------------------------------------
    # Assisted path: segment_ga_local anchored to clinician click point
    # ------------------------------------------------------------------
    try:
        ga_contours = ga_svc.segment_ga_local(
            image,
            click_x=gt_ga_click[0],
            click_y=gt_ga_click[1],
            disc_center_x=disc["disc_center_x"],
            disc_center_y=disc["disc_center_y"],
            disc_height_pixels=disc["disc_height_pixels"],
            en_face_split_x=int(disc["en_face_split_x"]),
            fovea_x=fovea["fovea_x"],
            fovea_y=fovea["fovea_y"],
        )
        result["dl_ga_contours"] = ga_contours
        if ga_contours:
            # Use the single local contour; nearest boundary point to DL fovea
            ga_region = [(int(pt[0][0]), int(pt[0][1])) for pt in ga_contours[0]]
            dist_result = dist_svc.calculate_fovea_to_ga_distance(
                fovea["fovea_x"],
                fovea["fovea_y"],
                ga_region,
                disc["pixel_to_micron_ratio"],
            )
            result["dl_ga_x"] = float(dist_result["nearest_ga_point_x"])
            result["dl_ga_y"] = float(dist_result["nearest_ga_point_y"])
            result["dl_distance_px"] = dist_result["distance_pixels"]
            result["dl_distance_um"] = dist_result["distance_microns"]
        else:
            result["dl_error"] = "Local GA segmentation found no region near click"
    except Exception as exc:
        result["dl_error"] = f"Assisted GA failed: {exc}"

    # ------------------------------------------------------------------
    # Autonomous path: segment_ga_regions — no click, anatomy-aware only
    # ------------------------------------------------------------------
    try:
        auto_contours = ga_svc.segment_ga_regions(
            image,
            disc_center_x=disc["disc_center_x"],
            disc_center_y=disc["disc_center_y"],
            disc_height_pixels=disc["disc_height_pixels"],
            en_face_split_x=int(disc["en_face_split_x"]),
            fovea_x=fovea["fovea_x"],
            fovea_y=fovea["fovea_y"],
        )
        result["auto_ga_contours"] = auto_contours
        if auto_contours:
            # Combine all contour boundary points; find nearest to DL fovea
            all_auto_pts = [
                (int(pt[0][0]), int(pt[0][1]))
                for cnt in auto_contours
                for pt in cnt
            ]
            auto_dist_result = dist_svc.calculate_fovea_to_ga_distance(
                fovea["fovea_x"],
                fovea["fovea_y"],
                all_auto_pts,
                disc["pixel_to_micron_ratio"],
            )
            result["auto_ga_x"] = float(auto_dist_result["nearest_ga_point_x"])
            result["auto_ga_y"] = float(auto_dist_result["nearest_ga_point_y"])
            result["auto_distance_px"] = auto_dist_result["distance_pixels"]
            result["auto_distance_um"] = auto_dist_result["distance_microns"]
        else:
            result["auto_error"] = "Autonomous GA segmentation found no regions"
    except Exception as exc:
        result["auto_error"] = f"Autonomous GA failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# Metrics CSV row
# ---------------------------------------------------------------------------


def build_csv_row(image_id: str, gt: Dict, dl: Dict) -> Dict[str, str]:
    row: Dict[str, str] = {f: "" for f in CSV_FIELDS}
    row["image_id"] = image_id
    row["foveal_involvement"] = "true" if gt["foveal_involvement"] else "false"

    def _fmt(v: Optional[float], decimals: int = 2) -> str:
        return "" if v is None else f"{v:.{decimals}f}"

    row["gt_fovea_x"] = _fmt(gt["gt_fovea_x"])
    row["gt_fovea_y"] = _fmt(gt["gt_fovea_y"])
    row["dl_fovea_x"] = _fmt(dl["dl_fovea_x"])
    row["dl_fovea_y"] = _fmt(dl["dl_fovea_y"])

    if gt["gt_fovea_x"] is not None and dl["dl_fovea_x"] is not None:
        fx = gt["gt_fovea_x"] - dl["dl_fovea_x"]
        fy = gt["gt_fovea_y"] - dl["dl_fovea_y"]
        row["fovea_error_px"] = _fmt(float(np.hypot(fx, fy)))

    row["gt_disc_height_px"] = _fmt(gt["gt_disc_height_px"])
    row["dl_disc_height_px"] = _fmt(dl["dl_disc_height_px"])
    if gt["gt_disc_height_px"] is not None and dl["dl_disc_height_px"] is not None:
        err_px = dl["dl_disc_height_px"] - gt["gt_disc_height_px"]
        row["disc_height_error_px"] = _fmt(err_px)
        row["disc_height_error_pct"] = _fmt(100.0 * err_px / gt["gt_disc_height_px"])

    row["gt_ga_x"] = _fmt(gt["gt_ga_x"])
    row["gt_ga_y"] = _fmt(gt["gt_ga_y"])
    row["dl_ga_x"] = _fmt(dl["dl_ga_x"])
    row["dl_ga_y"] = _fmt(dl["dl_ga_y"])
    if gt["gt_ga_x"] is not None and dl["dl_ga_x"] is not None:
        gx = gt["gt_ga_x"] - dl["dl_ga_x"]
        gy = gt["gt_ga_y"] - dl["dl_ga_y"]
        row["ga_point_error_px"] = _fmt(float(np.hypot(gx, gy)))

    row["gt_distance_px"] = _fmt(gt["gt_distance_px"])
    row["dl_distance_px"] = _fmt(dl["dl_distance_px"])
    row["gt_distance_um"] = _fmt(gt["gt_distance_um"], decimals=1)
    row["dl_distance_um"] = _fmt(dl["dl_distance_um"], decimals=1)
    if gt["gt_distance_um"] is not None and dl["dl_distance_um"] is not None:
        err_um = dl["dl_distance_um"] - gt["gt_distance_um"]
        row["distance_error_um"] = _fmt(err_um, decimals=1)
        row["distance_error_pct"] = _fmt(100.0 * err_um / max(gt["gt_distance_um"], 1.0))

    # Autonomous (no-click) GA fields
    row["auto_ga_x"] = _fmt(dl.get("auto_ga_x"))
    row["auto_ga_y"] = _fmt(dl.get("auto_ga_y"))
    row["auto_distance_px"] = _fmt(dl.get("auto_distance_px"))
    row["auto_distance_um"] = _fmt(dl.get("auto_distance_um"), decimals=1)
    if gt["gt_distance_um"] is not None and dl.get("auto_distance_um") is not None:
        auto_err_um = dl["auto_distance_um"] - gt["gt_distance_um"]
        row["auto_distance_error_um"] = _fmt(auto_err_um, decimals=1)
        row["auto_distance_error_pct"] = _fmt(100.0 * auto_err_um / max(gt["gt_distance_um"], 1.0))

    return row


# ---------------------------------------------------------------------------
# 4-panel comparison PNG
# ---------------------------------------------------------------------------


def _draw_label(img: np.ndarray, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h = 36
    panel = cv2.copyMakeBorder(img, h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return panel


def _resize_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / float(img.shape[0])
    tw = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (tw, target_h), interpolation=cv2.INTER_AREA)


def build_comparison_panel(
    input_path: Path,
    raw_path: Path,
    gt: Dict,
    dl: Dict,
    csv_row: Dict[str, str],
) -> np.ndarray:
    """Build a 4-panel side-by-side comparison image."""
    input_bgr = cv2.imread(str(input_path))
    raw_bgr = cv2.imread(str(raw_path))
    if input_bgr is None or raw_bgr is None:
        raise ValueError("Cannot read images for comparison panel")

    split_x = gt["split_x"]

    # Crop to en-face region
    ef_input = input_bgr[:, split_x:, :].copy()
    ef_raw = raw_bgr[:, split_x:, :].copy()
    ef_dl = input_bgr[:, split_x:, :].copy()

    def orig_to_ef(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x - split_x)), int(round(y)))

    # Panel 3: DL overlays (assisted, cyan) + autonomous (magenta)
    # Assisted GA contours (cyan/yellow)
    if dl.get("dl_ga_contours"):
        for cnt in dl["dl_ga_contours"]:
            shifted = cnt.copy()
            shifted[:, 0, 0] -= split_x
            cv2.drawContours(ef_dl, [shifted], -1, (0, 200, 200), 2)

    # Autonomous GA contours (magenta)
    if dl.get("auto_ga_contours"):
        for cnt in dl["auto_ga_contours"]:
            shifted = cnt.copy()
            shifted[:, 0, 0] -= split_x
            cv2.drawContours(ef_dl, [shifted], -1, (200, 0, 200), 2)

    # DL disc (red vertical line)
    if dl.get("dl_disc_center_x") is not None and dl.get("dl_disc_top_y") is not None:
        dx = int(round(dl["dl_disc_center_x"] - split_x))
        ty = int(round(dl["dl_disc_top_y"]))
        by = int(round(dl["dl_disc_bottom_y"]))
        cv2.line(ef_dl, (dx, ty), (dx, by), (0, 0, 255), 2)

    # DL fovea (green dot)
    if dl.get("dl_fovea_x") is not None:
        fx, fy = orig_to_ef(dl["dl_fovea_x"], dl["dl_fovea_y"])
        cv2.circle(ef_dl, (fx, fy), 6, (0, 255, 0), -1)
        cv2.circle(ef_dl, (fx, fy), 8, (0, 180, 0), 2)

    # Assisted GA nearest point + line to fovea (cyan)
    if dl.get("dl_ga_x") is not None and dl.get("dl_fovea_x") is not None:
        gx, gy = orig_to_ef(dl["dl_ga_x"], dl["dl_ga_y"])
        fx2, fy2 = orig_to_ef(dl["dl_fovea_x"], dl["dl_fovea_y"])
        cv2.line(ef_dl, (fx2, fy2), (gx, gy), (0, 220, 220), 2)
        cv2.circle(ef_dl, (gx, gy), 5, (0, 220, 220), -1)

    # Autonomous GA nearest point + line to fovea (magenta)
    if dl.get("auto_ga_x") is not None and dl.get("dl_fovea_x") is not None:
        agx, agy = orig_to_ef(dl["auto_ga_x"], dl["auto_ga_y"])
        fx3, fy3 = orig_to_ef(dl["dl_fovea_x"], dl["dl_fovea_y"])
        cv2.line(ef_dl, (fx3, fy3), (agx, agy), (220, 0, 220), 2)
        cv2.circle(ef_dl, (agx, agy), 5, (220, 0, 220), -1)

    # GT fovea overlay on panel 3 (orange ring) for quick visual comparison
    if gt.get("gt_fovea_x") is not None:
        gfx, gfy = orig_to_ef(gt["gt_fovea_x"], gt["gt_fovea_y"])
        cv2.circle(ef_dl, (gfx, gfy), 6, (255, 100, 0), 2)

    # GT disc overlay on panel 3 (orange line)
    if gt.get("gt_disc_x") is not None and gt.get("gt_disc_top_y") is not None:
        gdx = int(round(gt["gt_disc_x"] - split_x))
        gty = int(round(gt["gt_disc_top_y"]))
        gby = int(round(gt["gt_disc_bottom_y"]))
        cv2.line(ef_dl, (gdx, gty), (gdx, gby), (0, 140, 255), 2)

    # GT GA on panel 3 (blue circle)
    if gt.get("gt_ga_x") is not None:
        ggx, ggy = orig_to_ef(gt["gt_ga_x"], gt["gt_ga_y"])
        cv2.circle(ef_dl, (ggx, ggy), 5, (255, 0, 0), 2)

    # Panel 4: Metrics text
    h_ef = ef_input.shape[0]
    metrics_panel = np.full((h_ef, max(400, ef_input.shape[1] // 2), 3), 25, dtype=np.uint8)

    def _val(key: str) -> str:
        v = csv_row.get(key, "")
        return v if v else "N/A"

    lines = [
        f"Image: {input_path.stem}",
        "",
        f"Foveal involvement: {gt['foveal_involvement']}",
        "",
        f"Disc height (GT): {_val('gt_disc_height_px')} px",
        f"Disc height (DL): {_val('dl_disc_height_px')} px",
        f"Disc error: {_val('disc_height_error_px')} px ({_val('disc_height_error_pct')}%)",
        "",
        f"Fovea error: {_val('fovea_error_px')} px",
        f"GA point error: {_val('ga_point_error_px')} px",
        "",
        "--- DISTANCES (um) ---",
        f"Manual:     {_val('gt_distance_um')}",
        f"Assisted:   {_val('dl_distance_um')}  err={_val('distance_error_pct')}%",
        f"Autonomous: {_val('auto_distance_um')}  err={_val('auto_distance_error_pct')}%",
    ]
    if dl.get("dl_error"):
        lines += ["", f"Assisted err: {str(dl['dl_error'])[:40]}"]
    if dl.get("auto_error"):
        lines += [f"Auto err: {str(dl['auto_error'])[:40]}"]

    y0 = 30
    for line in lines:
        if not line:
            y0 += 8
            continue
        color = (200, 200, 200)
        if line.startswith("---"):
            color = (240, 200, 100)
        elif "err=" in line or "error" in line.lower():
            try:
                # Extract % value after "err="
                pct_str = line.split("err=")[-1].replace("%", "").strip()
                val = float(pct_str)
                if abs(val) > 15:
                    color = (80, 80, 255)
                elif abs(val) > 5:
                    color = (80, 200, 255)
                else:
                    color = (100, 220, 100)
            except (ValueError, IndexError):
                pass
        cv2.putText(metrics_panel, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y0 += 22

    # Legend
    legend_y = h_ef - 110
    legend_items = [
        ((0, 255, 0), True, "DL fovea"),
        ((255, 100, 0), False, "GT fovea"),
        ((0, 0, 255), None, "DL disc"),
        ((0, 140, 255), None, "GT disc"),
        ((0, 220, 220), None, "Assisted dist"),
        ((220, 0, 220), None, "Autonomous dist"),
    ]
    for i, (clr, filled, label) in enumerate(legend_items):
        ly = legend_y + i * 18
        if filled is None:
            cv2.line(metrics_panel, (10, ly), (20, ly), clr, 2)
        elif filled:
            cv2.circle(metrics_panel, (15, ly), 4, clr, -1)
        else:
            cv2.circle(metrics_panel, (15, ly), 4, clr, 2)
        cv2.putText(metrics_panel, label, (25, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    # Assemble 4 panels at equal height
    target_h = max(ef_input.shape[0], 300)
    p1 = _draw_label(_resize_h(ef_input, target_h), "1. Raw input")
    p2 = _draw_label(_resize_h(ef_raw, target_h), "2. Ground truth (raw_marked)")
    p3 = _draw_label(_resize_h(ef_dl, target_h), "3. DL  [cyan=assisted | magenta=autonomous | green=fovea | red=disc]")
    p4 = _draw_label(_resize_h(metrics_panel, target_h), "4. Metrics")

    # Equalize panel heights for hstack
    h_max = max(p1.shape[0], p2.shape[0], p3.shape[0], p4.shape[0])
    panels = []
    for p in (p1, p2, p3, p4):
        if p.shape[0] < h_max:
            pad = np.full((h_max - p.shape[0], p.shape[1], 3), 20, dtype=np.uint8)
            p = np.vstack([p, pad])
        panels.append(p)

    return np.hstack(panels)


# ---------------------------------------------------------------------------
# Clean distance comparison CSV (clinician-readable 4-column format)
# ---------------------------------------------------------------------------


def write_distance_comparison_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    """
    Write a simple 4-column CSV for clinical review:
        image_id | manual_distance_um | autonomous_distance_um | assisted_distance_um

    Only includes non-foveal-involvement rows with at least one distance value.
    """
    fields = ["image_id", "manual_distance_um", "autonomous_distance_um", "assisted_distance_um"]
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("foveal_involvement") == "true":
                continue
            manual = row.get("gt_distance_um", "")
            autonomous = row.get("auto_distance_um", "")
            assisted = row.get("dl_distance_um", "")
            if not any([manual, autonomous, assisted]):
                continue
            writer.writerow({
                "image_id": row["image_id"],
                "manual_distance_um": manual,
                "autonomous_distance_um": autonomous,
                "assisted_distance_um": assisted,
            })
            written += 1
    print(f"Distance comparison CSV → {out_path}  ({written} rows)")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _row_color_class(distance_error_pct: str) -> str:
    """Return CSS class name based on assisted distance error magnitude."""
    try:
        v = abs(float(distance_error_pct))
        if v < 5:
            return "row-good"
        if v < 15:
            return "row-warn"
        return "row-bad"
    except (ValueError, TypeError):
        return ""


def generate_html_report(
    rows: List[Dict[str, str]],
    comparison_dir: Path,
    output_path: Path,
) -> None:
    """Generate a clean, clinician-readable HTML validation report."""

    def _embed_png(png_path: Path) -> str:
        if not png_path.exists():
            return ""
        with open(png_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def _col(row: Dict, key: str) -> str:
        v = row.get(key, "") or ""
        return v if v else "—"

    def _fmt_dist(val_str: str) -> str:
        """Format distance µm value to 1 decimal, or dash if missing."""
        if not val_str:
            return "—"
        try:
            return f"{float(val_str):.1f}"
        except ValueError:
            return "—"

    def _fmt_err(val_str: str) -> str:
        """Format error % to 1 decimal with sign, or dash if missing."""
        if not val_str:
            return "—"
        try:
            return f"{float(val_str):+.1f}%"
        except ValueError:
            return "—"

    # Separate foveal-involvement and measurable rows
    rows_measurable = [r for r in rows if r.get("foveal_involvement") != "true" and r.get("gt_distance_um")]
    rows_involvement = [r for r in rows if r.get("foveal_involvement") == "true"]

    # Sort measurable rows by absolute assisted error (worst first)
    rows_measurable_sorted = sorted(
        rows_measurable,
        key=lambda r: abs(float(r["distance_error_pct"])) if r.get("distance_error_pct") else -1,
        reverse=True,
    )

    # Summary stats
    n_total = len(rows)
    n_involvement = len(rows_involvement)
    n_measurable = len(rows_measurable)

    assisted_errors = [float(r["distance_error_pct"]) for r in rows_measurable if r.get("distance_error_pct")]
    auto_errors = [float(r["auto_distance_error_pct"]) for r in rows_measurable if r.get("auto_distance_error_pct")]

    assisted_mae = float(np.mean(np.abs(assisted_errors))) if assisted_errors else float("nan")
    assisted_bias = float(np.mean(assisted_errors)) if assisted_errors else float("nan")
    auto_mae = float(np.mean(np.abs(auto_errors))) if auto_errors else float("nan")
    auto_bias = float(np.mean(auto_errors)) if auto_errors else float("nan")
    n_auto = len(auto_errors)

    # Build table rows HTML — measurable cases
    table_rows_html = ""
    for row in rows_measurable_sorted:
        css_class = _row_color_class(row.get("distance_error_pct", ""))
        img_path = comparison_dir / f"{row['image_id']}_comparison.png"
        img_src = _embed_png(img_path)
        img_tag = (
            f'<img src="{img_src}" style="max-width:100%;border:1px solid #ccc;border-radius:4px;">'
            if img_src else "<em>No comparison image</em>"
        )

        manual = _fmt_dist(row.get("gt_distance_um", ""))
        autonomous = _fmt_dist(row.get("auto_distance_um", ""))
        assisted = _fmt_dist(row.get("dl_distance_um", ""))
        assisted_err = _fmt_err(row.get("distance_error_pct", ""))
        auto_err = _fmt_err(row.get("auto_distance_error_pct", ""))

        table_rows_html += f"""
        <tr class="{css_class}" onclick="toggleImg('{row['image_id']}')">
          <td class="id-col">{row['image_id']}</td>
          <td class="num-col">{manual}</td>
          <td class="num-col">{autonomous}</td>
          <td class="num-col">{assisted}</td>
          <td class="err-col">{auto_err}</td>
          <td class="err-col">{assisted_err}</td>
        </tr>
        <tr id="img_{row['image_id']}" class="img-row" style="display:none;">
          <td colspan="6" style="padding:12px 20px;">{img_tag}</td>
        </tr>"""

    # Foveal involvement rows (no distance measurement possible)
    if rows_involvement:
        table_rows_html += """
        <tr class="section-header">
          <td colspan="6">Foveal Involvement Cases (distance measurement not applicable)</td>
        </tr>"""
        for row in rows_involvement:
            table_rows_html += f"""
        <tr class="row-involvement">
          <td class="id-col">{row['image_id']}</td>
          <td colspan="5" style="color:#888;font-style:italic;">Foveal involvement — GA not measurable</td>
        </tr>"""

    auto_summary_row = (
        f"<tr><th>Autonomous vs Manual</th>"
        f"<td>{n_auto} / {n_measurable}</td>"
        f"<td>MAE {auto_mae:.1f}%</td>"
        f"<td>Bias {auto_bias:+.1f}%</td></tr>"
        if not np.isnan(auto_mae) else
        "<tr><th>Autonomous vs Manual</th><td colspan='3'>No data yet</td></tr>"
    )
    assisted_summary_row = (
        f"<tr><th>Assisted vs Manual</th>"
        f"<td>{len(assisted_errors)} / {n_measurable}</td>"
        f"<td>MAE {assisted_mae:.1f}%</td>"
        f"<td>Bias {assisted_bias:+.1f}%</td></tr>"
        if not np.isnan(assisted_mae) else
        "<tr><th>Assisted vs Manual</th><td colspan='3'>No data yet</td></tr>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Atrophy Advisor – Validation Report</title>
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
  .subtitle {{ color: #6c757d; margin-bottom: 24px; font-size: 13px; }}

  /* Summary cards */
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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

  /* Method comparison table in summary */
  .method-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  .method-table th, .method-table td {{
    border: 1px solid #dee2e6; padding: 8px 12px; text-align: left;
  }}
  .method-table th {{ background: #f1f3f5; font-weight: 600; }}
  .method-table tr:nth-child(even) {{ background: #f8f9fa; }}

  /* Download link */
  .download-link {{
    display: inline-block;
    background: #1a6cff;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 20px;
  }}
  .download-link:hover {{ background: #1557cc; }}

  /* Legend */
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

  /* Main data table */
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
  .err-col {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .id-col {{ font-family: monospace; font-size: 12px; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}

  /* Row coloring by assisted error magnitude */
  .row-good {{ background: #f0fff4; }}
  .row-good:hover {{ background: #d4f5e0; }}
  .row-warn {{ background: #fffbf0; }}
  .row-warn:hover {{ background: #fff0cc; }}
  .row-bad {{ background: #fff5f5; }}
  .row-bad:hover {{ background: #ffd5d5; }}
  tr:not([class]):hover {{ background: #f8f9fa; }}

  /* Foveal involvement rows */
  .row-involvement {{ background: #fafafa; }}
  .row-involvement td {{ color: #888; }}
  .section-header td {{
    background: #495057; color: white; font-weight: 600;
    font-size: 12px; padding: 7px 12px;
  }}

  /* Image expand rows */
  .img-row td {{ padding: 0; }}
  .img-row {{ background: white !important; }}

  /* Error color coding in cells */
  .err-good {{ color: #28a745; }}
  .err-warn {{ color: #856404; }}
  .err-bad {{ color: #dc3545; }}
</style>
</head>
<body>

<h1>Atrophy Advisor — Validation Report</h1>
<p class="subtitle">
  Fovea-to-GA distance: Manual (physician ruler) vs Autonomous (fully automated) vs Assisted (clinician click + DL segmentation)
</p>

<div class="summary-grid">
  <div class="card">
    <div class="card-label">Total Images</div>
    <div class="card-value">{n_total}</div>
    <div class="card-sub">{n_involvement} foveal involvement &nbsp;·&nbsp; {n_measurable} measurable</div>
  </div>
  <div class="card">
    <div class="card-label">Assisted — MAE vs Manual</div>
    <div class="card-value">{assisted_mae:.1f}%</div>
    <div class="card-sub">Bias {assisted_bias:+.1f}% &nbsp;·&nbsp; n={len(assisted_errors)}</div>
  </div>
  <div class="card">
    <div class="card-label">Autonomous — MAE vs Manual</div>
    <div class="card-value">{"N/A" if np.isnan(auto_mae) else f"{auto_mae:.1f}%"}</div>
    <div class="card-sub">{"Run batch to generate" if np.isnan(auto_mae) else f"Bias {auto_bias:+.1f}% · n={n_auto}"}</div>
  </div>
</div>

<h2>Method Comparison Summary</h2>
<table class="method-table">
  <thead><tr><th>Method</th><th>Cases with data</th><th>MAE vs Manual</th><th>Bias vs Manual</th></tr></thead>
  <tbody>
    {assisted_summary_row}
    {auto_summary_row}
  </tbody>
</table>

<a class="download-link" href="distance_comparison.csv" download>
  ⬇ Download Distance Comparison CSV
</a>

<h2>Per-Image Results</h2>
<div class="legend">
  <span>Row color (Assisted error):</span>
  <span class="legend-item"><span class="swatch" style="background:#d4f5e0;border:1px solid #aae0be;"></span> &lt;5% (good)</span>
  <span class="legend-item"><span class="swatch" style="background:#fff0cc;border:1px solid #ffd66e;"></span> 5–15% (moderate)</span>
  <span class="legend-item"><span class="swatch" style="background:#ffd5d5;border:1px solid #f5a0a0;"></span> &gt;15% (large error)</span>
  <span style="color:#6c757d;">· Click row to expand comparison image</span>
</div>

<table id="validationTable">
<thead>
  <tr>
    <th onclick="sortTable(0)">Case ID ▾</th>
    <th onclick="sortTable(1)" class="num-col">Manual (µm)</th>
    <th onclick="sortTable(2)" class="num-col">Autonomous (µm)</th>
    <th onclick="sortTable(3)" class="num-col">Assisted (µm)</th>
    <th onclick="sortTable(4)" class="err-col">Auto Error</th>
    <th onclick="sortTable(5)" class="err-col">Assisted Error</th>
  </tr>
</thead>
<tbody>{table_rows_html}</tbody>
</table>

<script>
function toggleImg(id) {{
  var row = document.getElementById('img_' + id);
  if (row) row.style.display = row.style.display === 'none' ? '' : 'none';
}}
function sortTable(n) {{
  var table = document.getElementById('validationTable');
  var rows = Array.from(table.tBodies[0].rows).filter(function(r) {{
    return !r.id.startsWith('img_') && !r.classList.contains('section-header') && !r.classList.contains('row-involvement');
  }});
  var asc = table.dataset.sort == n && table.dataset.dir == 'asc';
  rows.sort(function(a, b) {{
    var va = a.cells[n] ? a.cells[n].textContent.trim().replace(/[+%]/g, '') : '';
    var vb = b.cells[n] ? b.cells[n].textContent.trim().replace(/[+%]/g, '') : '';
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

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless batch validation for Bland-Altman pipeline.")
    p.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input_images")
    p.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "raw_marked")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_validation")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--no-model", action="store_true", help="Skip DL disc model (use geometric fallback)")
    p.add_argument(
        "--cv2-seed", type=int, default=int(os.environ.get("OCT_CV2_SEED", "0")),
        help="Seed for OpenCV's global RNG (default 0, or $OCT_CV2_SEED). "
             "cv2.kmeans draws from it, so segmentation varies seed to seed; "
             "sweep this to put error bars on a validation result.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Same rationale as tests/conftest.py: cv2.kmeans is seeded from one
    # process-global RNG, so a run's clustering depends on everything drawn
    # before it. Pinning it makes a single run reproducible; varying it across
    # runs is what exposes the segmentation's own seed sensitivity.
    cv2.setRNGSeed(args.cv2_seed)

    out_dir = args.output_dir
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize services once (expensive: model load)
    print("Loading DL services…")
    model_path = str(PROJECT_ROOT / "weights" / "best_disc_model.pth") if not args.no_model else ""
    disc_svc = DiscDetectorService(model_path=model_path)
    fovea_svc = FoveaDetectorService()
    ga_svc = GASegmenterService(use_sam=False)
    dist_svc = DistanceCalculatorService()
    print(f"Disc model loaded: {'yes' if disc_svc.model is not None else 'fallback mode'}")

    # Enumerate raw_marked images
    raw_images = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    # Skip images that have no corresponding input image
    valid_pairs = [
        (args.input_dir / p.name, p)
        for p in raw_images
        if (args.input_dir / p.name).exists()
    ]

    # Apply offset/limit
    valid_pairs = valid_pairs[args.offset:]
    if args.limit > 0:
        valid_pairs = valid_pairs[: args.limit]

    print(f"Processing {len(valid_pairs)} image(s)…")

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
            gt_ga_click: Optional[Tuple[float, float]] = None
            if gt.get("gt_ga_x") is not None and gt.get("gt_ga_y") is not None:
                gt_ga_click = (gt["gt_ga_x"], gt["gt_ga_y"])
            dl = run_dl_inference(input_path, disc_svc, fovea_svc, ga_svc, dist_svc, gt_ga_click=gt_ga_click)
        except Exception as exc:
            msg = f"DL inference failed for {image_id}: {exc}"
            print(f"ERROR: {msg}")
            traceback.print_exc()
            errors.append(msg)
            dl = {k: None for k in [
                "dl_fovea_x", "dl_fovea_y", "dl_disc_height_px",
                "dl_disc_top_y", "dl_disc_bottom_y", "dl_disc_center_x", "dl_disc_center_y",
                "dl_pixel_to_micron", "dl_split_x", "dl_ga_x", "dl_ga_y",
                "dl_distance_px", "dl_distance_um", "dl_ga_contours", "dl_error",
                "auto_ga_x", "auto_ga_y", "auto_distance_px", "auto_distance_um",
                "auto_ga_contours", "auto_error",
            ]}
            dl["dl_error"] = str(exc)

        csv_row = build_csv_row(image_id, gt, dl)
        all_rows.append(csv_row)

        # Build and save comparison panel
        try:
            panel = build_comparison_panel(input_path, raw_path, gt, dl, csv_row)
            out_png = comp_dir / f"{image_id}_comparison.png"
            cv2.imwrite(str(out_png), panel)
        except Exception as exc:
            print(f"\n  WARNING: Panel generation failed: {exc}")

        status_parts = []
        if csv_row["distance_error_pct"]:
            status_parts.append(f"dist_err={csv_row['distance_error_pct']}%")
        if gt["foveal_involvement"]:
            status_parts.append("FOVEAL_INVOLVEMENT")
        if dl.get("dl_error"):
            status_parts.append(f"DL_ERR={dl['dl_error'][:30]}")
        print(", ".join(status_parts) if status_parts else "ok")

    # Write CSV — merge with any existing data so partial --offset/--limit runs are additive
    csv_path = out_dir / "bland_altman_data.csv"
    existing_rows: Dict[str, Dict[str, str]] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_rows[r["image_id"]] = {k: r.get(k, "") for k in CSV_FIELDS}
    for row in all_rows:
        existing_rows[row["image_id"]] = row
    merged_rows = sorted(existing_rows.values(), key=lambda r: r["image_id"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"\nCSV → {csv_path}  ({len(merged_rows)} rows total, {len(all_rows)} new/updated)")

    # Write clean 4-column distance comparison CSV for clinical review
    dist_csv_path = out_dir / "distance_comparison.csv"
    write_distance_comparison_csv(merged_rows, dist_csv_path)

    # Write HTML report
    html_path = out_dir / "validation_report.html"
    generate_html_report(merged_rows, comp_dir, html_path)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
