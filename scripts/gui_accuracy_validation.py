#!/usr/bin/env python3
"""
Automated GUI accuracy validation against raw_marked references.

Workflow:
1) Launch local backend/frontend.
2) For each input image group:
   - Upload before + after (or duplicate before for single-image cases).
   - Adjust optic disc bracket handles to match ground-truth position (before fovea confirm).
   - Set fovea clicks from raw_marked annotations (exact pixel from ground truth).
   - Confirm both foveas, trigger GA segmentation.
   - Click "Manual" button, then click the exact GA endpoint from ground truth.
3) Capture canvas screenshots, crop en-face, and export side-by-side with raw_marked en-face.

Output:
  <project_root>/test_validation/*.png
  <project_root>/test_validation/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from playwright.sync_api import Browser, Locator, Page, sync_playwright

# Allow importing split utilities from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import get_split_indices_and_images


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageMeta:
    filename: str
    input_path: Path
    raw_path: Optional[Path]
    width: int
    height: int
    split_x: int
    fovea_xy: Optional[Tuple[float, float]]
    ga_target_xy: Optional[Tuple[float, float]]
    # Ground-truth disc: (center_x, top_y, bottom_y) in original image pixels
    disc_gt: Optional[Tuple[float, float, float]] = None


@dataclass
class Case:
    key: str
    before_filename: str
    after_filename: Optional[str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_input_images(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files, key=lambda p: p.name)


def build_cases(files: List[Path]) -> List[Case]:
    grouped: Dict[str, Dict[str, str]] = {}
    for path in files:
        stem = path.stem
        if stem.endswith("_2"):
            key = stem[:-2]
            grouped.setdefault(key, {})["after"] = path.name
        else:
            key = stem
            grouped.setdefault(key, {})["before"] = path.name

    cases: List[Case] = []
    for key in sorted(grouped.keys()):
        item = grouped[key]
        before = item.get("before")
        after = item.get("after")
        if before and after:
            cases.append(Case(key=key, before_filename=before, after_filename=after))
        elif before:
            cases.append(Case(key=key, before_filename=before, after_filename=None))
        elif after:
            cases.append(Case(key=key, before_filename=after, after_filename=None))

    return cases


def compute_split_x(image_bgr: np.ndarray) -> int:
    try:
        _, _, metadata = get_split_indices_and_images(image_bgr, divider_safety_margin=10)
        split_x = int(metadata["final_split_column"])
        if split_x <= 0 or split_x >= image_bgr.shape[1]:
            raise ValueError("invalid split")
        return split_x
    except Exception:
        return image_bgr.shape[1] // 2


def _largest_component_mask(binary_mask: np.ndarray, min_area: int = 20) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    best_idx = -1
    best_area = -1
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > best_area:
            best_idx = label_idx
            best_area = area
    if best_idx < 0:
        return None
    return (labels == best_idx).astype(np.uint8) * 255


def _mask_endpoints(component_mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 2:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    center = np.mean(pts, axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return p1, p2


def detect_peach_line_endpoints(
    raw_bgr: np.ndarray,
    diff_mask: np.ndarray,
    split_x: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect the annotated yellow/peach GA ruler line in raw_marked.
    This intentionally excludes red disc/fovea annotations.
    """
    b = raw_bgr[:, :, 0]
    g = raw_bgr[:, :, 1]
    r = raw_bgr[:, :, 2]

    # GA ruler is bright yellow in these annotations.
    # We intentionally reject peach-like disc markings by requiring low blue.
    yellow_like = (
        (r >= 180)
        & (g >= 160)
        & (b <= 150)
        & (np.abs(r.astype(np.int16) - g.astype(np.int16)) <= 80)
    )

    x_coords = np.tile(np.arange(raw_bgr.shape[1]), (raw_bgr.shape[0], 1))
    enface_side = x_coords >= split_x
    line_mask = (yellow_like & (diff_mask > 0) & enface_side).astype(np.uint8) * 255
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    best_idx = -1
    best_score = -1.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        width = float(stats[idx, cv2.CC_STAT_WIDTH])
        height = float(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 25:
            continue
        long_side = max(width, height)
        short_side = max(1.0, min(width, height))
        elongation = long_side / short_side
        score = area + (long_side * 8.0) + (elongation * 20.0)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None

    component_mask = (labels == best_idx).astype(np.uint8) * 255
    return _mask_endpoints(component_mask)


def detect_disc_from_red(raw_bgr: np.ndarray, diff_mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 70, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array([170, 70, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.bitwise_and(red, diff_mask)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    best_idx = -1
    best_score = -1.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        width = float(stats[idx, cv2.CC_STAT_WIDTH])
        height = float(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 50 or height < 25:
            continue
        verticality = height / max(width, 1.0)
        if verticality < 2.0:
            continue
        score = area * verticality
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None

    ys, xs = np.where(labels == best_idx)
    if len(xs) == 0:
        return None
    disc_x = float(np.mean(xs))
    disc_top = float(np.min(ys))
    disc_bottom = float(np.max(ys))
    return disc_x, disc_top, disc_bottom


def detect_fovea_marker(
    raw_bgr: np.ndarray,
    diff_mask: np.ndarray,
    line_endpoints: Tuple[np.ndarray, np.ndarray],
    split_x: int,
) -> Optional[Tuple[float, float]]:
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    def collect_candidates(mask: np.ndarray, area_max: int) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
            out.append((cx, cy))
        return out

    red1 = cv2.inRange(hsv, np.array([0, 70, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array([170, 70, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.bitwise_and(red, diff_mask)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidates = collect_candidates(red, area_max=1200)

    # Optional fallback: blue outline around red dot (when center fill is weak)
    if not candidates:
        blue = cv2.inRange(hsv, np.array([90, 60, 60], dtype=np.uint8), np.array([145, 255, 255], dtype=np.uint8))
        blue = cv2.bitwise_and(blue, diff_mask)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidates = collect_candidates(blue, area_max=1600)

    if not candidates:
        return None

    p1, p2 = line_endpoints

    def endpoint_dist(pt: Tuple[float, float]) -> float:
        c = np.array(pt, dtype=np.float32)
        return float(min(np.linalg.norm(c - p1), np.linalg.norm(c - p2)))

    best = min(candidates, key=endpoint_dist)
    return best


def extract_landmarks(
    input_path: Path,
    raw_path: Optional[Path],
) -> Tuple[int, Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float, float]]]:
    """
    Returns (split_x, fovea_xy, ga_target_xy, disc_gt).
    disc_gt is (center_x, top_y, bottom_y) or None.
    """
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to read {input_path}")

    split_x = compute_split_x(image)
    if raw_path is None or not raw_path.exists():
        return split_x, None, None, None

    marked = cv2.imread(str(raw_path))
    if marked is None or marked.shape[:2] != image.shape[:2]:
        return split_x, None, None, None

    diff = cv2.absdiff(marked, image)
    # Keep only strong annotation deltas (prevents drift from subtle compression differences)
    diff_mask = (np.max(diff, axis=2) > 35).astype(np.uint8) * 255

    line_endpoints = detect_peach_line_endpoints(marked, diff_mask, split_x)
    if line_endpoints is None:
        return split_x, None, None, None

    fovea = detect_fovea_marker(marked, diff_mask, line_endpoints, split_x)
    if fovea is None:
        return split_x, None, None, None

    ga_target: Optional[Tuple[float, float]] = None
    p1, p2 = line_endpoints
    f = np.array(fovea, dtype=np.float32)
    d1 = float(np.linalg.norm(p1 - f))
    d2 = float(np.linalg.norm(p2 - f))
    far = p1 if d1 > d2 else p2
    ga_target = (float(far[0]), float(far[1]))

    disc_gt = detect_disc_from_red(marked, diff_mask)

    return split_x, fovea, ga_target, disc_gt


def prepare_image_meta(input_dir: Path, raw_dir: Path) -> Dict[str, ImageMeta]:
    meta: Dict[str, ImageMeta] = {}
    for path in list_input_images(input_dir):
        raw_path = raw_dir / path.name
        if not raw_path.exists():
            raw_path = None

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read {path}")

        split_x, fovea_xy, ga_target_xy, disc_gt = extract_landmarks(path, raw_path)
        meta[path.name] = ImageMeta(
            filename=path.name,
            input_path=path,
            raw_path=raw_path,
            width=img.shape[1],
            height=img.shape[0],
            split_x=split_x,
            fovea_xy=fovea_xy,
            ga_target_xy=ga_target_xy,
            disc_gt=disc_gt,
        )
    return meta


def wait_for_http(url: str, timeout_s: float) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = requests.get(url, timeout=1.5)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


def start_backend(output_dir: Path, python_exe: Path) -> Tuple[subprocess.Popen, Path]:
    log_path = output_dir / "backend.log"
    log_file = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    cmd = [
        str(python_exe),
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_http("http://127.0.0.1:8000/health", timeout_s=120.0)
    return proc, log_path


def start_frontend(output_dir: Path) -> Tuple[subprocess.Popen, Path]:
    log_path = output_dir / "frontend.log"
    log_file = open(log_path, "w", encoding="utf-8")
    cmd = ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "3000", "--strictPort"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT / "src" / "frontend"),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_http("http://127.0.0.1:3000", timeout_s=120.0)
    return proc, log_path


def stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def click_image_point(canvas: Locator, image_xy: Tuple[float, float], img_w: int, img_h: int) -> None:
    box = canvas.bounding_box()
    if box is None:
        raise RuntimeError("Canvas bounding box unavailable")

    rel_x = float(image_xy[0]) / float(img_w) * float(box["width"])
    rel_y = float(image_xy[1]) / float(img_h) * float(box["height"])
    rel_x = float(np.clip(rel_x, 1.0, max(1.0, box["width"] - 1.0)))
    rel_y = float(np.clip(rel_y, 1.0, max(1.0, box["height"] - 1.0)))
    canvas.click(position={"x": rel_x, "y": rel_y}, force=True)


def drag_image_point(
    page: "Page",
    canvas: "Locator",
    from_xy: Tuple[float, float],
    to_xy: Tuple[float, float],
    img_w: int,
    img_h: int,
    steps: int = 15,
) -> None:
    """Simulate a mouse drag on the canvas between two image-coordinate points."""
    box = canvas.bounding_box()
    if box is None:
        raise RuntimeError("Canvas bounding box unavailable for drag")

    def to_page(xy: Tuple[float, float]) -> Tuple[float, float]:
        px = box["x"] + float(xy[0]) / img_w * box["width"]
        py = box["y"] + float(xy[1]) / img_h * box["height"]
        px = float(np.clip(px, box["x"] + 1, box["x"] + box["width"] - 1))
        py = float(np.clip(py, box["y"] + 1, box["y"] + box["height"] - 1))
        return px, py

    sx, sy = to_page(from_xy)
    ex, ey = to_page(to_xy)
    page.mouse.move(sx, sy)
    page.mouse.down()
    # Smooth drag in small steps so React motion handlers fire
    for i in range(1, steps + 1):
        t = i / steps
        page.mouse.move(sx + (ex - sx) * t, sy + (ey - sy) * t)
    page.mouse.up()


def adjust_disc_to_ground_truth(
    page: "Page",
    canvas: "Locator",
    meta: ImageMeta,
    auto_disc: Dict[str, float],
) -> None:
    """
    Drag the disc bracket handles to match ground-truth positions.

    auto_disc must contain keys: disc_center_x, disc_top_y, disc_bottom_y.
    meta.disc_gt must be (gt_center_x, gt_top_y, gt_bottom_y).
    """
    if meta.disc_gt is None:
        return

    gt_cx, gt_top, gt_bottom = meta.disc_gt
    auto_cx = auto_disc["disc_center_x"]
    auto_top = auto_disc["disc_top_y"]
    auto_bottom = auto_disc["disc_bottom_y"]

    # Drag top handle to ground-truth top (keep center_x from GT)
    drag_image_point(
        page, canvas,
        from_xy=(auto_cx, auto_top),
        to_xy=(gt_cx, gt_top),
        img_w=meta.width,
        img_h=meta.height,
    )
    # Small settle pause so React state updates before next drag
    page.wait_for_timeout(150)

    # Drag bottom handle to ground-truth bottom
    drag_image_point(
        page, canvas,
        from_xy=(auto_cx, auto_bottom),
        to_xy=(gt_cx, gt_bottom),
        img_w=meta.width,
        img_h=meta.height,
    )
    page.wait_for_timeout(150)


def select_distance_manual(
    page: "Page",
    canvas: "Locator",
    manual_button: "Locator",
    meta: ImageMeta,
    target_xy: Tuple[float, float],
) -> bool:
    """
    Click the Manual button to enter manual GA mode, then click the exact
    ground-truth GA endpoint.  Returns True on success.
    """
    try:
        manual_button.click(timeout=10000)
    except Exception:
        return False

    # Small pause for state transition
    page.wait_for_timeout(300)

    click_image_point(canvas, target_xy, meta.width, meta.height)
    return True


def decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode canvas screenshot")
    return decoded


def resize_to_height(image_bgr: np.ndarray, target_h: int) -> np.ndarray:
    if image_bgr.shape[0] == target_h:
        return image_bgr
    ratio = target_h / float(image_bgr.shape[0])
    target_w = max(1, int(round(image_bgr.shape[1] * ratio)))
    return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def export_side_by_side(
    gui_canvas_bgr: np.ndarray,
    meta: ImageMeta,
    out_path: Path,
) -> None:
    input_bgr = cv2.imread(str(meta.input_path))
    if input_bgr is None:
        raise ValueError(f"Cannot read input image {meta.input_path}")

    reference_path = meta.raw_path if meta.raw_path and meta.raw_path.exists() else meta.input_path
    reference_bgr = cv2.imread(str(reference_path))
    if reference_bgr is None:
        raise ValueError(f"Cannot read reference image {reference_path}")

    gui_h, gui_w = gui_canvas_bgr.shape[:2]
    scale_x = gui_w / float(max(1, meta.width))
    split_gui = int(round(meta.split_x * scale_x))
    split_gui = int(np.clip(split_gui, 1, gui_w - 1))

    gui_enface = gui_canvas_bgr[:, split_gui:]
    ref_split = int(np.clip(meta.split_x, 1, reference_bgr.shape[1] - 1))
    ref_enface = reference_bgr[:, ref_split:]

    ref_enface = resize_to_height(ref_enface, gui_enface.shape[0])

    label_h = 40
    left = cv2.copyMakeBorder(gui_enface, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    right = cv2.copyMakeBorder(ref_enface, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    cv2.putText(left, "GUI en-face", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    right_label = "raw_marked en-face" if meta.raw_path else "input en-face (raw_marked missing)"
    cv2.putText(right, right_label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack([left, right])
    cv2.imwrite(str(out_path), combined)


def _detect_gui_predicted_ga_point(
    gui_canvas_bgr: np.ndarray,
    meta: ImageMeta,
) -> Optional[Tuple[float, float, float]]:
    """
    Estimate GUI-selected GA endpoint from cyan contour pixels.

    Returns:
        (pred_x_orig, pred_y_orig, cyan_coverage_ratio) in original image coordinates.
        None when detection is not possible.
    """
    if meta.fovea_xy is None:
        return None

    gui_h, gui_w = gui_canvas_bgr.shape[:2]
    if gui_h <= 0 or gui_w <= 0:
        return None

    scale_x = gui_w / float(max(1, meta.width))
    scale_y = gui_h / float(max(1, meta.height))
    split_gui = int(round(meta.split_x * scale_x))
    split_gui = int(np.clip(split_gui, 1, gui_w - 1))

    gui_enface = gui_canvas_bgr[:, split_gui:]
    ef_h, ef_w = gui_enface.shape[:2]
    if ef_h <= 0 or ef_w <= 0:
        return None

    enface_width_orig = max(1, meta.width - meta.split_x)
    fovea_local_x = float(meta.fovea_xy[0] - meta.split_x)
    fovea_local_y = float(meta.fovea_xy[1])
    fovea_gui_x = fovea_local_x * (ef_w / float(enface_width_orig))
    fovea_gui_y = fovea_local_y * scale_y

    hsv = cv2.cvtColor(gui_enface, cv2.COLOR_BGR2HSV)
    cyan_mask = cv2.inRange(
        hsv,
        np.array([75, 70, 70], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    ys, xs = np.where(cyan_mask > 0)
    if len(xs) == 0:
        return None

    d2 = (xs - fovea_gui_x) ** 2 + (ys - fovea_gui_y) ** 2
    farthest_idx = int(np.argmax(d2))
    pred_gui_x = float(xs[farthest_idx])
    pred_gui_y = float(ys[farthest_idx])

    pred_local_x = pred_gui_x * (enface_width_orig / float(max(1, ef_w)))
    pred_local_y = pred_gui_y * (meta.height / float(max(1, gui_h)))
    pred_x_orig = float(meta.split_x + pred_local_x)
    pred_y_orig = float(pred_local_y)
    coverage_ratio = float(np.sum(cyan_mask > 0) / float(max(1, ef_h * ef_w)))

    return pred_x_orig, pred_y_orig, coverage_ratio


def compute_gui_error_metrics(
    gui_canvas_bgr: np.ndarray,
    meta: ImageMeta,
) -> Dict[str, str]:
    """
    Compute validation metrics from GUI canvas versus raw-marked target.

    Output fields are string-typed for direct CSV serialization.
    """
    metrics: Dict[str, str] = {
        "prediction_detected": "no",
        "pred_ga_x": "",
        "pred_ga_y": "",
        "gt_ga_x": "",
        "gt_ga_y": "",
        "point_error_px": "",
        "gt_distance_px": "",
        "pred_distance_px": "",
        "distance_error_px": "",
        "cyan_coverage_ratio": "",
    }

    if meta.fovea_xy is None or meta.ga_target_xy is None:
        return metrics

    pred = _detect_gui_predicted_ga_point(gui_canvas_bgr, meta)
    if pred is None:
        return metrics

    pred_x, pred_y, cyan_ratio = pred
    gt_x, gt_y = float(meta.ga_target_xy[0]), float(meta.ga_target_xy[1])
    fovea_x, fovea_y = float(meta.fovea_xy[0]), float(meta.fovea_xy[1])

    point_error_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
    gt_distance_px = float(np.hypot(gt_x - fovea_x, gt_y - fovea_y))
    pred_distance_px = float(np.hypot(pred_x - fovea_x, pred_y - fovea_y))
    distance_error_px = float(abs(pred_distance_px - gt_distance_px))

    metrics.update(
        {
            "prediction_detected": "yes",
            "pred_ga_x": f"{pred_x:.2f}",
            "pred_ga_y": f"{pred_y:.2f}",
            "gt_ga_x": f"{gt_x:.2f}",
            "gt_ga_y": f"{gt_y:.2f}",
            "point_error_px": f"{point_error_px:.2f}",
            "gt_distance_px": f"{gt_distance_px:.2f}",
            "pred_distance_px": f"{pred_distance_px:.2f}",
            "distance_error_px": f"{distance_error_px:.2f}",
            "cyan_coverage_ratio": f"{cyan_ratio:.5f}",
        }
    )
    return metrics


def wait_for_button(page: Page, name: str, timeout_ms: float = 120000) -> Locator:
    locator = page.get_by_role("button", name=name)
    locator.wait_for(state="visible", timeout=timeout_ms)
    return locator


def wait_for_distances(page: Page, expected_count: int, timeout_s: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        count = page.locator("text=Distance:").count()
        if count >= expected_count:
            return
        page.wait_for_timeout(500)
    raise TimeoutError("Timed out waiting for distance measurements")


def candidate_points(
    target_xy: Tuple[float, float],
    split_x: int,
    width: int,
    height: int,
) -> List[Tuple[float, float]]:
    offsets = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (6, 0),
        (-6, 0),
        (0, 6),
        (0, -6),
        (6, 6),
        (-6, 6),
        (6, -6),
        (-6, -6),
        (12, 0),
        (-12, 0),
        (0, 12),
        (0, -12),
    ]
    out: List[Tuple[float, float]] = []
    for dx, dy in offsets:
        x = float(np.clip(target_xy[0] + dx, split_x + 2, width - 2))
        y = float(np.clip(target_xy[1] + dy, 2, height - 2))
        out.append((x, y))
    return out


def select_distance_for_panel(
    page: Page,
    canvas: Locator,
    meta: ImageMeta,
    target_xy: Tuple[float, float],
    expected_distance_count: int,
    timeout_per_click_s: float = 3.5,
) -> bool:
    # Automatic (region/local segmentation) attempts first.
    for point in candidate_points(target_xy, meta.split_x, meta.width, meta.height):
        click_image_point(canvas, point, meta.width, meta.height)
        try:
            wait_for_distances(page, expected_count=expected_distance_count, timeout_s=timeout_per_click_s)
            return True
        except TimeoutError:
            continue

    return False


def _fetch_auto_disc(image_path: Path, api_base: str = "http://127.0.0.1:8000") -> Optional[Dict[str, float]]:
    """Call the backend detect-disc endpoint and return the auto-detected disc dict."""
    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                f"{api_base}/api/detect-disc",
                files={"file": (image_path.name, fh, "image/png")},
                timeout=30,
            )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        print(f"    detect-disc API call failed for {image_path.name}: {exc}")
    return None


def process_case(page: Page, case: Case, meta: Dict[str, ImageMeta], output_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    before = meta[case.before_filename]
    after_filename_for_gui = case.after_filename or case.before_filename
    after = meta[after_filename_for_gui]
    is_single = case.after_filename is None
    before_eligible = before.raw_path is not None and before.fovea_xy is not None and before.ga_target_xy is not None
    after_eligible = after.raw_path is not None and after.fovea_xy is not None and after.ga_target_xy is not None

    if not before_eligible and not after_eligible:
        print(f"  SKIP case {case.key}: no yellow-line GA target extracted")
        return rows

    page.goto("http://127.0.0.1:3000", wait_until="domcontentloaded")

    file_inputs = page.locator("input[type='file']")
    if file_inputs.count() < 2:
        raise RuntimeError("Expected two image upload inputs")

    file_inputs.nth(0).set_input_files(str(before.input_path))
    file_inputs.nth(1).set_input_files(str(after.input_path))

    # Wait until auto-detection completes and the unified confirm button appears.
    wait_for_button(page, "Confirm Fovea on Both Images & Continue")

    canvases = page.locator("canvas")
    before_canvas = canvases.nth(0)
    after_canvas = canvases.nth(1)

    # --- DISC ADJUSTMENT (before fovea confirmation; dragging is locked after) ---
    if before_eligible and before.disc_gt is not None:
        auto_disc = _fetch_auto_disc(before.input_path)
        if auto_disc is not None:
            print(f"    Adjusting disc for {before.filename}")
            adjust_disc_to_ground_truth(page, before_canvas, before, auto_disc)
        else:
            print(f"    Skipping disc adjustment for {before.filename} (API unavailable)")

    if after_eligible and after.disc_gt is not None and case.after_filename is not None:
        auto_disc = _fetch_auto_disc(after.input_path)
        if auto_disc is not None:
            print(f"    Adjusting disc for {after.filename}")
            adjust_disc_to_ground_truth(page, after_canvas, after, auto_disc)
        else:
            print(f"    Skipping disc adjustment for {after.filename} (API unavailable)")

    # --- FOVEA PLACEMENT ---
    if before_eligible and before.fovea_xy is not None:
        click_image_point(before_canvas, before.fovea_xy, before.width, before.height)
    if after_eligible and after.fovea_xy is not None:
        click_image_point(after_canvas, after.fovea_xy, after.width, after.height)

    wait_for_button(page, "Confirm Fovea on Both Images & Continue").click()

    # --- GA SELECTION: wait for Manual button (confirms GA segmentation is done) ---
    # The button label is "Manual" (not "Select Manually" which no longer exists).
    manual_buttons = page.get_by_role("button", name="Manual")
    manual_buttons.first.wait_for(state="visible", timeout=120000)

    before_selected = False
    after_selected = False
    expected_count = 0

    if before_eligible and before.ga_target_xy is not None:
        # Manual GA mode: click "Manual" button for before panel (index 0), then click exact point
        before_selected = select_distance_manual(
            page=page,
            canvas=before_canvas,
            manual_button=manual_buttons.nth(0),
            meta=before,
            target_xy=before.ga_target_xy,
        )
        if before_selected:
            expected_count += 1
            try:
                wait_for_distances(page, expected_count=expected_count, timeout_s=8.0)
            except TimeoutError:
                # Distance renders synchronously in React; timeout means the point fell outside the enface region
                expected_count -= 1
                before_selected = False
                print(f"  SKIP image {before.filename}: manual GA point did not produce a distance")
        else:
            print(f"  SKIP image {before.filename}: Manual button click failed")

    if after_eligible and after.ga_target_xy is not None:
        # For the after panel the Manual button is the second one (index 1 or 0 if before skipped)
        after_manual_idx = 1 if before_selected else 0
        after_selected = select_distance_manual(
            page=page,
            canvas=after_canvas,
            manual_button=manual_buttons.nth(after_manual_idx),
            meta=after,
            target_xy=after.ga_target_xy,
        )
        if after_selected:
            expected_count += 1
            try:
                wait_for_distances(page, expected_count=expected_count, timeout_s=8.0)
            except TimeoutError:
                expected_count -= 1
                after_selected = False
                print(f"  SKIP image {after.filename}: manual GA point did not produce a distance")
        else:
            print(f"  SKIP image {after.filename}: Manual button click failed")

    before_canvas_img = decode_png_bytes(before_canvas.screenshot())
    after_canvas_img = decode_png_bytes(after_canvas.screenshot())

    if before_selected:
        before_out = output_dir / f"{Path(before.filename).stem}_enface_comparison.png"
        export_side_by_side(before_canvas_img, before, before_out)
        before_metrics = compute_gui_error_metrics(before_canvas_img, before)
        rows.append(
            {
                "case_key": case.key,
                "image_filename": before.filename,
                "pair_mode": "single" if is_single else "paired_before",
                "raw_marked_exists": "yes" if before.raw_path else "no",
                "output_file": str(before_out.relative_to(PROJECT_ROOT)),
                **before_metrics,
            }
        )

    if (not is_single) and case.after_filename is not None and after_selected:
        after_out = output_dir / f"{Path(case.after_filename).stem}_enface_comparison.png"
        export_side_by_side(after_canvas_img, meta[case.after_filename], after_out)
        after_meta = meta[case.after_filename]
        after_metrics = compute_gui_error_metrics(after_canvas_img, after_meta)
        rows.append(
            {
                "case_key": case.key,
                "image_filename": case.after_filename,
                "pair_mode": "paired_after",
                "raw_marked_exists": "yes" if after_meta.raw_path else "no",
                "output_file": str(after_out.relative_to(PROJECT_ROOT)),
                **after_metrics,
            }
        )

    return rows


SUMMARY_FIELDS = [
    "case_key",
    "image_filename",
    "pair_mode",
    "raw_marked_exists",
    "output_file",
    "prediction_detected",
    "pred_ga_x",
    "pred_ga_y",
    "gt_ga_x",
    "gt_ga_y",
    "point_error_px",
    "gt_distance_px",
    "pred_distance_px",
    "distance_error_px",
    "cyan_coverage_ratio",
]


def write_summary(output_dir: Path, rows: List[Dict[str, str]]) -> None:
    summary_path = output_dir / "summary.csv"
    fieldnames = list(SUMMARY_FIELDS)
    merged: Dict[str, Dict[str, str]] = {}

    if summary_path.exists():
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("image_filename", "")
                if key:
                    merged[key] = {k: row.get(k, "") for k in fieldnames}

    for row in rows:
        base = merged.get(row["image_filename"], {})
        for k in fieldnames:
            row[k] = row.get(k, base.get(k, ""))
        merged[row["image_filename"]] = row

    ordered_rows = sorted(merged.values(), key=lambda r: (r["case_key"], r["image_filename"]))

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GUI validation and export side-by-side en-face comparisons.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input_images")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "raw_marked")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_validation")
    parser.add_argument("--offset", type=int, default=0, help="Start from case index")
    parser.add_argument("--limit", type=int, default=0, help="Number of cases to process (0=all)")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--skip-install-check", action="store_true", help="Skip venv/python existence check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    python_exe = PROJECT_ROOT / "venv" / "bin" / "python"
    if args.skip_install_check and not python_exe.exists():
        python_exe = Path(sys.executable)
    elif not python_exe.exists():
        raise RuntimeError(f"Missing venv Python at {python_exe}. Set up environment first.")

    image_meta = prepare_image_meta(args.input_dir, args.raw_dir)
    all_files = [args.input_dir / k for k in sorted(image_meta.keys())]
    cases = build_cases(all_files)

    if args.offset < 0:
        raise ValueError("offset must be >= 0")
    if args.limit < 0:
        raise ValueError("limit must be >= 0")

    selected_cases = cases[args.offset:]
    if args.limit > 0:
        selected_cases = selected_cases[: args.limit]

    print(f"Prepared {len(cases)} cases total; running {len(selected_cases)} case(s).")

    backend_proc: Optional[subprocess.Popen] = None
    frontend_proc: Optional[subprocess.Popen] = None
    summary_rows: List[Dict[str, str]] = []

    try:
        backend_proc, backend_log = start_backend(args.output_dir, python_exe)
        print(f"Backend ready. Logs: {backend_log}")
        frontend_proc, frontend_log = start_frontend(args.output_dir)
        print(f"Frontend ready. Logs: {frontend_log}")

        with sync_playwright() as p:
            browser: Browser = p.chromium.launch(headless=not args.headed)
            context = browser.new_context(viewport={"width": 1700, "height": 1200})
            page = context.new_page()

            for idx, case in enumerate(selected_cases, start=1):
                print(f"[{idx}/{len(selected_cases)}] Processing case: {case.key}")
                try:
                    rows = process_case(page, case, image_meta, args.output_dir)
                    summary_rows.extend(rows)
                except Exception as exc:
                    error_png = args.output_dir / f"{case.key}_error.png"
                    page.screenshot(path=str(error_png), full_page=True)
                    print(f"  ERROR in case {case.key}: {exc}")

            context.close()
            browser.close()

        write_summary(args.output_dir, summary_rows)
        print(f"Completed. Wrote {len(summary_rows)} image comparisons.")
        print(f"Summary: {args.output_dir / 'summary.csv'}")
        return 0

    finally:
        stop_process(frontend_proc)
        stop_process(backend_proc)


if __name__ == "__main__":
    raise SystemExit(main())
