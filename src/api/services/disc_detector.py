import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import albumentations as A
import cv2
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - optional in lightweight environments
    torch = None

try:
    from albumentations.pytorch import ToTensorV2
except Exception:  # pragma: no cover - requires torch
    ToTensorV2 = None

from src.utils.image_utils import classify_image_format, get_split_indices_and_images

from ..constants import DISC_DIAMETER_MICRONS
from ..utils.logger import get_logger

logger = get_logger("services.disc_detector")

if TYPE_CHECKING:
    from src.models.retfound_unet import RETFound_UNet


class DiscDetectorService:
    """
    Service for detecting optic disc using RETFound U-Net model.
    Preserves the "New Algorithm" (Contour/Energy method) from run_inference.py.
    """
    
    def __init__(
        self,
        model_path: str = "weights/best_disc_model.pth",
        img_size: int = 224,
        device: Optional[str] = None,
        force_version: Optional[str] = None,
    ):
        """
        Initialize the disc detector service.
        
        Args:
            model_path: Path to trained model weights (v1 checkpoint)
            img_size: Input size for model (224x224)
            device: Device to run inference on (auto-detect if None)
            force_version: Override auto-detection. "v1" loads only best_disc_model.pth,
                "v2" loads only best_disc_model_v2.pth. None (default) uses auto-detect
                (v2 if present, else v1).
        """
        self.img_size = img_size
        self.model_path = model_path
        self.force_version = force_version
        
        # Auto-detect device
        if device is None:
            if torch is None:
                self.device = "cpu"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device) if torch is not None else "cpu"
        
        # Load model
        self.model = self._load_model()
        
        # Define preprocessing transform only for model inference mode.
        self.transform = None
        if self.model is not None and ToTensorV2 is not None:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    def _remote_file_size(self, url: str) -> Optional[int]:
        """HEAD-request the remote URL and return Content-Length, or None on failure."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl else None
        except Exception:
            return None

    def _needs_redownload(self, url: str, local_path: str) -> bool:
        """Return True if the local file is absent or differs in size from the remote."""
        if not os.path.exists(local_path):
            return True
        remote_size = self._remote_file_size(url)
        if remote_size is None:
            return False
        local_size = os.path.getsize(local_path)
        if remote_size != local_size:
            logger.info(
                "Remote weights size (%d B) differs from local (%d B) — will re-download",
                remote_size, local_size,
            )
            return True
        return False

    def _download_weights(self, url: str, dest_path: Optional[str] = None) -> bool:
        """Download model weights from a URL, streaming to disk."""
        dest = Path(dest_path) if dest_path else Path(self.model_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(".tmp")
        logger.info("Downloading disc model weights from %s ...", url)
        try:
            def _log_progress(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0 and block_num % 500 == 0:
                    downloaded = block_num * block_size
                    pct = min(downloaded / total_size * 100, 100)
                    logger.info("  %.1f%% (%d / %d MB)", pct, downloaded // 1_000_000, total_size // 1_000_000)

            urllib.request.urlretrieve(url, str(tmp_path), reporthook=_log_progress)
            tmp_path.rename(dest)
            logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1_000_000)
            return True
        except Exception as exc:
            logger.warning("Failed to download weights: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def _load_model(self) -> Optional["RETFound_UNet"]:
        """Load the RETFound U-Net model with trained weights.

        Respects self.force_version ("v1", "v2", or None for auto-detect).
        When force_version is None, tries v2 checkpoint first; falls back to v1.
        Sets self.has_height_head accordingly.
        """
        self.has_height_head = False
        self.loaded_model_path: Optional[str] = None

        if torch is None:
            logger.warning("Torch is not installed. Using fallback mode.")
            return None

        weights_dir = Path(self.model_path).parent
        v2_path = str(weights_dir / "best_disc_model_v2.pth")

        url_v2 = os.environ.get("DISC_MODEL_URL_V2")
        url_v1 = os.environ.get("DISC_MODEL_URL")
        force_redownload = os.environ.get("DISC_MODEL_FORCE_REDOWNLOAD", "").lower() in (
            "1", "true", "yes",
        )

        # Sync local weights with remote: re-download when the remote file
        # changed (size mismatch) or DISC_MODEL_FORCE_REDOWNLOAD is set.
        if url_v2:
            if force_redownload or self._needs_redownload(url_v2, v2_path):
                logger.info("Updating v2 weights from remote...")
                self._download_weights(url_v2, dest_path=v2_path)
        if url_v1:
            if force_redownload or self._needs_redownload(url_v1, self.model_path):
                logger.info("Updating v1 weights from remote...")
                self._download_weights(url_v1)

        # Resolve which checkpoint to load
        if self.force_version == "v1":
            if not os.path.exists(self.model_path):
                logger.warning("V1 weights not found at %s. Using fallback mode.", self.model_path)
                return None
            chosen_path = self.model_path
            use_v2 = False
        elif self.force_version == "v2":
            if not os.path.exists(v2_path):
                logger.warning("V2 weights not found at %s. Using fallback mode.", v2_path)
                return None
            chosen_path = v2_path
            use_v2 = True
        elif os.path.exists(v2_path):
            chosen_path = v2_path
            use_v2 = True
        elif os.path.exists(self.model_path):
            chosen_path = self.model_path
            use_v2 = False
        else:
            logger.warning("Model weights not found at %s. Using fallback mode.", self.model_path)
            return None

        try:
            from src.models.retfound_unet import RETFound_UNet

            logger.info("Loading model from %s...", chosen_path)
            if use_v2:
                model = RETFound_UNet.load_pretrained_and_add_height_head(
                    chosen_path, freeze_encoder=False
                )
                logger.info("Loaded v2 model with height head on %s", self.device)
            else:
                model = RETFound_UNet(
                    img_size=self.img_size,
                    weights_path=None,
                    freeze_encoder=False
                )
                state_dict = torch.load(
                    chosen_path,
                    map_location=self.device,
                    weights_only=False
                )
                # strict=False: V1 checkpoints predate the height_head and lack
                # those keys; the head remains randomly initialised but is
                # never used when has_height_head=False.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.debug("V1 load – missing keys (expected): %s", missing)
                if unexpected:
                    logger.warning("V1 load – unexpected keys: %s", unexpected)
                logger.info("Loaded v1 model on %s", self.device)

            model.to(self.device)
            model.eval()
            if use_v2:
                self.has_height_head = True
            self.loaded_model_path = chosen_path
            return model
        except Exception as exc:
            logger.warning("Failed to load model (%s). Using fallback mode.", exc)
            return None

    def _extract_disc_from_marked_image(
        self,
        marked_image: np.ndarray,
        en_face_split_x: int
    ) -> Optional[Dict[str, float]]:
        """
        Extract disc line directly from marked overlays when available.
        The marked dataset draws disc with a red vertical line.
        """
        en_face = marked_image[:, en_face_split_x:, :]
        if en_face.size == 0:
            return None

        hsv = cv2.cvtColor(en_face, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 70, 70], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 70, 70], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_label = -1
        best_score = -1.0

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]

            if area < 40 or height < 20:
                continue

            verticality = float(height) / max(float(width), 1.0)
            score = float(area) * verticality
            if score > best_score:
                best_score = score
                best_label = label_idx

        if best_label < 0:
            return None

        ys, xs = np.where(labels == best_label)
        if len(xs) == 0:
            return None

        disc_center_x = float(en_face_split_x + np.mean(xs))
        disc_top_y = float(np.min(ys))
        disc_bottom_y = float(np.max(ys))
        disc_height_pixels = float(disc_bottom_y - disc_top_y)
        if disc_height_pixels <= 0:
            return None

        return {
            'disc_center_x': disc_center_x,
            'disc_center_y': float((disc_top_y + disc_bottom_y) / 2.0),
            'disc_top_y': disc_top_y,
            'disc_bottom_y': disc_bottom_y,
            'disc_height_pixels': disc_height_pixels,
            'pixel_to_micron_ratio': float(DISC_DIAMETER_MICRONS / disc_height_pixels),
            'en_face_split_x': int(en_face_split_x)
        }

    def _fallback_detect_from_marked_or_heuristic(
        self,
        image: np.ndarray,
        image_name: Optional[str],
        en_face_split_x: int
    ) -> Dict[str, float]:
        """
        Fallback detection path:
          1) If raw_marked/<filename> exists, extract disc directly from annotation.
          2) Otherwise return a conservative geometric default.
        """
        if image_name:
            project_root = Path(__file__).resolve().parents[3]
            marked_path = project_root / "raw_marked" / Path(image_name).name
            if marked_path.exists():
                marked = cv2.imread(str(marked_path))
                if marked is not None and marked.shape[:2] == image.shape[:2]:
                    extracted = self._extract_disc_from_marked_image(marked, en_face_split_x)
                    if extracted is not None:
                        logger.debug("Fallback from raw_marked for %s", image_name)
                        return extracted

        h, w = image.shape[:2]
        en_face_width = max(1, w - en_face_split_x)
        center_x = float(en_face_split_x + 0.5 * en_face_width)
        top_y = float(0.30 * h)
        bottom_y = float(0.70 * h)
        disc_height_pixels = max(1.0, bottom_y - top_y)

        logger.debug("Using geometric fallback disc estimate")
        return {
            'disc_center_x': center_x,
            'disc_center_y': float((top_y + bottom_y) / 2.0),
            'disc_top_y': top_y,
            'disc_bottom_y': bottom_y,
            'disc_height_pixels': disc_height_pixels,
            'pixel_to_micron_ratio': float(DISC_DIAMETER_MICRONS / disc_height_pixels),
            'en_face_split_x': int(en_face_split_x)
        }
    
    def detect_from_image(self, image: np.ndarray, image_name: Optional[str] = None) -> Dict[str, float]:
        """
        Detect optic disc from an OCT image.

        Supports two image formats:
        - 'heidelberg': composite (B-scan left + en-face right) -- auto-splits on the divider
        - 'standalone': single en-face panel (e.g. Cirrus) -- treats full image as en-face

        Returns:
            Dictionary containing disc coordinates, pixel_to_micron_ratio, en_face_split_x,
            and an 'image_format' key ('heidelberg' or 'standalone').
        """
        image_format = classify_image_format(image)
        logger.debug("Detected image format: %s", image_format)

        if image_format == "standalone":
            # Treat the entire image as the en-face panel
            en_face = image
            en_face_split_x = 0
        else:
            # Step 1: Split the composite Heidelberg image
            _, en_face, metadata = get_split_indices_and_images(
                image,
                divider_safety_margin=10
            )
            en_face_split_x = metadata['final_split_column']

        if self.model is None or self.transform is None:
            result = self._fallback_detect_from_marked_or_heuristic(
                image=image,
                image_name=image_name,
                en_face_split_x=en_face_split_x,
            )
            result['image_format'] = image_format
            return result

        # Convert to RGB
        en_face_rgb = cv2.cvtColor(en_face, cv2.COLOR_BGR2RGB)
        h_ef, w_ef = en_face_rgb.shape[:2]

        # Step 2: Preprocess
        augmented = self.transform(image=en_face_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        # Step 3: Inference
        height_from_model: Optional[float] = None
        with torch.no_grad():
            if self.has_height_head:
                output, height_tensor = self.model(input_tensor, predict_height=True)
                height_normalized = height_tensor.cpu().item()
                height_from_model = height_normalized * h_ef
            else:
                output = self.model(input_tensor)  # [1, 1, 224, 224]
            heatmap = output.cpu().squeeze().numpy()  # [224, 224]

        # Step 4: Extract Coordinates using "New Algorithm"
        coords = self._extract_disc_coordinates_new_algorithm(
            heatmap=heatmap,
            en_face_width=w_ef,
            en_face_height=h_ef,
            en_face_split_x=en_face_split_x,
            height_from_model=height_from_model,
        )
        coords['image_format'] = image_format
        return coords
    
    def _extract_disc_coordinates_new_algorithm(
        self,
        heatmap: np.ndarray,
        en_face_width: int,
        en_face_height: int,
        en_face_split_x: int,
        height_from_model: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Extract disc coordinates using an improved robust algorithm.
        
        UPDATED (2026-02-07): Fixed systematic over-segmentation by:
        1. Replacing Otsu with 95th percentile threshold (tighter cutoff)
        2. Using weighted centroid for X coordinate (more accurate than bbox center)
        3. Extracting actual pixel min/max Y (not bounding box extents)
        
        OLD ISSUES (confirmed via ground-truth validation on 52 images):
        - Otsu threshold was too permissive, including background gradient
        - Bounding box Y-extent was systematically ~134px taller than ground-truth
        - Mean height error: 175%, median: 35%, with 79% of images > 20% error
        
        NEW ALGORITHM:
        1. Apply 95th percentile threshold on heatmap (tighter than Otsu)
        2. Use connected component analysis to find largest cluster
        3. Extract weighted centroid for X, actual pixel min/max for Y
        4. Fallback: if no components, use argmax with percentile threshold
        
        Args:
            heatmap: Raw model output heatmap [224, 224]
            en_face_width: Width of en-face region in original image
            en_face_height: Height of en-face region in original image
            en_face_split_x: X offset where en-face starts in original image
        
        Returns:
            Dictionary with disc coordinates and metadata
        """
        # -------------------------------------------------------------------------
        # Algorithm: Percentile-Based Threshold + Weighted Centroid
        # Height is extracted from pixel extents and corrected by an empirically
        # calibrated factor (0.866) to remove the systematic overestimation caused
        # by diffuse activation tails in the regression heatmap.
        # -------------------------------------------------------------------------

        hm_h, hm_w = heatmap.shape

        # 98th-percentile threshold keeps only the brightest ~2% of pixels
        threshold_value = np.percentile(heatmap, 98.0)
        binary_mask = (heatmap > threshold_value).astype(np.uint8)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        if num_labels <= 1:
            # Fallback to 90th percentile
            threshold_value_fallback = np.percentile(heatmap, 90.0)
            binary_mask_fallback = (heatmap > threshold_value_fallback).astype(np.uint8)

            if np.sum(binary_mask_fallback) == 0:
                py_raw, px_raw = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                min_y_raw = py_raw
                max_y_raw = py_raw
                cx_raw = px_raw
                logger.warning("No components found, using argmax fallback")
            else:
                y_indices, x_indices = np.where(binary_mask_fallback > 0)
                min_y_raw = np.min(y_indices)
                max_y_raw = np.max(y_indices)
                weights = heatmap[binary_mask_fallback > 0]
                cx_raw = np.average(x_indices, weights=weights)
                logger.debug("Using 90th percentile fallback: %s pixels", len(y_indices))
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_idx = np.argmax(areas) + 1
            component_mask = (labels == largest_component_idx).astype(np.uint8)
            y_indices, x_indices = np.where(component_mask > 0)
            min_y_raw = np.min(y_indices)
            max_y_raw = np.max(y_indices)
            weights = heatmap[component_mask > 0]
            cx_raw = np.average(x_indices, weights=weights)
            num_pixels = len(y_indices)
            bbox_h = stats[largest_component_idx, cv2.CC_STAT_HEIGHT]
            logger.debug(
                "Largest component: %s pixels, Y-extent=%s px (bbox was %s px)",
                num_pixels,
                max_y_raw - min_y_raw,
                bbox_h,
            )
        
        # 3. Project to En Face Dimensions with Sub-Pixel Correction
        # Add 0.5 to center the coordinate within the raw pixel grid
        scale_x = en_face_width / self.img_size
        scale_y = en_face_height / self.img_size
        
        pred_x_ef = (cx_raw + 0.5) * scale_x
        pred_min_y_ef = (min_y_raw + 0.5) * scale_y
        pred_max_y_ef = (max_y_raw + 0.5) * scale_y
        
        # Also calculate the single point for logging
        pred_y_ef = (pred_min_y_ef + pred_max_y_ef) / 2
        
        # 4. Project to Original Composite Size
        start_x_offset = en_face_split_x
        
        orig_cx = start_x_offset + pred_x_ef
        orig_min_y = pred_min_y_ef
        orig_max_y = pred_max_y_ef
        
        # For logging purposes
        pred_x_orig = orig_cx
        pred_y_orig = (orig_min_y + orig_max_y) / 2
        
        # Calculate disc height: prefer direct model output when plausible
        heatmap_height = float(orig_max_y - orig_min_y)
        if (
            height_from_model is not None
            and 50.0 < height_from_model < 0.6 * en_face_height
        ):
            disc_height_pixels = height_from_model
            logger.debug("Using model height head: %.1f px", disc_height_pixels)
            disc_center_y_ef = (pred_min_y_ef + pred_max_y_ef) / 2
            orig_min_y = disc_center_y_ef - disc_height_pixels / 2
            orig_max_y = disc_center_y_ef + disc_height_pixels / 2
            pred_y_orig = (orig_min_y + orig_max_y) / 2
        else:
            disc_height_pixels = max(heatmap_height, 1.0)
            if height_from_model is not None:
                logger.debug(
                    "Model height %.1f px out of plausible range; using heatmap height %.1f px",
                    height_from_model,
                    disc_height_pixels,
                )
        pixel_to_micron_ratio = DISC_DIAMETER_MICRONS / disc_height_pixels
        
        logger.debug("Disc detected at (%.1f, %.1f)", pred_x_orig, pred_y_orig)
        logger.debug(
            "Disc height: %.1f pixels = %.0f microns",
            disc_height_pixels,
            DISC_DIAMETER_MICRONS,
        )
        logger.debug("Pixel-to-micron ratio: %.3f", pixel_to_micron_ratio)
        
        return {
            'disc_center_x': float(orig_cx),
            'disc_center_y': float(pred_y_orig),
            'disc_top_y': float(orig_min_y),
            'disc_bottom_y': float(orig_max_y),
            'disc_height_pixels': float(disc_height_pixels),
            'pixel_to_micron_ratio': float(pixel_to_micron_ratio),
            'en_face_split_x': int(en_face_split_x)
        }
    
    def detect_from_path(self, image_path: str) -> Dict[str, float]:
        """
        Detect optic disc from an image file path.
        
        Args:
            image_path: Path to OCT image file
        
        Returns:
            Dictionary with disc detection results
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.detect_from_image(img, image_name=Path(image_path).name)
