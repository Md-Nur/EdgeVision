from typing import Tuple, Dict, Any, List
import numpy as np
from PIL import Image


def segment_leaf_foreground(image_np: np.ndarray, threshold_factor: float = 0.05) -> np.ndarray:
    """
    Segment the leaf foreground from background using Excess Green Index (ExG)
    and color contrast. Suitable for plant pathology imagery with plain or varied backgrounds.
    ExG = 2 * G - R - B.
    Returns binary mask (H, W) where True = leaf foreground.
    """
    img_float = image_np.astype(np.float32) / 255.0
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Excess green index + dark foreground exclusion for diseased necrotic lesions
    exg = 2.0 * g - r - b
    # Also include necrotic/brown regions: R > B and high saturation or non-white background
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    # Mask out uniform white or near-black boundaries
    non_white = (r < 0.92) | (g < 0.92) | (b < 0.92)
    non_black = (r > 0.05) | (g > 0.05) | (b > 0.05)

    mask = ((exg > threshold_factor) | (r > b * 1.1)) & non_white & non_black

    # If mask is empty, fall back to non-white pixels
    if mask.sum() == 0:
        mask = non_white & non_black
    if mask.sum() == 0:
        mask = np.ones_like(gray, dtype=bool)

    return mask


def compute_xai_metrics(cam: np.ndarray, image_pil: Image.Image) -> Dict[str, float]:
    """
    Compute quantitative Explainable AI metrics:
      1. Energy Concentration: Proportion of CAM energy inside the leaf foreground.
      2. Pointing Game Hit: Whether the peak saliency point falls on the leaf foreground.
    """
    w, h = image_pil.size
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))

    img_np = np.array(image_pil.convert("RGB"))
    leaf_mask = segment_leaf_foreground(img_np)

    total_energy = float(cam_resized.sum())
    if total_energy > 0:
        foreground_energy = float(cam_resized[leaf_mask].sum())
        energy_concentration = foreground_energy / total_energy
    else:
        energy_concentration = 0.0

    # Peak saliency coordinate
    peak_idx = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
    pointing_game_hit = float(leaf_mask[peak_idx])

    return {
        "energy_concentration": float(energy_concentration),
        "pointing_game_hit": float(pointing_game_hit),
    }


def aggregate_xai_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-sample XAI metrics into dataset-wide summary statistics."""
    if not results:
        return {"mean_energy_concentration": 0.0, "pointing_game_accuracy": 0.0}

    concs = [r["energy_concentration"] for r in results]
    hits = [r["pointing_game_hit"] for r in results]

    return {
        "mean_energy_concentration": float(np.mean(concs)),
        "std_energy_concentration": float(np.std(concs)),
        "pointing_game_accuracy": float(np.mean(hits)),
    }
