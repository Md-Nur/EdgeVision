import numpy as np
import pytest
from PIL import Image
import torch
import torch.nn as nn

from src.stress_test import (
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_brightness_shift,
    apply_contrast_shift,
    apply_random_shadow,
    apply_jpeg_compression,
    apply_background_replacement,
    CORRUPTION_FUNCTIONS,
)
from src.xai.quantitative import segment_leaf_foreground
from src.xai.gradcam import GradCAM


def test_corruption_functions_dimensions_and_types():
    """Verify all 7 corruptions maintain image shape and return valid PIL images."""
    w, h = 120, 100
    img = Image.new("RGB", (w, h), color=(100, 180, 70))

    for corr_name, fn in CORRUPTION_FUNCTIONS.items():
        for severity in [1, 2, 3]:
            if fn in (apply_gaussian_noise, apply_random_shadow, apply_background_replacement):
                out = fn(img, severity=severity, seed=42)
            else:
                out = fn(img, severity=severity)
            assert isinstance(out, Image.Image), f"{corr_name} sev {severity} did not return PIL Image"
            assert out.size == (w, h), f"{corr_name} sev {severity} altered size: {out.size} != {(w, h)}"
            arr = np.array(out)
            assert arr.shape == (h, w, 3)
            assert arr.dtype == np.uint8


def test_spurious_attention_mathematical_fraction():
    """Verify spurious attention calculation accurately computes energy outside leaf mask."""
    h, w = 10, 10
    leaf_mask = np.zeros((h, w), dtype=bool)
    leaf_mask[:5, :] = True  # Top half is leaf, bottom half is background

    cam = np.zeros((h, w), dtype=np.float32)
    # Put 3 units of attention on leaf (top half) and 1 unit on background (bottom half)
    cam[2, 2] = 3.0
    cam[7, 7] = 1.0

    total_mass = float(cam.sum())
    bg_mass = float(cam[~leaf_mask].sum())
    spurious_ratio = bg_mass / total_mass

    assert np.isclose(total_mass, 4.0)
    assert np.isclose(bg_mass, 1.0)
    assert np.isclose(spurious_ratio, 0.25)


def test_gradcam_tensor_hook_toy_model():
    """Verify GradCAM generates normalized [0, 1] heatmaps using tensor gradient hooks."""
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SimpleConvNet().eval()
    cam_gen = GradCAM(model, model.conv)

    x = torch.randn(1, 3, 32, 32)
    cam_map = cam_gen.generate_cam(x, target_class=1, use_gradcam_plusplus=True)

    assert cam_map.shape == (32, 32)
    assert cam_map.min() >= 0.0
    assert cam_map.max() <= 1.0
    cam_gen.remove_hooks()
