import pytest
import torch
import numpy as np
from PIL import Image

from src.models.backbones import create_model
from src.losses.focal_loss import FocalLoss, get_loss_function
from src.evaluation.metrics import compute_metrics
from src.xai.gradcam import GradCAM
from src.xai.quantitative import compute_xai_metrics, segment_leaf_foreground


def test_model_forward_and_freezing():
    model = create_model("efficientnet_b0", num_classes=15, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 15)

    # Test Phase 1 freezing
    model.freeze_backbone()
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(frozen_params) > 0
    assert len(trainable_params) > 0  # Head remains trainable

    # Test Phase 2 unfreezing
    model.unfreeze_top_layers(ratio=0.3)
    trainable_p2 = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_p2) > len(trainable_params)

    # Test discriminative parameter groups
    groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-4


def test_loss_functions():
    logits = torch.randn(4, 15, requires_grad=True)
    targets = torch.tensor([0, 5, 14, 2], dtype=torch.long)

    # Weighted Cross Entropy
    weights = torch.ones(15)
    crit_wce = get_loss_function("weighted_ce", class_weights=weights, label_smoothing=0.1)
    loss_wce = crit_wce(logits, targets)
    assert loss_wce.item() > 0

    # Focal Loss
    crit_focal = get_loss_function("focal", class_weights=weights, label_smoothing=0.1, gamma=2.0)
    loss_focal = crit_focal(logits, targets)
    assert loss_focal.item() > 0

    loss_focal.backward()
    assert logits.grad is not None


def test_metrics_computation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    names = ["C0", "C1", "C2"]

    metrics = compute_metrics(y_true, y_pred, class_names=names)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class_recall" in metrics
    assert metrics["accuracy"] == 5 / 6
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_gradcam_and_xai_metrics():
    model = create_model("efficientnet_b0", num_classes=15, pretrained=False)
    target_layer = model.get_target_layer_for_cam()
    cam_gen = GradCAM(model=model, target_layer=target_layer)

    x = torch.randn(1, 3, 224, 224)
    cam = cam_gen.generate_cam(x, target_class=0, use_gradcam_plusplus=False)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0

    cam_pp = cam_gen.generate_cam(x, target_class=0, use_gradcam_plusplus=True)
    assert cam_pp.shape == (224, 224)

    # Test quantitative metric calculation
    dummy_img = Image.new("RGB", (224, 224), color=(34, 139, 34))  # Green image
    q_metrics = compute_xai_metrics(cam, dummy_img)
    assert "energy_concentration" in q_metrics
    assert "pointing_game_hit" in q_metrics
    assert 0.0 <= q_metrics["energy_concentration"] <= 1.0

    cam_gen.remove_hooks()
