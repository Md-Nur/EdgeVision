"""
Custom Lightweight CNN for Multi-Crop Leaf Disease Diagnosis.

Trained from scratch (random initialization, NO pretrained weights).
Architecture Design:
- Small stem (Conv3x3 stride 2 + BN + Hardswish/ReLU6)
- 4 stages of depthwise-separable conv blocks (BatchNorm + Hardswish/ReLU6)
- Optional attention module after each stage: none | se | cbam
- Global Average Pooling (GAP) + Dropout
- Two heads:
  1. Crop head (3 crops: Papaya, Potato, Rice)
  2. Disease head (12 classes)
- Masking at inference: restricts disease prediction to classes of the predicted crop
- Target: Under 1.5M parameters
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


# Mapping of each crop to its valid disease label indices
# Papaya: 0..3, Potato: 4..7, Rice: 8..11
CROP_TO_CLASSES = {
    0: [0, 1, 2, 3],       # Papaya
    1: [4, 5, 6, 7],       # Potato
    2: [8, 9, 10, 11],     # Rice
}
CLASS_TO_CROP = {idx: crop for crop, classes in CROP_TO_CLASSES.items() for idx in classes}


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention module."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w


class ChannelAttention(nn.Module):
    """Channel attention sub-module for CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module for CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(scale))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial Attention)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise-Separable Convolutional Block with inverted bottleneck expansion,
    depthwise 3x3 conv, pointwise 1x1 conv, and residual skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
        act_layer: type = nn.Hardswish,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expansion phase (pointwise conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_layer(inplace=True),
            ])

        # Depthwise conv + Pointwise projection
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            act_layer(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class CustomNet(nn.Module):
    """
    Custom Lightweight CNN trained from scratch (NO pretrained weights).
    Features:
    - 4 stages of depthwise-separable conv blocks
    - Optional attention after each stage: 'none' | 'se' | 'cbam'
    - Dual heads: crop_head (3 crops) and disease_head (12 classes)
    - Under 1.5M parameters
    """

    def __init__(
        self,
        attention: str = "none",
        num_crops: int = 3,
        num_diseases: int = 12,
        dropout: float = 0.3,
        act_layer: type = nn.Hardswish,
    ):
        super().__init__()
        self.attention_type = attention.lower().strip()
        assert self.attention_type in ("none", "se", "cbam"), (
            f"Unsupported attention: {attention}. Choose from 'none', 'se', 'cbam'."
        )
        self.num_crops = num_crops
        self.num_diseases = num_diseases

        # 1. Stem: Conv3x3 stride 2 -> 112x112
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            act_layer(inplace=True),
        )

        # 2. 4 Stages of Depthwise-Separable Conv Blocks
        stage_channels = [32, 48, 96, 160, 256]
        self.stages = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(4):
            in_c = stage_channels[i]
            out_c = stage_channels[i + 1]

            # Each stage has a downsampling block (stride 2) and an extraction block (stride 1)
            stage_blocks = [
                DepthwiseSeparableConv(in_c, out_c, stride=2, expand_ratio=2, act_layer=act_layer),
                DepthwiseSeparableConv(out_c, out_c, stride=1, expand_ratio=2, act_layer=act_layer),
            ]
            self.stages.append(nn.Sequential(*stage_blocks))

            # Optional attention module after each stage
            if self.attention_type == "se":
                self.attentions.append(SqueezeExcitation(out_c, reduction=16))
            elif self.attention_type == "cbam":
                self.attentions.append(CBAM(out_c, reduction=16))
            else:
                self.attentions.append(nn.Identity())

        # 3. Pooling & Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        # Dual Heads
        self.crop_head = nn.Linear(stage_channels[-1], num_crops)
        self.disease_head = nn.Linear(stage_channels[-1], num_diseases)

        # Weight initialization for from-scratch training
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (crop_logits, disease_logits)."""
        x = self.stem(x)
        for stage, att in zip(self.stages, self.attentions):
            x = stage(x)
            x = att(x)
        feat = self.gap(x).flatten(1)
        feat = self.dropout(feat)

        crop_logits = self.crop_head(feat)
        disease_logits = self.disease_head(feat)
        return crop_logits, disease_logits

    def predict_with_masking(
        self,
        crop_logits: torch.Tensor,
        disease_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical crop masking at inference:
        Restricts the disease prediction to the 4 classes belonging to the predicted crop.

        Returns:
            pred_crops: (B,) predicted crop indices (0..2)
            flat_disease_preds: (B,) unmasked argmax over all 12 disease classes
            masked_disease_preds: (B,) masked argmax restricted to predicted crop classes
        """
        pred_crops = torch.argmax(crop_logits, dim=-1)
        flat_disease_preds = torch.argmax(disease_logits, dim=-1)

        # Build mask for valid classes: 0..3 for crop 0, 4..7 for crop 1, 8..11 for crop 2
        masked_logits = disease_logits.clone()
        class_crop_map = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            device=disease_logits.device,
            dtype=torch.long,
        )

        # valid_mask is True where class matches predicted crop
        valid_mask = class_crop_map.unsqueeze(0) == pred_crops.unsqueeze(1)
        masked_logits = masked_logits.masked_fill(~valid_mask, -1e9)
        masked_disease_preds = torch.argmax(masked_logits, dim=-1)

        return pred_crops, flat_disease_preds, masked_disease_preds

    def get_parameter_count(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_flops(self, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> int:
        """Measure FLOPs via PyTorch FlopCounterMode."""
        x = torch.randn(*input_shape, device=next(self.parameters()).device)
        with FlopCounterMode(display=False) as fcm:
            _ = self(x)
        return int(fcm.get_total_flops())


def build_custom_net(
    attention: str = "none",
    dropout: float = 0.3,
    act_layer: type = nn.Hardswish,
) -> CustomNet:
    """Factory helper to build CustomNet with validated configuration."""
    model = CustomNet(
        attention=attention,
        num_crops=3,
        num_diseases=12,
        dropout=dropout,
        act_layer=act_layer,
    )
    params = model.get_parameter_count()
    print(f"[CustomNet Architecture]: attention={attention} | Params: {params:,} (Target < 1,500,000: {'PASS' if params < 1.5e6 else 'FAIL'})")
    return model


if __name__ == "__main__":
    print("Testing CustomNet architectures (none, se, cbam)...")
    for att in ["none", "se", "cbam"]:
        m = build_custom_net(attention=att)
        flops = m.get_flops()
        print(f"  {att.upper()} -> Params: {m.get_parameter_count():,} | FLOPs: {flops:,} ({flops/1e6:.1f} MFLOPs)")
