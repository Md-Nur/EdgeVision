from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import timm


class CropDiseaseClassifier(nn.Module):
    """
    Transfer learning classifier supporting EfficientNet, ResNet, etc.
    Provides fine-grained methods for two-phase training (freezing/unfreezing)
    and discriminative learning rates.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 15,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Create backbone from timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the last convolutional feature map layer for Grad-CAM."""
        name = self.backbone_name.lower()
        if "efficientnet" in name:
            # In timm efficientnet, conv_head is the last conv before pooling
            if hasattr(self.model, "conv_head"):
                return self.model.conv_head
            elif hasattr(self.model, "blocks"):
                return self.model.blocks[-1]
        elif "resnet" in name:
            # In resnet, layer4 is the last residual stage
            if hasattr(self.model, "layer4"):
                return self.model.layer4[-1]

        # Fallback: find the last Conv2d module
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise AttributeError(f"Could not automatically determine CAM layer for {self.backbone_name}")

    def freeze_backbone(self) -> None:
        """Phase 1: Freeze backbone, train only the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier parameters
        classifier = self.model.get_classifier()
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True
        elif isinstance(classifier, nn.Parameter):
            classifier.requires_grad = True

    def unfreeze_top_layers(self, ratio: float = 0.3) -> None:
        """Phase 2: Unfreeze top portion (ratio) of backbone layers + classifier."""
        # First ensure all parameters are frozen
        all_params = list(self.model.named_parameters())
        total_params = len(all_params)
        cutoff = int(total_params * (1.0 - ratio))

        for idx, (name, param) in enumerate(all_params):
            if idx >= cutoff:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Ensure head is always trainable
        classifier = self.model.get_classifier()
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Return parameter groups with discriminative learning rates:
        lower learning rate for pretrained backbone, higher for classification head.
        """
        classifier_param_ids = set()
        classifier = self.model.get_classifier()
        if isinstance(classifier, nn.Module):
            classifier_param_ids = {id(p) for p in classifier.parameters()}
        elif isinstance(classifier, nn.Parameter):
            classifier_param_ids = {id(classifier)}

        backbone_params = []
        head_params = []

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in classifier_param_ids:
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            })
        return param_groups


def create_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = 15,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> CropDiseaseClassifier:
    """Factory function for model creation."""
    return CropDiseaseClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
