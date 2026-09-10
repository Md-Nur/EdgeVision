from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM and Grad-CAM++ implementation for CNN and hybrid backbones.
    Uses forward and backward hooks to capture activation maps and gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self.handles = [
            self.target_layer.register_forward_hook(self._save_activations),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        use_gradcam_plusplus: bool = False,
    ) -> np.ndarray:
        """
        Generate CAM heatmap normalized to [0, 1].
        input_tensor: (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations  # (1, C, H_feat, W_feat)
        gradients = self.gradients      # (1, C, H_feat, W_feat)

        if use_gradcam_plusplus:
            # Grad-CAM++ weighting
            # Second and third order gradients
            g2 = gradients.pow(2)
            g3 = gradients.pow(3)
            sum_act = activations.sum(dim=(2, 3), keepdim=True)
            alpha_num = g2
            alpha_denom = 2 * g2 + sum_act * g3 + 1e-7
            alphas = alpha_num / alpha_denom
            weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        image_pil: Image.Image,
        cam: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend CAM heatmap onto the original PIL image."""
        w, h = image_pil.size
        # Resize CAM to match PIL image
        cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))

        # Colorize
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap]
        except (AttributeError, KeyError):
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
        heatmap = cmap(cam_resized)[:, :, :3]  # Drop alpha channel
        heatmap = np.uint8(255 * heatmap)

        img_np = np.array(image_pil.convert("RGB"))
        blended = np.uint8(alpha * heatmap + (1.0 - alpha) * img_np)
        return blended

    def remove_hooks(self):
        """Clean up registered hooks."""
        for h in self.handles:
            h.remove()
