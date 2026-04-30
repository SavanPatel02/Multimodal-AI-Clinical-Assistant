from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from src.vision.image_utils import load_image, preprocess_xray, get_image_metadata

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


class MedicalImageEncoder(nn.Module):
    """
    DenseNet121 encoder with GradCAM heatmap generation.
    GradCAM highlights which regions of the X-ray the model focuses on
    for the top detected pathology.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._features_model: Optional[nn.Module] = None
        self._classifier: Optional[nn.Linear] = None
        self._transform: Optional[T.Compose] = None
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

    def load(self) -> None:
        if self._features_model is not None:
            return

        base = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)

        # Keep features as a separate model so we can hook into the last conv layer
        self._features_model = base.features.to(self.device).eval()

        weights_path = Path("models/vision_encoder/classifier.pt")
        self._classifier = nn.Linear(1024, len(PATHOLOGIES))
        if weights_path.exists():
            self._classifier.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
        self._classifier = self._classifier.to(self.device).eval()

        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _register_hooks(self) -> list:
        """Register forward/backward hooks on the last DenseNet block."""
        handles = []

        def save_activation(module, input, output):
            self._activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        # Last dense block is the final layer before transition/pool
        last_block = self._features_model[-1]
        handles.append(last_block.register_forward_hook(save_activation))
        handles.append(last_block.register_backward_hook(save_gradient))
        return handles

    def _compute_gradcam(
        self,
        tensor: torch.Tensor,
        class_idx: int,
        orig_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap for the given class index.
        Returns a uint8 heatmap resized to orig_size (H, W).
        """
        self._features_model.train()
        self._classifier.train()

        handles = self._register_hooks()
        try:
            tensor_grad = tensor.clone().requires_grad_(True)

            # Forward
            feat_map = self._features_model(tensor_grad)
            feat_map_relu = torch.nn.functional.relu(feat_map)
            pooled = feat_map_relu.mean(dim=[2, 3])          # global avg pool
            logits = self._classifier(pooled)

            # Backward for the target class
            self._classifier.zero_grad()
            self._features_model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

        finally:
            for h in handles:
                h.remove()
            self._features_model.eval()
            self._classifier.eval()

        if self._gradients is None or self._activations is None:
            return np.zeros((orig_size[0], orig_size[1]), dtype=np.uint8)

        # Weight activations by gradients (global average pooled)
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalise to 0-255
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_uint8 = (cam * 255).astype(np.uint8)

        # Resize to original image size
        heatmap = cv2.resize(cam_uint8, (orig_size[1], orig_size[0]))
        return heatmap

    def _overlay_heatmap(
        self,
        pil_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Blend GradCAM heatmap onto the original image.
        Only colours pixels that belong to actual image content —
        black background areas (ultrasound, MRI borders) are left untouched.
        """
        img_np = np.array(pil_img.convert("RGB"))

        # Build a content mask: pixels where grayscale > threshold are real tissue
        gray = np.array(pil_img.convert("L"))
        content_mask = gray > 20   # anything darker than this is background

        # Zero out heatmap in background regions so no colour bleeds there
        heatmap_masked = heatmap.copy()
        heatmap_masked[~content_mask] = 0

        # Normalise after masking so scale is still meaningful
        if heatmap_masked.max() > 0:
            heatmap_masked = (heatmap_masked / heatmap_masked.max() * 255).astype(np.uint8)

        colored = cv2.applyColorMap(heatmap_masked, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Blend only in content areas; keep background pixels original
        blended = cv2.addWeighted(img_np, 1 - alpha, colored_rgb, alpha, 0)
        mask_3d = np.stack([content_mask] * 3, axis=-1)
        overlay = np.where(mask_3d, blended, img_np)
        return Image.fromarray(overlay.astype(np.uint8))

    def encode(self, image_path: str | Path) -> dict:
        """
        Encode a medical image and generate a GradCAM overlay.

        Returns:
            {
                "feature_vector"    : list[float],
                "pathology_scores"  : dict[str, float],
                "top_findings"      : list[str],
                "image_metadata"    : dict,
                "gradcam_overlay"   : PIL.Image  (heatmap on original),
                "latency_s"         : float,
            }
        """
        if self._features_model is None:
            self.load()

        pil_img   = load_image(Path(image_path))
        meta      = get_image_metadata(pil_img)
        processed = preprocess_xray(pil_img, size=224)

        t0 = time.perf_counter()
        tensor = self._transform(processed).unsqueeze(0).to(self.device)

        # --- Standard forward (no grad) for scores + feature vector ---
        with torch.no_grad():
            feat_map  = self._features_model(tensor)
            feat_relu = torch.nn.functional.relu(feat_map)
            pooled    = feat_relu.mean(dim=[2, 3])   # (1, 1024)
            logits    = self._classifier(pooled)      # (1, 14)
            probs     = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        pathology_scores = {
            p: round(float(s), 4) for p, s in zip(PATHOLOGIES, probs)
        }
        top_findings = [
            p for p, s in sorted(pathology_scores.items(), key=lambda x: -x[1])
            if s > 0.3
        ][:3]

        # --- GradCAM for top pathology ---
        top_class_idx = int(np.argmax(probs))
        orig_h, orig_w = pil_img.height, pil_img.width
        heatmap = self._compute_gradcam(tensor, top_class_idx, (orig_h, orig_w))
        gradcam_overlay = self._overlay_heatmap(pil_img, heatmap, alpha=0.45)

        latency = time.perf_counter() - t0

        return {
            "feature_vector"  : pooled.squeeze(0).detach().cpu().numpy().tolist(),
            "pathology_scores": pathology_scores,
            "top_findings"    : top_findings,
            "image_metadata"  : meta,
            "gradcam_overlay" : gradcam_overlay,
            "latency_s"       : round(latency, 4),
        }


_encoder: Optional[MedicalImageEncoder] = None


def get_image_encoder(device: str = "cpu") -> MedicalImageEncoder:
    global _encoder
    if _encoder is None:
        _encoder = MedicalImageEncoder(device=device)
        _encoder.load()
    return _encoder
