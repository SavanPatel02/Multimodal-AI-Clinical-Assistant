from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}


def load_image(path: str | Path) -> Image.Image:
    """Load any supported medical image and return a PIL RGB image."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        except ImportError:
            raise RuntimeError("pydicom required for DICOM files: pip install pydicom")
    elif suffix in SUPPORTED_FORMATS:
        img = Image.open(str(path)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {suffix}")

    return img


def preprocess_xray(img: Image.Image, size: int = 224) -> Image.Image:
    """CLAHE contrast enhancement + resize for chest X-ray."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(arr)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    return pil


def get_image_metadata(img: Image.Image) -> dict:
    """Return basic metadata dict for logging."""
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "shape": (img.height, img.width, len(img.getbands())),
    }


def draw_bounding_boxes(
    img: Image.Image,
    boxes: list[dict],  # [{"label": str, "box": [x1,y1,x2,y2], "score": float}]
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    """Overlay bounding boxes onto a PIL image for Streamlit display."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for item in boxes:
        box = item.get("box", [])
        label = item.get("label", "")
        score = item.get("score", 1.0)
        if len(box) == 4:
            draw.rectangle(box, outline=color, width=width)
            draw.text((box[0], box[1] - 12), f"{label} {score:.2f}", fill=color)
    return out


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize PIL image to bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
