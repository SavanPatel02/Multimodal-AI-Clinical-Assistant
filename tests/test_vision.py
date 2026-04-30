"""Tests for vision encoding pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.vision.image_utils import (
    draw_bounding_boxes,
    get_image_metadata,
    image_to_bytes,
    load_image,
    preprocess_xray,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rgb_image(tmp_path) -> Path:
    """512×512 synthetic grayscale-like chest X-ray image."""
    arr = np.random.randint(0, 200, (512, 512, 3), dtype=np.uint8)
    path = tmp_path / "test_xray.jpg"
    Image.fromarray(arr).save(str(path))
    return path


# ── image_utils tests ─────────────────────────────────────────────────────────

class TestLoadImage:
    def test_loads_rgb(self, rgb_image):
        img = load_image(rgb_image)
        assert img.mode == "RGB"
        assert img.size == (512, 512)

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "scan.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(bad)


class TestPreprocessXray:
    def test_output_size(self, rgb_image):
        img = load_image(rgb_image)
        out = preprocess_xray(img, size=224)
        assert out.size == (224, 224)

    def test_output_rgb(self, rgb_image):
        img = load_image(rgb_image)
        out = preprocess_xray(img)
        assert out.mode == "RGB"


class TestGetImageMetadata:
    def test_metadata_keys(self, rgb_image):
        img = load_image(rgb_image)
        meta = get_image_metadata(img)
        assert "width" in meta
        assert "height" in meta
        assert "mode" in meta
        assert meta["width"] == 512
        assert meta["height"] == 512


class TestDrawBoundingBoxes:
    def test_draws_without_error(self, rgb_image):
        img = load_image(rgb_image)
        boxes = [{"label": "Pneumonia", "box": [50, 50, 200, 200], "score": 0.87}]
        result = draw_bounding_boxes(img, boxes)
        assert isinstance(result, Image.Image)

    def test_empty_boxes_returns_copy(self, rgb_image):
        img = load_image(rgb_image)
        result = draw_bounding_boxes(img, [])
        assert result.size == img.size


class TestImageToBytes:
    def test_returns_bytes(self, rgb_image):
        img = load_image(rgb_image)
        data = image_to_bytes(img)
        assert isinstance(data, bytes)
        assert len(data) > 0


# ── image_encoder tests ───────────────────────────────────────────────────────

class TestMedicalImageEncoder:
    def test_encode_output_structure(self, rgb_image):
        from src.vision.image_encoder import MedicalImageEncoder
        encoder = MedicalImageEncoder(device="cpu")
        encoder.load()
        result = encoder.encode(rgb_image)

        assert "feature_vector" in result
        assert "pathology_scores" in result
        assert "top_findings" in result
        assert "image_metadata" in result
        assert "gradcam_overlay" in result
        assert "latency_s" in result
        assert len(result["feature_vector"]) == 1024
        assert len(result["pathology_scores"]) == 14
        assert isinstance(result["top_findings"], list)
