"""
Tests for ESRGAN inference pipeline.

- Test single image produces correct 4x output size
- Test tiled inference produces same shape as full inference
- Test model loading and inference API
"""

import os
import tempfile
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rrdb_net import RRDBNet
from src.inference import tiled_upscale, run_inference, build_model


def make_small_model(scale: int = 4) -> RRDBNet:
    """Create a lightweight model for testing."""
    model = RRDBNet(in_nc=3, out_nc=3, nf=16, nb=2, gc=8, scale=scale)
    model.eval()
    return model


def make_cfg(scale: int = 4, tile_size: int = 32, tile_overlap: int = 8) -> dict:
    """Create a minimal inference config dict."""
    return {
        "model": {
            "in_nc": 3,
            "out_nc": 3,
            "nf": 16,
            "nb": 2,
            "gc": 8,
            "scale": scale,
            "checkpoint": "",
        },
        "io": {"input": "", "output": "results"},
        "tiling": {
            "enabled": True,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
        },
    }


class TestInferenceOutputSize:
    """Test that inference produces correctly upscaled output."""

    def test_4x_output_size(self):
        """Single image: 4x upscale gives correct H and W."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 32, 32)
        cfg = make_cfg(scale=4)

        sr = run_inference(model, lr, cfg, device)
        assert sr.shape == (3, 128, 128), f"Expected (3, 128, 128), got {sr.shape}"

    def test_4x_output_non_square(self):
        """Non-square input: 4x upscale gives correct dimensions."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 16, 24)
        cfg = make_cfg(scale=4)

        sr = run_inference(model, lr, cfg, device)
        assert sr.shape == (3, 64, 96), f"Expected (3, 64, 96), got {sr.shape}"

    def test_output_range(self):
        """Output values must be in [0, 1]."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 16, 16)
        cfg = make_cfg(scale=4)

        sr = run_inference(model, lr, cfg, device)
        assert sr.min().item() >= 0.0, f"Output min {sr.min().item()} < 0"
        assert sr.max().item() <= 1.0, f"Output max {sr.max().item()} > 1"

    def test_no_tiling_4x(self):
        """Without tiling, single pass should also produce 4x output."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 16, 16)
        cfg = make_cfg(scale=4)
        cfg["tiling"]["enabled"] = False

        with torch.no_grad():
            sr = run_inference(model, lr, cfg, device)
        assert sr.shape == (3, 64, 64)


class TestTiledInference:
    """Tests for the tiled inference function."""

    def test_tiled_output_shape(self):
        """Tiled inference should produce the same shape as direct inference."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 32, 32)

        sr_tiled = tiled_upscale(model, lr, scale=4, tile_size=16, tile_overlap=4, device=device)
        assert sr_tiled.shape == (3, 128, 128), f"Expected (3, 128, 128), got {sr_tiled.shape}"

    def test_tiled_output_range(self):
        """Tiled output should be in [0, 1]."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 32, 32)

        sr = tiled_upscale(model, lr, scale=4, tile_size=16, tile_overlap=4, device=device)
        assert sr.min().item() >= 0.0
        assert sr.max().item() <= 1.0

    def test_tiled_image_larger_than_tile(self):
        """Image larger than tile_size is handled correctly."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 48, 48)

        sr = tiled_upscale(model, lr, scale=4, tile_size=16, tile_overlap=4, device=device)
        assert sr.shape == (3, 192, 192)

    def test_tiled_small_image(self):
        """Image smaller than tile_size: treated as single tile."""
        model = make_small_model(scale=4)
        device = torch.device("cpu")
        lr = torch.rand(3, 8, 8)

        sr = tiled_upscale(model, lr, scale=4, tile_size=32, tile_overlap=8, device=device)
        assert sr.shape == (3, 32, 32)


class TestInferenceWithSavedImage:
    """Test inference reading from and writing to disk."""

    def test_process_image_saves_file(self):
        """run_inference + save produces an output file."""
        from src.inference import process_image

        model = make_small_model(scale=4)
        device = torch.device("cpu")
        cfg = make_cfg(scale=4)
        # Patch model into cfg is not needed since we pass the model directly

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small input image
            input_path = os.path.join(tmpdir, "input.png")
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(input_path)

            output_path = os.path.join(tmpdir, "output.png")
            process_image(model, input_path, output_path, cfg, device)

            assert os.path.exists(output_path), "Output file was not created"
            result = Image.open(output_path)
            assert result.size == (128, 128), (
                f"Expected 128x128, got {result.size}"
            )
