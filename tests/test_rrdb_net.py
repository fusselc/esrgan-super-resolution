"""
Tests for RRDBNet generator.

- Test forward pass shape (LR input → 4x SR output)
- Test no BatchNorm in architecture
- Test model runs on CPU
"""

import torch
import pytest
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rrdb_net import RRDBNet, RRDB, ResidualDenseBlock


class TestRRDBNet:
    """Tests for the RRDB generator network."""

    @pytest.fixture
    def small_model(self):
        """Lightweight model for fast tests (nb=4 instead of 23)."""
        return RRDBNet(in_nc=3, out_nc=3, nf=32, nb=4, gc=16, scale=4)

    def test_output_shape_scale4(self, small_model):
        """Generator must output 4x the spatial resolution of the input."""
        lr = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            sr = small_model(lr)
        assert sr.shape == (1, 3, 128, 128), (
            f"Expected (1, 3, 128, 128), got {sr.shape}"
        )

    def test_output_shape_batch(self, small_model):
        """Works with batch size > 1."""
        lr = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            sr = small_model(lr)
        assert sr.shape == (2, 3, 128, 128)

    def test_output_shape_different_spatial(self, small_model):
        """Works with different input spatial sizes."""
        lr = torch.randn(1, 3, 16, 24)
        with torch.no_grad():
            sr = small_model(lr)
        assert sr.shape == (1, 3, 64, 96)

    def test_no_batchnorm(self, small_model):
        """Architecture must not contain any BatchNorm layers."""
        for name, module in small_model.named_modules():
            assert not isinstance(module, torch.nn.BatchNorm2d), (
                f"Found BatchNorm2d at: {name}"
            )

    def test_output_channel_count(self, small_model):
        """Output must have 3 channels (RGB)."""
        lr = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            sr = small_model(lr)
        assert sr.shape[1] == 3

    def test_full_model_shape(self):
        """Full 23-block model with standard channels (slow on CPU, but checks shape)."""
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        lr = torch.randn(1, 3, 8, 8)  # Tiny input to keep it fast
        with torch.no_grad():
            sr = model(lr)
        assert sr.shape == (1, 3, 32, 32)

    def test_scale2(self):
        """Model with scale=2 produces 2x output."""
        model = RRDBNet(in_nc=3, out_nc=3, nf=16, nb=2, gc=8, scale=2)
        lr = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            sr = model(lr)
        assert sr.shape == (1, 3, 64, 64)

    def test_residual_dense_block_shape(self):
        """ResidualDenseBlock preserves spatial shape."""
        rdb = ResidualDenseBlock(nf=32, gc=16)
        x = torch.randn(1, 32, 16, 16)
        out = rdb(x)
        assert out.shape == x.shape

    def test_rrdb_shape(self):
        """RRDB block preserves shape."""
        rrdb = RRDB(nf=32, gc=16)
        x = torch.randn(1, 32, 16, 16)
        out = rrdb(x)
        assert out.shape == x.shape

    def test_cpu_inference(self, small_model):
        """Model runs correctly on CPU."""
        small_model.eval()
        lr = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            sr = small_model(lr)
        assert sr.shape == (1, 3, 64, 64)
        assert not torch.isnan(sr).any(), "Output contains NaN values"
        assert not torch.isinf(sr).any(), "Output contains Inf values"
