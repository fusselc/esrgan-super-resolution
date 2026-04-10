"""
Tests for the VGG-style Discriminator.

- Test scalar output per image
- Test no BatchNorm in architecture
- Test runs on CPU
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.discriminator import Discriminator


class TestDiscriminator:
    """Tests for the VGG-style Discriminator."""

    @pytest.fixture
    def disc(self):
        """Standard discriminator instance."""
        return Discriminator(in_nc=3, ndf=64)

    def test_output_shape(self, disc):
        """Discriminator must output (B, 1) scalar logits."""
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            out = disc(x)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_output_shape_single(self, disc):
        """Works with batch size 1."""
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = disc(x)
        assert out.shape == (1, 1)

    def test_output_is_scalar_per_image(self, disc):
        """Each image produces exactly one output value."""
        batch_size = 3
        x = torch.randn(batch_size, 3, 128, 128)
        with torch.no_grad():
            out = disc(x)
        assert out.numel() == batch_size

    def test_no_batchnorm(self, disc):
        """Discriminator must not contain any BatchNorm layers."""
        for name, module in disc.named_modules():
            assert not isinstance(module, torch.nn.BatchNorm2d), (
                f"Found BatchNorm2d at: {name}"
            )

    def test_output_is_logit(self, disc):
        """Output should be raw logits (not passed through sigmoid)."""
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = disc(x)
        # Raw logits can be any real number
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_spatial_sizes(self, disc):
        """AdaptiveAvgPool allows different spatial sizes."""
        for h, w in [(64, 64), (128, 128), (256, 256)]:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                out = disc(x)
            assert out.shape == (1, 1), f"Failed for size ({h}, {w})"

    def test_cpu_forward(self, disc):
        """Discriminator runs correctly on CPU."""
        disc.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = disc(x)
        assert out.shape == (2, 1)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, disc):
        """Gradients should flow through the discriminator."""
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        out = disc(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
