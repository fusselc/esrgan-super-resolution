"""
Tests for ESRGAN loss functions.

- L1Loss: valid scalar output, zero for identical inputs
- PerceptualLoss: valid scalar output, positive, VGG frozen
- GANLoss: valid RaGAN loss for discriminator and generator
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.losses import L1Loss, PerceptualLoss, GANLoss


class TestL1Loss:
    """Tests for L1Loss."""

    def test_output_is_scalar(self):
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        loss_fn = L1Loss()
        loss = loss_fn(pred, target)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_zero_for_identical_inputs(self):
        x = torch.randn(2, 3, 32, 32)
        loss_fn = L1Loss()
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_different_inputs(self):
        pred = torch.zeros(2, 3, 32, 32)
        target = torch.ones(2, 3, 32, 32)
        loss_fn = L1Loss()
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0

    def test_symmetric(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss_fn = L1Loss()
        assert loss_fn(pred, target).item() == pytest.approx(
            loss_fn(target, pred).item(), rel=1e-5
        )

    def test_differentiable(self):
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        loss_fn = L1Loss()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestPerceptualLoss:
    """Tests for VGG19-based PerceptualLoss."""

    @pytest.fixture
    def perc_loss(self):
        return PerceptualLoss()

    def test_output_is_scalar(self, perc_loss):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = perc_loss(pred, target)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_non_negative(self, perc_loss):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = perc_loss(pred, target)
        assert loss.item() >= 0.0

    def test_zero_or_near_zero_for_identical_inputs(self, perc_loss):
        x = torch.rand(1, 3, 64, 64)
        loss = perc_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_vgg_parameters_frozen(self, perc_loss):
        """All VGG parameters must have requires_grad=False."""
        for name, param in perc_loss.vgg_features.named_parameters():
            assert not param.requires_grad, (
                f"VGG param {name} should be frozen but requires_grad=True"
            )

    def test_accepts_batch(self, perc_loss):
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        loss = perc_loss(pred, target)
        assert loss.shape == ()

    def test_differentiable_through_pred(self, perc_loss):
        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)
        loss = perc_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestGANLoss:
    """Tests for Relativistic average GAN (RaGAN) loss."""

    @pytest.fixture
    def gan_loss(self):
        return GANLoss()

    def test_discriminator_loss_scalar(self, gan_loss):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        loss = gan_loss(real, fake, for_discriminator=True)
        assert loss.shape == ()

    def test_generator_loss_scalar(self, gan_loss):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        loss = gan_loss(real, fake, for_discriminator=False)
        assert loss.shape == ()

    def test_discriminator_loss_non_negative(self, gan_loss):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        loss = gan_loss(real, fake, for_discriminator=True)
        assert loss.item() >= 0.0

    def test_generator_loss_non_negative(self, gan_loss):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        loss = gan_loss(real, fake, for_discriminator=False)
        assert loss.item() >= 0.0

    def test_optimal_discriminator(self, gan_loss):
        """When D perfectly separates: real >> fake, D loss should be near 0."""
        real = torch.full((8, 1), 10.0)   # very large logits
        fake = torch.full((8, 1), -10.0)  # very small logits
        loss_d = gan_loss(real, fake, for_discriminator=True)
        assert loss_d.item() < 0.1

    def test_differentiable(self, gan_loss):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1, requires_grad=True)
        loss = gan_loss(real, fake, for_discriminator=False)
        loss.backward()
        assert fake.grad is not None
