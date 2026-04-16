"""
Loss functions for ESRGAN training.

Includes:
- L1Loss: pixel-wise L1 loss
- PerceptualLoss: VGG19-based feature loss (conv5_4, pre-activation)
- GANLoss: Relativistic average GAN (RaGAN) loss using BCEWithLogitsLoss
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1Loss(nn.Module):
    """Simple L1 (mean absolute error) loss."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class PerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss.

    Extracts features from conv5_4 (index 35 in VGG19 features)
    BEFORE the ReLU activation, as specified in ESRGAN.

    Inputs are normalized with ImageNet mean/std and clipped to [0, 1].
    All VGG parameters are frozen.
    """

    # ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # conv5_4 is at index 35 in vgg19.features (before the activation at 36)
    VGG_LAYER_IDX = 35

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract up to conv5_4 (inclusive), before its activation
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[: self.VGG_LAYER_IDX + 1])
        # Freeze all parameters
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        self.vgg_features.eval()

        self.loss_fn = nn.L1Loss()

        # Register normalization buffers
        mean = torch.tensor(self.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp to [0, 1] then normalize with ImageNet stats."""
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) predicted SR image in [0, 1]
            target: (B, 3, H, W) HR ground truth in [0, 1]
        Returns:
            scalar perceptual loss
        """
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        with torch.no_grad():
            feat_target = self.vgg_features(target_norm)
        feat_pred = self.vgg_features(pred_norm)
        return self.loss_fn(feat_pred, feat_target.detach())


class GANLoss(nn.Module):
    """
    Relativistic average GAN (RaGAN) loss using BCEWithLogitsLoss.

    For the generator:
        loss_G = BCE(D(fake) - mean(D(real)), 1) + BCE(D(real) - mean(D(fake)), 0)

    For the discriminator:
        loss_D = BCE(D(real) - mean(D(fake)), 1) + BCE(D(fake) - mean(D(real)), 0)
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
        for_discriminator: bool,
    ) -> torch.Tensor:
        """
        Args:
            pred_real: discriminator logits for real (HR) images
            pred_fake: discriminator logits for fake (SR) images
            for_discriminator: True when computing discriminator loss
        Returns:
            scalar RaGAN loss
        """
        mean_real = pred_real.mean(dim=0, keepdim=True)
        mean_fake = pred_fake.mean(dim=0, keepdim=True)

        real_diff = pred_real - mean_fake
        fake_diff = pred_fake - mean_real

        ones = torch.ones_like(real_diff)
        zeros = torch.zeros_like(fake_diff)

        if for_discriminator:
            # Discriminator wants real > fake
            loss = self.bce(real_diff, ones) + self.bce(fake_diff, zeros)
        else:
            # Generator wants fake > real
            loss = self.bce(fake_diff, ones) + self.bce(real_diff, zeros)

        return loss
