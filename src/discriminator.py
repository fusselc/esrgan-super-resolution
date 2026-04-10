"""
VGG-style Discriminator for ESRGAN (Relativistic average GAN).

Architecture:
- VGG-style feature extractor
- No BatchNorm (uses spectral norm on conv layers optionally)
- LeakyReLU (0.2) throughout
- Outputs a scalar (single value per image)
- Standard PyTorch initialization
"""

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Sequential:
    """Convolution block: Conv + LeakyReLU (no BN)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class Discriminator(nn.Module):
    """
    VGG-style Discriminator for ESRGAN.

    Input: (B, 3, H, W) — expects 128x128 HR patches.
    Output: (B, 1) scalar logits (pre-sigmoid, for BCEWithLogitsLoss).
    """

    def __init__(self, in_nc: int = 3, ndf: int = 64) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(in_nc, ndf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            _conv_block(ndf, ndf, stride=2),         # 64x64
            # Block 2: 64 -> 32
            _conv_block(ndf, ndf * 2, stride=1),
            _conv_block(ndf * 2, ndf * 2, stride=2), # 32x32
            # Block 3: 32 -> 16
            _conv_block(ndf * 2, ndf * 4, stride=1),
            _conv_block(ndf * 4, ndf * 4, stride=2), # 16x16
            # Block 4: 16 -> 8
            _conv_block(ndf * 4, ndf * 8, stride=1),
            _conv_block(ndf * 8, ndf * 8, stride=2), # 8x8
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),          # (B, ndf*8, 4, 4)
            nn.Flatten(),                     # (B, ndf*8*4*4)
            nn.Linear(ndf * 8 * 4 * 4, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1),               # scalar logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor in [0, 1]
        Returns:
            (B, 1) logits
        """
        feat = self.features(x)
        return self.classifier(feat)
