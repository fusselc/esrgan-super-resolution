"""
RRDB (Residual in Residual Dense Block) Generator Network for ESRGAN.

Architecture:
- 23 RRDB blocks
- 64 base channels (nf)
- 32 growth channels (gc)
- No BatchNorm anywhere
- LeakyReLU (negative_slope=0.2)
- Kaiming Normal initialization
- Nearest-neighbor + conv upsampling (NO PixelShuffle)
- Residual scaling = 0.2 in all RRDB blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kaiming_init(module: nn.Module) -> None:
    """Apply Kaiming Normal initialization to Conv2d layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DenseLayer(nn.Module):
    """Single dense layer: Conv + LeakyReLU."""

    def __init__(self, in_channels: int, growth_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with 5 dense layers and residual scaling.

    Each dense layer concatenates all previous outputs.
    Final layer maps to nf channels (no activation).
    Residual scaling = 0.2.
    """

    RESIDUAL_SCALE = 0.2

    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = DenseLayer(nf, gc)
        self.conv2 = DenseLayer(nf + gc, gc)
        self.conv3 = DenseLayer(nf + 2 * gc, gc)
        self.conv4 = DenseLayer(nf + 3 * gc, gc)
        # Last layer: no activation, output nf channels
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, kernel_size=3, padding=1, bias=True)
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.RESIDUAL_SCALE + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).

    Contains 3 Residual Dense Blocks with residual scaling = 0.2.
    """

    RESIDUAL_SCALE = 0.2

    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.RESIDUAL_SCALE + x


class RRDBNet(nn.Module):
    """
    RRDB Generator Network (ESRGAN).

    Architecture:
      1. Initial feature extraction conv
      2. N RRDB blocks (default: 23)
      3. Trunk convolution
      4. Upsampling: nearest-neighbor interpolation + Conv (x2 twice = x4 total)
      5. HRconv + output conv

    No BatchNorm. LeakyReLU(0.2) throughout.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale

        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1, bias=True)

        # RRDB trunk
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])

        # Trunk convolution
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=True)

        # Upsampling layers (nearest-neighbor + conv, no PixelShuffle)
        # Each block upsamples by 2x
        assert scale in (2, 4), f"Scale must be 2 or 4, got {scale}"
        self.upconv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if scale == 4:
            self.upconv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=True)
            self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # High-resolution conv
        self.HRconv = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=True)
        self.lrelu_hr = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Output conv (no activation)
        self.conv_last = nn.Conv2d(nf, out_nc, kernel_size=3, padding=1, bias=True)

        # Initialize weights
        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # Upsample x2
        fea = self.lrelu1(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )

        # Second upsample x2 (total x4)
        if self.scale == 4:
            fea = self.lrelu2(
                self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )

        out = self.conv_last(self.lrelu_hr(self.HRconv(fea)))
        return out
