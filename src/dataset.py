"""
Dataset for ESRGAN training.

- Reads HR images from a directory.
- Generates LR images on-the-fly via bicubic downsampling.
- HR crop: 128×128, LR crop: 32×32 (4× scale factor).
- Augmentations: random horizontal flip, random rotation (0°, 90°, 180°, 270°).
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(directory: str) -> List[Path]:
    """Recursively collect image paths from a directory."""
    root = Path(directory)
    paths = [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def _random_crop(hr_img: torch.Tensor, hr_size: int, scale: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly crop HR patch and derive corresponding LR patch.

    Args:
        hr_img:  (C, H, W) HR image tensor in [0, 1]
        hr_size: size of the HR crop (e.g. 128)
        scale:   downscaling factor (e.g. 4)
    Returns:
        (hr_patch, lr_patch) tensors
    """
    _, h, w = hr_img.shape
    # Ensure image is large enough
    if h < hr_size or w < hr_size:
        hr_img = F.interpolate(
            hr_img.unsqueeze(0),
            size=(max(h, hr_size), max(w, hr_size)),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        _, h, w = hr_img.shape

    top = random.randint(0, h - hr_size)
    left = random.randint(0, w - hr_size)

    hr_patch = hr_img[:, top : top + hr_size, left : left + hr_size]

    # Derive LR via bicubic downsampling
    lr_size = hr_size // scale
    lr_patch = F.interpolate(
        hr_patch.unsqueeze(0),
        size=(lr_size, lr_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)

    return hr_patch, lr_patch


def _augment(hr_patch: torch.Tensor, lr_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random augmentations:
      - Random horizontal flip
      - Random rotation (0°, 90°, 180°, 270°)
    """
    # Horizontal flip
    if random.random() > 0.5:
        hr_patch = torch.flip(hr_patch, dims=[-1])
        lr_patch = torch.flip(lr_patch, dims=[-1])

    # Rotation (k in {0,1,2,3} means k*90°)
    k = random.randint(0, 3)
    if k > 0:
        hr_patch = torch.rot90(hr_patch, k=k, dims=[-2, -1])
        lr_patch = torch.rot90(lr_patch, k=k, dims=[-2, -1])

    return hr_patch, lr_patch


class SRDataset(Dataset):
    """
    Super-Resolution dataset.

    Reads HR images, generates LR on-the-fly via bicubic downsampling.
    Applies random crop and augmentation during training.

    Args:
        hr_dir:      Path to HR images directory.
        hr_crop_size: Size of the HR crop (default: 128).
        scale:       Downscaling factor (default: 4).
        augment:     Whether to apply data augmentation (default: True).
    """

    def __init__(
        self,
        hr_dir: str,
        hr_crop_size: int = 128,
        scale: int = 4,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.hr_paths = _collect_images(hr_dir)
        if not self.hr_paths:
            raise ValueError(f"No images found in directory: {hr_dir}")
        self.hr_crop_size = hr_crop_size
        self.scale = scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.hr_paths[idx]
        img = Image.open(path).convert("RGB")
        # Convert to tensor [0, 1]
        hr_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        hr_patch, lr_patch = _random_crop(hr_tensor, self.hr_crop_size, self.scale)

        if self.augment:
            hr_patch, lr_patch = _augment(hr_patch, lr_patch)

        return lr_patch, hr_patch
