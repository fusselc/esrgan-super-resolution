"""
Utility functions for ESRGAN.

- PSNR computation
- SSIM computation
- Image load / save helpers
- Tensor ↔ NumPy conversions
"""

from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy conversion
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) or (B, C, H, W) float tensor in [0, 1]
    to a uint8 NumPy array in HWC or BHWC format.
    """
    if tensor.ndim == 4:
        # (B, C, H, W) → (B, H, W, C)
        arr = tensor.detach().cpu().float().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
        return (arr * 255.0).round().astype(np.uint8)
    elif tensor.ndim == 3:
        # (C, H, W) → (H, W, C)
        arr = tensor.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return (arr * 255.0).round().astype(np.uint8)
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {tensor.ndim}-D")


def numpy_to_tensor(arr: np.ndarray, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Convert a HWC uint8 NumPy array (values in [0, 255]) to a
    (C, H, W) float32 tensor in [0, 1].
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Load an image from disk and return as a (C, H, W) float32 tensor in [0, 1].

    Uses PIL to load, converts to RGB.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a (C, H, W) or (B, C, H, W) float tensor in [0, 1] to disk.

    If batched, only the first image is saved.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor_to_numpy(tensor)
    img = Image.fromarray(arr)
    img.save(path)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred:    Predicted image tensor (float, [0, max_val]) or ndarray.
        target:  Ground-truth image tensor or ndarray.
        max_val: Maximum value of the image (default: 1.0).
    Returns:
        PSNR value in dB (float). Returns inf if MSE == 0.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float().numpy()

    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    window_size: int = 11,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Works on single images (C, H, W) tensors or (H, W, C) / (H, W) ndarrays.
    Computes SSIM per channel and averages.

    Args:
        pred:        Predicted image.
        target:      Ground-truth image.
        max_val:     Dynamic range of the image (default: 1.0).
        window_size: Size of the Gaussian window (default: 11).
    Returns:
        Mean SSIM (float).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float().numpy()
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)  # CHW → HWC
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float().numpy()
        if target.ndim == 3:
            target = target.transpose(1, 2, 0)

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    if pred.ndim == 2:
        # Grayscale
        return _ssim_channel(pred, target, max_val, window_size)

    # Multi-channel: average over channels
    ssim_vals = [
        _ssim_channel(pred[..., c], target[..., c], max_val, window_size)
        for c in range(pred.shape[-1])
    ]
    return float(np.mean(ssim_vals))


def _gaussian_kernel(size: int, sigma: float = 1.5) -> np.ndarray:
    """Create a 1D Gaussian kernel."""
    coords = np.arange(size) - size // 2
    kernel = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _ssim_channel(
    pred: np.ndarray,
    target: np.ndarray,
    max_val: float,
    window_size: int,
) -> float:
    """Compute SSIM for a single 2-D channel."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    kernel_1d = _gaussian_kernel(window_size)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    def _conv(img: np.ndarray) -> np.ndarray:
        return cv2.filter2D(img, -1, kernel_2d, borderType=cv2.BORDER_REFLECT)

    mu1 = _conv(pred)
    mu2 = _conv(target)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _conv(pred * pred) - mu1_sq
    sigma2_sq = _conv(target * target) - mu2_sq
    sigma12 = _conv(pred * target) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(ssim_map.mean())
