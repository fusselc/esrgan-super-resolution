"""
ESRGAN Inference Module.

Supports:
- Single image upscaling
- Directory batch processing
- Tiled inference with configurable tile size and overlap
- Seamless stitching via overlap blending
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml

from .rrdb_net import RRDBNet
from .utils import load_image, save_image, tensor_to_numpy

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def upscale_tile(
    model: torch.nn.Module,
    lr_patch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the generator on a single LR patch.

    Args:
        model:    RRDBNet generator.
        lr_patch: (1, C, H, W) float32 tensor in [0, 1].
        device:   Target device.
    Returns:
        (1, C, H*scale, W*scale) SR tensor in [0, 1].
    """
    lr_patch = lr_patch.to(device)
    with torch.no_grad():
        sr_patch = model(lr_patch).clamp(0.0, 1.0)
    return sr_patch


def tiled_upscale(
    model: torch.nn.Module,
    lr_img: torch.Tensor,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run super-resolution on a large image using overlapping tiles.

    Tiles are blended using a linear weight ramp in the overlap region
    to avoid visible seams.

    Args:
        model:        RRDBNet generator.
        lr_img:       (C, H, W) float32 LR input tensor.
        scale:        Upscaling factor.
        tile_size:    LR tile size in pixels.
        tile_overlap: Overlap in pixels (in LR space) between adjacent tiles.
        device:       Target device.
    Returns:
        (C, H*scale, W*scale) SR output tensor in [0, 1].
    """
    c, h, w = lr_img.shape
    sr_h, sr_w = h * scale, w * scale

    output = torch.zeros(c, sr_h, sr_w, dtype=torch.float32)
    weight_map = torch.zeros(1, sr_h, sr_w, dtype=torch.float32)

    stride = tile_size - tile_overlap
    if tile_overlap >= tile_size:
        raise ValueError(
            f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})"
        )
    if stride <= 0:
        stride = 1

    # Generate tile positions
    y_starts = list(range(0, h - tile_size, stride))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))

    x_starts = list(range(0, w - tile_size, stride))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    # Ensure we cover cases where image is smaller than tile_size
    if h <= tile_size:
        y_starts = [0]
    if w <= tile_size:
        x_starts = [0]

    for y in y_starts:
        y_end = min(y + tile_size, h)
        for x in x_starts:
            x_end = min(x + tile_size, w)
            tile = lr_img[:, y:y_end, x:x_end].unsqueeze(0)
            sr_tile = upscale_tile(model, tile, device).squeeze(0).cpu()

            # Corresponding SR coordinates
            sy, sx = y * scale, x * scale
            sey, sex = y_end * scale, x_end * scale

            sr_th, sr_tw = sr_tile.shape[1], sr_tile.shape[2]

            # Build a linear blend weight (tapering near tile edges)
            blend_w = _build_blend_weight(sr_th, sr_tw, tile_overlap * scale)

            output[:, sy:sey, sx:sex] += sr_tile * blend_w
            weight_map[:, sy:sey, sx:sex] += blend_w

    # Normalize by accumulated weights
    output = output / weight_map.clamp(min=1e-8)
    return output.clamp(0.0, 1.0)


def _build_blend_weight(h: int, w: int, overlap_sr: int) -> torch.Tensor:
    """
    Build a (1, H, W) blend weight tensor that linearly ramps from 0 to 1
    in the overlap region at the borders, and is 1.0 in the interior.
    """
    weight = torch.ones(1, h, w, dtype=torch.float32)
    ramp = min(overlap_sr, h // 2, w // 2)
    if ramp > 0:
        for i in range(ramp):
            v = (i + 1) / (ramp + 1)
            weight[:, i, :] *= v
            weight[:, -(i + 1), :] *= v
            weight[:, :, i] *= v
            weight[:, :, -(i + 1)] *= v
    return weight


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> RRDBNet:
    """Build and load the generator model from a config dict."""
    model_cfg = cfg["model"]
    model = RRDBNet(
        in_nc=model_cfg["in_nc"],
        out_nc=model_cfg["out_nc"],
        nf=model_cfg["nf"],
        nb=model_cfg["nb"],
        gc=model_cfg["gc"],
        scale=model_cfg["scale"],
    ).to(device)
    model.eval()

    checkpoint_path = model_cfg.get("checkpoint", "")
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("generator", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded generator weights from: {checkpoint_path}")
    else:
        print("No checkpoint provided; using random weights.")

    return model


def run_inference(
    model: RRDBNet,
    lr_img: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Run inference on a single LR image tensor.

    Args:
        model:  RRDBNet generator in eval mode.
        lr_img: (C, H, W) float32 tensor.
        cfg:    Full inference config dict.
        device: Target device.
    Returns:
        (C, H*scale, W*scale) SR tensor.
    """
    tiling_cfg = cfg.get("tiling", {})
    use_tiling = tiling_cfg.get("enabled", True)
    tile_size = tiling_cfg.get("tile_size", 128)
    tile_overlap = tiling_cfg.get("tile_overlap", 16)
    scale = cfg["model"]["scale"]

    if use_tiling:
        sr = tiled_upscale(model, lr_img, scale, tile_size, tile_overlap, device)
    else:
        with torch.no_grad():
            sr = model(lr_img.unsqueeze(0).to(device)).clamp(0.0, 1.0).squeeze(0).cpu()

    return sr


def process_image(
    model: RRDBNet,
    input_path: str,
    output_path: str,
    cfg: dict,
    device: torch.device,
) -> None:
    """Upscale a single image and save to output_path."""
    lr_img = load_image(input_path)
    sr_img = run_inference(model, lr_img, cfg, device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_image(sr_img, output_path)
    print(f"  {input_path} → {output_path}")


def process_directory(
    model: RRDBNet,
    input_dir: str,
    output_dir: str,
    cfg: dict,
    device: torch.device,
) -> None:
    """Upscale all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in sorted(image_paths):
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_image(model, str(path), str(out_path), cfg, device)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESRGAN Inference")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Path to inference config YAML file")
    parser.add_argument("--input", type=str, default=None,
                        help="Input image or directory (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to generator checkpoint (overrides config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.input:
        cfg["io"]["input"] = args.input
    if args.output:
        cfg["io"]["output"] = args.output
    if args.checkpoint:
        cfg["model"]["checkpoint"] = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(cfg, device)

    input_path = cfg["io"]["input"]
    output_dir = cfg["io"]["output"]

    if not input_path:
        print("No input specified. Set 'io.input' in the config or pass --input.")
        return

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        process_directory(model, input_path, output_dir, cfg, device)
    elif os.path.isfile(input_path):
        out_name = Path(input_path).stem + "_sr.png"
        out_path = os.path.join(output_dir, out_name)
        print(f"Processing single image: {input_path}")
        process_image(model, input_path, out_path, cfg, device)
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
