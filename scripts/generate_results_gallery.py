"""
Results Gallery Generator for ESRGAN.

Loads the demo checkpoint, runs 4× tiled inference on sample images,
builds side-by-side Before/After composites, computes quality metrics
(PSNR, SSIM, LPIPS), and emits a paste-ready Markdown table + image gallery.

Usage:
    python scripts/generate_results_gallery.py [--checkpoint checkpoints/esrgan_demo_4x.pth]
    python -m scripts.generate_results_gallery

Outputs:
    results/gallery/<name>_comparison.png  — side-by-side composite images
    results/gallery_readme.md              — Markdown table + gallery block
"""

import argparse
import os
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running as `python scripts/generate_results_gallery.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.inference import build_model, run_inference
from src.utils import compute_psnr, compute_ssim, load_image, tensor_to_numpy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_CHECKPOINT = "checkpoints/esrgan_demo_4x.pth"
INFERENCE_CONFIG = "configs/inference_config.yaml"
GALLERY_DIR = "results/gallery"
GALLERY_MD_PATH = "results/gallery_readme.md"

# Public-domain test images to download when data/sample/ is empty
FALLBACK_IMAGES = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/320px-Bikesgray.jpg",
        "bikes.jpg",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        "ant.jpg",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "transparency_demo.png",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/320px-Collage_of_Nine_Dogs.jpg",
        "dogs.jpg",
    ),
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# LPIPS (optional)
# ---------------------------------------------------------------------------

_LPIPS_MODEL = None


def _try_load_lpips(device: torch.device):
    """Attempt to load an LPIPS model; return None if unavailable."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # noqa: F401

        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device)
        _LPIPS_MODEL.eval()
        print("LPIPS model loaded (AlexNet backbone).")
        return _LPIPS_MODEL
    except ImportError:
        print("lpips package not installed — LPIPS column will be N/A.")
        print("  Install with: pip install lpips")
        return None


def _compute_lpips(
    lpips_model,
    sr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    device: torch.device,
) -> Optional[float]:
    """Return LPIPS score or None if model unavailable."""
    if lpips_model is None:
        return None
    with torch.no_grad():
        # LPIPS expects (B, C, H, W) in [-1, 1]
        sr_b = sr_tensor.unsqueeze(0).to(device) * 2.0 - 1.0
        hr_b = hr_tensor.unsqueeze(0).to(device) * 2.0 - 1.0
        score = lpips_model(sr_b, hr_b).item()
    return score


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _collect_sample_images(data_dir: str, max_images: int = 6) -> List[Path]:
    """Return up to max_images image paths from data_dir."""
    root = Path(data_dir)
    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths[:max_images]


def _download_fallback_images(dest_dir: str) -> List[Path]:
    """Download a small set of public-domain test images."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for url, fname in FALLBACK_IMAGES:
        out_path = dest / fname
        if not out_path.exists():
            print(f"Downloading fallback image: {fname} …")
            try:
                urllib.request.urlretrieve(url, str(out_path))
                downloaded.append(out_path)
                print(f"  ✓ {out_path}")
            except Exception as exc:
                print(f"  ✗ Failed ({exc})")
        else:
            downloaded.append(out_path)
    return downloaded


def _ensure_images(data_dir: str, max_images: int = 6) -> List[Path]:
    """Return available images, downloading fallbacks if needed."""
    paths = _collect_sample_images(data_dir, max_images)
    if not paths:
        print(f"No images found in '{data_dir}'. Downloading fallback test images …")
        fallback_dir = str(_REPO_ROOT / "data" / "sample_fallback")
        paths = _download_fallback_images(fallback_dir)
    if not paths:
        raise RuntimeError(
            "Could not obtain any test images. "
            "Please place images in data/sample/ and re-run."
        )
    return paths[:max_images]


# ---------------------------------------------------------------------------
# Image manipulation
# ---------------------------------------------------------------------------

def _bicubic_upscale(lr_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic upscale of a (C, H, W) tensor to produce a naive baseline."""
    c, h, w = lr_tensor.shape
    return (
        F.interpolate(
            lr_tensor.unsqueeze(0),
            size=(h * scale, w * scale),
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .clamp(0.0, 1.0)
    )


def _make_lr_from_hr(hr_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """Simulate a degraded LR image from an HR tensor via bicubic downsampling."""
    c, h, w = hr_tensor.shape
    lr = (
        F.interpolate(
            hr_tensor.unsqueeze(0),
            size=(h // scale, w // scale),
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .clamp(0.0, 1.0)
    )
    return lr


_FONT_SEARCH_PATHS = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # macOS
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(font_size: int) -> ImageFont.ImageFont:
    """Return the first available TrueType font, or Pillow's built-in default."""
    for path in _FONT_SEARCH_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _add_label(img: Image.Image, text: str, font_size: int = 20) -> Image.Image:
    """Overlay a text label at the bottom-left of a PIL image."""
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    margin = 6
    x, y = margin, img.height - font_size - margin * 2
    # Drop shadow for readability
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def _build_comparison(
    lr_tensor: torch.Tensor,
    sr_tensor: torch.Tensor,
    hr_tensor: Optional[torch.Tensor],
    scale: int,
    image_name: str,
    metrics: dict,
) -> Image.Image:
    """
    Build a side-by-side composite:
      [LR (bicubic up)] | [ESRGAN SR] | [HR original]  (HR only if available)
    """
    target_h, target_w = sr_tensor.shape[1], sr_tensor.shape[2]

    # Bicubic upscale of LR for fair visual comparison
    lr_up = _bicubic_upscale(lr_tensor.cpu(), scale)

    def _to_pil(t: torch.Tensor) -> Image.Image:
        arr = tensor_to_numpy(t.cpu())
        return Image.fromarray(arr).resize((target_w, target_h), Image.LANCZOS)

    panels = [
        _add_label(_to_pil(lr_up), "Input (Bicubic ×4)"),
        _add_label(
            _to_pil(sr_tensor.cpu()),
            f"ESRGAN ×4  PSNR {metrics.get('psnr', 0):.2f} dB",
        ),
    ]
    if hr_tensor is not None:
        panels.append(_add_label(_to_pil(hr_tensor.cpu()), "Ground Truth HR"))

    total_w = sum(p.width for p in panels) + (len(panels) - 1) * 4
    composite = Image.new("RGB", (total_w, target_h), (40, 40, 40))
    x_offset = 0
    for panel in panels:
        composite.paste(panel, (x_offset, 0))
        x_offset += panel.width + 4

    return composite


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    sr_tensor: torch.Tensor,
    hr_tensor: Optional[torch.Tensor],
    lpips_model,
    device: torch.device,
) -> dict:
    """Return dict with psnr, ssim, lpips keys (None if HR unavailable)."""
    if hr_tensor is None:
        return {"psnr": None, "ssim": None, "lpips": None}

    # Match spatial sizes (SR might be slightly larger/smaller due to scale)
    sr = sr_tensor.cpu()
    hr = hr_tensor.cpu()
    if sr.shape != hr.shape:
        hr = F.interpolate(
            hr.unsqueeze(0),
            size=(sr.shape[1], sr.shape[2]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)

    psnr = compute_psnr(sr, hr)
    ssim = compute_ssim(sr, hr)
    lpips_val = _compute_lpips(lpips_model, sr, hr, device)
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], fmt: str = ".4f", na: str = "N/A") -> str:
    return f"{val:{fmt}}" if val is not None else na


def _generate_markdown(rows: list, gallery_dir_rel: str) -> str:
    """Return the full Markdown Results block ready to paste into README."""
    md_lines = [
        "## Results",
        "",
        "> **4× super-resolution on astronomical and historical images** — "
        "trained with a quick demo run (500 PSNR + 300 GAN iterations), showcasing the ESRGAN pipeline.",
        "",
        "| Image | PSNR (dB) | SSIM | LPIPS ↓ |",
        "|-------|-----------|------|---------|",
    ]
    for row in rows:
        psnr_s = _fmt(row["psnr"], ".2f")
        ssim_s = _fmt(row["ssim"], ".4f")
        lpips_s = _fmt(row["lpips"], ".4f")
        md_lines.append(f"| {row['name']} | {psnr_s} | {ssim_s} | {lpips_s} |")

    md_lines += ["", "### Before / After Gallery", ""]
    for row in rows:
        img_path = f"{gallery_dir_rel}/{row['name']}_comparison.png"
        md_lines.append(
            f'<img src="{img_path}" alt="{row["name"]} — LR vs ESRGAN SR" width="100%">'
        )
        md_lines.append("")

    md_lines += [
        "---",
        "",
        "**One-liner to reproduce:**",
        "```bash",
        "python scripts/train_quick_demo.py && python scripts/generate_results_gallery.py",
        "```",
    ]
    return "\n".join(md_lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Before/After results gallery with quality metrics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / INFERENCE_CONFIG),
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / DEMO_CHECKPOINT),
        help=f"Path to generator checkpoint (default: {DEMO_CHECKPOINT})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_REPO_ROOT / "data" / "sample"),
        help="Directory containing HR source images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / GALLERY_DIR),
        help="Directory to write composite images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=6,
        help="Maximum number of images to process (default: 6)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load inference config
    # ------------------------------------------------------------------
    config_path = args.config
    if not os.path.isfile(config_path):
        config_path = str(_REPO_ROOT / INFERENCE_CONFIG)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Override checkpoint from CLI
    cfg["model"]["checkpoint"] = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Config      : {config_path}\n")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = build_model(cfg, device)
    model.eval()
    scale = cfg["model"]["scale"]

    # ------------------------------------------------------------------
    # Load LPIPS (optional)
    # ------------------------------------------------------------------
    lpips_model = _try_load_lpips(device)

    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------
    image_paths = _ensure_images(args.data_dir, args.max_images)
    print(f"\nProcessing {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"  {p}")
    print()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list = []
    for img_path in image_paths:
        name = img_path.stem
        print(f"→ {name}")

        # Load as HR; simulate LR via bicubic downsampling
        hr_tensor = load_image(str(img_path))
        lr_tensor = _make_lr_from_hr(hr_tensor, scale)

        # Run ESRGAN inference
        sr_tensor = run_inference(model, lr_tensor, cfg, device)

        # Quality metrics
        metrics = _compute_metrics(sr_tensor, hr_tensor, lpips_model, device)
        print(
            f"   PSNR: {_fmt(metrics['psnr'], '.2f')} dB  "
            f"SSIM: {_fmt(metrics['ssim'], '.4f')}  "
            f"LPIPS: {_fmt(metrics['lpips'], '.4f')}"
        )

        # Side-by-side composite
        composite = _build_comparison(lr_tensor, sr_tensor, hr_tensor, scale, name, metrics)
        out_path = out_dir / f"{name}_comparison.png"
        composite.save(str(out_path))
        print(f"   Saved: {out_path}")

        rows.append({"name": name, **metrics})

    # ------------------------------------------------------------------
    # Write Markdown
    # ------------------------------------------------------------------
    # Use a repo-root-relative path for images in the Markdown
    gallery_dir_rel = os.path.relpath(str(out_dir), str(_REPO_ROOT))

    md_content = _generate_markdown(rows, gallery_dir_rel)

    md_out = _REPO_ROOT / GALLERY_MD_PATH
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md_content, encoding="utf-8")

    print(f"\n{'='*60}")
    print("Results gallery written to:")
    print(f"  Markdown : {md_out}")
    print(f"  Images   : {out_dir}/")
    print(f"{'='*60}\n")
    print("Paste-ready Markdown:\n")
    print(md_content)


if __name__ == "__main__":
    main()
