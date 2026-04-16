"""
Quick Demo Training Script for ESRGAN.

Runs a short two-phase training session suitable for generating a demo checkpoint:
  Phase 1: 500 PSNR-pretraining iterations
  Phase 2: 300 GAN fine-tuning iterations

Uses existing `src.train` helpers and writes TensorBoard logs to `runs/`.
Final checkpoint is saved to `checkpoints/esrgan_demo_4x.pth`.

Usage:
    python scripts/train_quick_demo.py [--data-dir data/sample] [--no-tensorboard]
    python -m scripts.train_quick_demo
"""

import argparse
import os
import sys
from itertools import cycle
from pathlib import Path

# Allow running as `python scripts/train_quick_demo.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.dataset import SRDataset
from src.discriminator import Discriminator
from src.losses import GANLoss, L1Loss, PerceptualLoss
from src.rrdb_net import RRDBNet
from src.train import (
    build_discriminator,
    build_generator,
    save_checkpoint,
    train_gan,
    train_psnr,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_PSNR_ITERS = 500
DEMO_GAN_ITERS = 300
DEMO_CHECKPOINT_INTERVAL = 200
DEMO_LOG_INTERVAL = 50
DEMO_CHECKPOINT_PATH = "checkpoints/esrgan_demo_4x.pth"
TENSORBOARD_LOG_DIR = "runs/esrgan_demo"

# Default model/optimizer settings matching train_config.yaml
_DEFAULT_CFG: dict = {
    "model": {
        "in_nc": 3,
        "out_nc": 3,
        "nf": 64,
        "nb": 23,
        "gc": 32,
        "scale": 4,
    },
    "optimizer": {
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.9,
        "beta2": 0.99,
    },
    "loss": {
        "l1_weight": 1.0,
        "perceptual_weight": 1.0,
        "gan_weight": 0.005,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str) -> dict:
    """Load YAML config and overlay demo-specific defaults."""
    if os.path.isfile(config_path):
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = {}

    # Ensure all required sections exist, falling back to defaults
    for section, defaults in _DEFAULT_CFG.items():
        if section not in cfg:
            cfg[section] = {}
        for key, val in defaults.items():
            cfg[section].setdefault(key, val)
    return cfg


def _make_demo_dataloader(data_dir: str, cfg: dict) -> DataLoader:
    """Build a tiny dataloader from data_dir images."""
    dataset = SRDataset(
        hr_dir=data_dir,
        hr_crop_size=cfg["model"].get("scale", 4) * 32,  # 128 for 4×
        scale=cfg["model"]["scale"],
        augment=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _try_create_writer(log_dir: str, enabled: bool):
    """Return a SummaryWriter if tensorboard is available, else None."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs → {log_dir}")
        return writer
    except ImportError:
        print("TensorBoard not installed; skipping scalar logging.")
        return None


def _log_scalar(writer, tag: str, value: float, step: int) -> None:
    if writer is not None:
        writer.add_scalar(tag, value, step)


# ---------------------------------------------------------------------------
# Demo training loop (thin wrappers around src.train functions)
# ---------------------------------------------------------------------------

def _run_psnr_phase(
    cfg: dict,
    generator: RRDBNet,
    optimizer_g: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    writer,
    checkpoint_dir: str,
) -> None:
    """Phase 1 with inline TensorBoard logging."""
    l1_loss_fn = L1Loss()
    data_iter = cycle(dataloader)

    generator.train()
    print(f"\n{'='*60}")
    print(f"Phase 1: PSNR Pretraining — {DEMO_PSNR_ITERS} iterations")
    print(f"{'='*60}\n")

    for i in range(DEMO_PSNR_ITERS):
        lr_t, hr_t = next(data_iter)
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)

        optimizer_g.zero_grad()
        sr = generator(lr_t)
        loss = l1_loss_fn(sr, hr_t)
        loss.backward()
        optimizer_g.step()

        _log_scalar(writer, "train/psnr_l1_loss", loss.item(), i + 1)

        if (i + 1) % DEMO_LOG_INTERVAL == 0:
            print(f"[PSNR] Iter {i+1}/{DEMO_PSNR_ITERS} | L1: {loss.item():.6f}")

        if (i + 1) % DEMO_CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"demo_psnr_iter_{i+1}.pth")
            torch.save({"generator": generator.state_dict(), "iteration": i + 1}, ckpt_path)
            print(f"  ✓ Checkpoint: {ckpt_path}")

    print("\nPhase 1 complete.\n")


def _run_gan_phase(
    cfg: dict,
    generator: RRDBNet,
    discriminator: Discriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    writer,
    checkpoint_dir: str,
) -> None:
    """Phase 2 with inline TensorBoard logging."""
    l1_w = cfg["loss"]["l1_weight"]
    perc_w = cfg["loss"]["perceptual_weight"]
    gan_w = cfg["loss"]["gan_weight"]

    l1_loss_fn = L1Loss()
    perc_loss_fn = PerceptualLoss().to(device)
    gan_loss_fn = GANLoss()

    data_iter = cycle(dataloader)
    generator.train()
    discriminator.train()

    print(f"\n{'='*60}")
    print(f"Phase 2: GAN Fine-tuning — {DEMO_GAN_ITERS} iterations")
    print(f"{'='*60}\n")

    for i in range(DEMO_GAN_ITERS):
        lr_t, hr_t = next(data_iter)
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)

        # Discriminator update
        optimizer_d.zero_grad()
        sr_detached = generator(lr_t).detach()
        pred_real = discriminator(hr_t)
        pred_fake = discriminator(sr_detached)
        loss_d = gan_loss_fn(pred_real, pred_fake, for_discriminator=True)
        loss_d.backward()
        optimizer_d.step()

        # Generator update
        optimizer_g.zero_grad()
        sr = generator(lr_t)
        pred_real = discriminator(hr_t).detach()
        pred_fake = discriminator(sr)

        loss_l1 = l1_loss_fn(sr, hr_t) * l1_w
        loss_perc = perc_loss_fn(sr, hr_t) * perc_w
        loss_gan = gan_loss_fn(pred_real, pred_fake, for_discriminator=False) * gan_w
        loss_g = loss_l1 + loss_perc + loss_gan
        loss_g.backward()
        optimizer_g.step()

        _log_scalar(writer, "train/gan_loss_d", loss_d.item(), i + 1)
        _log_scalar(writer, "train/gan_loss_g", loss_g.item(), i + 1)
        _log_scalar(writer, "train/gan_loss_l1", loss_l1.item(), i + 1)
        _log_scalar(writer, "train/gan_loss_perceptual", loss_perc.item(), i + 1)
        _log_scalar(writer, "train/gan_loss_adversarial", loss_gan.item(), i + 1)

        if (i + 1) % DEMO_LOG_INTERVAL == 0:
            print(
                f"[GAN] Iter {i+1}/{DEMO_GAN_ITERS} | "
                f"D: {loss_d.item():.4f} | "
                f"G: {loss_g.item():.4f} "
                f"(L1: {loss_l1.item():.4f}, "
                f"Perc: {loss_perc.item():.4f}, "
                f"Adv: {loss_gan.item():.4f})"
            )

        if (i + 1) % DEMO_CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"demo_gan_iter_{i+1}.pth")
            save_checkpoint(
                ckpt_path, generator, discriminator, optimizer_g, optimizer_d, i + 1
            )
            print(f"  ✓ Checkpoint: {ckpt_path}")

    print("\nPhase 2 complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick demo training — 500 PSNR + 300 GAN iterations.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML (optional; falls back to built-in defaults)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing HR training images (overrides config hr_dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEMO_CHECKPOINT_PATH,
        help=f"Path for the final demo checkpoint (default: {DEMO_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard scalar logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _load_cfg(args.config)

    # Resolve data directory: CLI > config > fallback
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = cfg.get("data", {}).get("hr_dir", "data/sample")

    # Make path absolute relative to repo root when running from any CWD
    if not os.path.isabs(data_dir):
        data_dir = str(_REPO_ROOT / data_dir)

    print(f"Using HR data directory : {data_dir}")
    print(f"Final checkpoint will be: {args.output}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build models
    generator = build_generator(cfg).to(device)
    discriminator = build_discriminator().to(device)

    opt_cfg = cfg["optimizer"]
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=opt_cfg["lr_g"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=opt_cfg["lr_d"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
    )

    dataloader = _make_demo_dataloader(data_dir, cfg)
    print(f"Dataset: {len(dataloader.dataset)} images\n")

    checkpoint_dir = str(_REPO_ROOT / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer (optional)
    tb_log_dir = str(_REPO_ROOT / TENSORBOARD_LOG_DIR)
    writer = _try_create_writer(tb_log_dir, not args.no_tensorboard)

    try:
        # Phase 1: PSNR pretraining
        _run_psnr_phase(
            cfg=cfg,
            generator=generator,
            optimizer_g=optimizer_g,
            dataloader=dataloader,
            device=device,
            writer=writer,
            checkpoint_dir=checkpoint_dir,
        )

        # Phase 2: GAN fine-tuning
        _run_gan_phase(
            cfg=cfg,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            dataloader=dataloader,
            device=device,
            writer=writer,
            checkpoint_dir=checkpoint_dir,
        )

        # Save final demo checkpoint
        final_path = args.output
        if not os.path.isabs(final_path):
            final_path = str(_REPO_ROOT / final_path)

        save_checkpoint(
            final_path,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            DEMO_PSNR_ITERS + DEMO_GAN_ITERS,
        )
        print(f"\n✓ Demo checkpoint saved: {final_path}")
        print(
            "\nNext step — generate the results gallery:\n"
            "  python scripts/generate_results_gallery.py\n"
        )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
