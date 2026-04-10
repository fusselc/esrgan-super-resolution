"""
ESRGAN Two-Phase Training Script.

Phase 1 - PSNR Pretraining:
  - Generator only
  - Loss: L1

Phase 2 - GAN Fine-tuning:
  - Generator + Discriminator
  - Loss: L1 + Perceptual (VGG) + Adversarial (RaGAN)

All hyperparameters are loaded from a YAML config file.
Debug mode: ~2000 iterations per phase for quick validation.
"""

import argparse
import os
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from .dataset import SRDataset
from .discriminator import Discriminator
from .losses import GANLoss, L1Loss, PerceptualLoss
from .rrdb_net import RRDBNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_generator(cfg: dict) -> RRDBNet:
    m = cfg["model"]
    return RRDBNet(
        in_nc=m["in_nc"],
        out_nc=m["out_nc"],
        nf=m["nf"],
        nb=m["nb"],
        gc=m["gc"],
        scale=m["scale"],
    )


def build_discriminator() -> Discriminator:
    return Discriminator(in_nc=3, ndf=64)


def save_checkpoint(
    path: str,
    generator: RRDBNet,
    discriminator: Discriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    iteration: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(
    path: str,
    generator: RRDBNet,
    discriminator: Discriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    return ckpt.get("iteration", 0)


def make_dataloader(cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    dl_cfg = cfg["dataloader"]
    dataset = SRDataset(
        hr_dir=data_cfg["hr_dir"],
        hr_crop_size=data_cfg["hr_crop_size"],
        scale=data_cfg["scale"],
        augment=data_cfg.get("augment", True),
    )
    return DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=dl_cfg.get("shuffle", True),
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Phase 1: PSNR Pretraining
# ---------------------------------------------------------------------------

def train_psnr(
    cfg: dict,
    generator: RRDBNet,
    optimizer_g: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    num_iters: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    log_interval: int = 100,
    start_iter: int = 0,
) -> None:
    """Phase 1: Generator-only L1 pretraining."""
    l1_loss_fn = L1Loss()
    data_iter = cycle(dataloader)

    generator.train()
    print(f"\n{'='*60}")
    print(f"Phase 1: PSNR Pretraining — {num_iters} iterations")
    print(f"{'='*60}\n")

    for i in range(start_iter, num_iters):
        lr, hr = next(data_iter)
        lr, hr = lr.to(device), hr.to(device)

        optimizer_g.zero_grad()
        sr = generator(lr)
        loss = l1_loss_fn(sr, hr)
        loss.backward()
        optimizer_g.step()

        if (i + 1) % log_interval == 0:
            print(f"[PSNR] Iter {i+1}/{num_iters} | L1: {loss.item():.6f}")

        if (i + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"psnr_iter_{i+1}.pth")
            torch.save({"generator": generator.state_dict(), "iteration": i + 1}, ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

    print("\nPhase 1 complete.\n")


# ---------------------------------------------------------------------------
# Phase 2: GAN Fine-tuning
# ---------------------------------------------------------------------------

def train_gan(
    cfg: dict,
    generator: RRDBNet,
    discriminator: Discriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    num_iters: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    log_interval: int = 100,
    start_iter: int = 0,
) -> None:
    """Phase 2: GAN fine-tuning with L1 + Perceptual + RaGAN losses."""
    loss_weights = cfg["loss"]
    l1_w = loss_weights["l1_weight"]
    perc_w = loss_weights["perceptual_weight"]
    gan_w = loss_weights["gan_weight"]

    l1_loss_fn = L1Loss()
    perc_loss_fn = PerceptualLoss().to(device)
    gan_loss_fn = GANLoss()

    data_iter = cycle(dataloader)

    generator.train()
    discriminator.train()

    print(f"\n{'='*60}")
    print(f"Phase 2: GAN Fine-tuning — {num_iters} iterations")
    print(f"Loss weights: L1={l1_w}, Perceptual={perc_w}, GAN={gan_w}")
    print(f"{'='*60}\n")

    for i in range(start_iter, num_iters):
        lr, hr = next(data_iter)
        lr, hr = lr.to(device), hr.to(device)

        # ── Discriminator update ──────────────────────────────────────────
        optimizer_d.zero_grad()
        sr = generator(lr).detach()  # detach: no gradients needed through generator here

        pred_real = discriminator(hr)
        pred_fake = discriminator(sr)
        loss_d = gan_loss_fn(pred_real, pred_fake, for_discriminator=True)
        loss_d.backward()
        optimizer_d.step()

        # ── Generator update ──────────────────────────────────────────────
        optimizer_g.zero_grad()
        sr = generator(lr)  # recompute with gradient graph for G update

        pred_real = discriminator(hr).detach()
        pred_fake = discriminator(sr)

        loss_l1 = l1_loss_fn(sr, hr) * l1_w
        loss_perc = perc_loss_fn(sr, hr) * perc_w
        loss_gan = gan_loss_fn(pred_real, pred_fake, for_discriminator=False) * gan_w
        loss_g = loss_l1 + loss_perc + loss_gan
        loss_g.backward()
        optimizer_g.step()

        if (i + 1) % log_interval == 0:
            print(
                f"[GAN] Iter {i+1}/{num_iters} | "
                f"D: {loss_d.item():.4f} | "
                f"G: {loss_g.item():.4f} "
                f"(L1: {loss_l1.item():.4f}, "
                f"Perc: {loss_perc.item():.4f}, "
                f"Adv: {loss_gan.item():.4f})"
            )

        if (i + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"gan_iter_{i+1}.pth")
            save_checkpoint(
                ckpt_path, generator, discriminator, optimizer_g, optimizer_d, i + 1
            )
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

    print("\nPhase 2 complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESRGAN Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume GAN training from",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode (use full iteration counts from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve debug mode
    debug_cfg = cfg.get("debug", {})
    use_debug = debug_cfg.get("enabled", False) and not args.no_debug

    train_cfg = cfg["training"]
    if use_debug:
        psnr_iters = debug_cfg.get("psnr_iters", 2000)
        gan_iters = debug_cfg.get("gan_iters", 2000)
        checkpoint_interval = debug_cfg.get("checkpoint_interval", 500)
        print(">>> DEBUG MODE: short run <<<")
    else:
        psnr_iters = train_cfg["psnr_iters"]
        gan_iters = train_cfg["gan_iters"]
        checkpoint_interval = train_cfg["checkpoint_interval"]

    log_interval = train_cfg.get("log_interval", 100)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Build dataloader
    dataloader = make_dataloader(cfg)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Phase 1: PSNR Pretraining ────────────────────────────────────────────
    train_psnr(
        cfg=cfg,
        generator=generator,
        optimizer_g=optimizer_g,
        dataloader=dataloader,
        device=device,
        num_iters=psnr_iters,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        log_interval=log_interval,
    )

    # ── Phase 2: GAN Fine-tuning ─────────────────────────────────────────────
    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        start_iter = load_checkpoint(
            args.resume, generator, discriminator, optimizer_g, optimizer_d, device
        )
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    train_gan(
        cfg=cfg,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        dataloader=dataloader,
        device=device,
        num_iters=gan_iters,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        log_interval=log_interval,
        start_iter=start_iter,
    )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final.pth")
    save_checkpoint(
        final_path, generator, discriminator, optimizer_g, optimizer_d, gan_iters
    )
    print(f"\nFinal checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()
