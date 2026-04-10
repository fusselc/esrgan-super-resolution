# esrgan-super-resolution

**Educational PyTorch reimplementation of ESRGAN** — a modular, config-driven super-resolution pipeline featuring perceptual + adversarial losses and tiled inference. Designed for real-world super-resolution tasks including astronomical imagery and historical photo restoration.

---

## Overview

[ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)](https://arxiv.org/abs/1809.00219) improves upon SRGAN with:

- **RRDB blocks** (Residual in Residual Dense Blocks) — no Batch Normalization
- **Relativistic average GAN (RaGAN)** adversarial training
- **VGG perceptual loss** extracted before activation (conv5_4 feature maps)
- **Two-phase training**: PSNR pretraining followed by GAN fine-tuning

This repository is an educational, research-grade baseline suitable for extensions such as Real-ESRGAN and SwinIR.

---

## Repository Structure

```
esrgan-super-resolution/
├── src/
│   ├── __init__.py
│   ├── rrdb_net.py       # RRDB Generator (23 blocks, 64ch, nearest-neighbor upsampling)
│   ├── discriminator.py  # VGG-style Discriminator (no BN, scalar output)
│   ├── losses.py         # L1, VGG Perceptual, RaGAN Adversarial losses
│   ├── dataset.py        # HR→LR bicubic downsampling + augmentation dataset
│   ├── train.py          # Two-phase training script
│   ├── inference.py      # Tiled inference + batch processing
│   └── utils.py          # PSNR, SSIM, image I/O, tensor↔numpy utilities
├── configs/
│   ├── train_config.yaml
│   └── inference_config.yaml
├── notebooks/
│   ├── demo_upscale.ipynb
│   └── astronomy_pipeline.ipynb
├── tests/
│   ├── test_rrdb_net.py
│   ├── test_discriminator.py
│   ├── test_losses.py
│   └── test_inference.py
├── data/
│   └── sample/           # Place training HR images here
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/fusselc/esrgan-super-resolution.git
cd esrgan-super-resolution
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, torchvision 0.15+

---

## Training

### 1. Prepare Training Data

Place HR (high-resolution) images in `data/sample/` or any directory. Update `configs/train_config.yaml`:

```yaml
data:
  hr_dir: "data/sample"  # ← path to your HR images
```

Dataset guidance:

- **Supported formats:** PNG, JPG/JPEG (also BMP/TIFF/WEBP are accepted by the loader)
- **Recommended resolution:** use images with the shorter side **> 256 px** for stable random crops
- **Recommended dataset size:** at least **~100 HR images** for meaningful training
- **LR generation:** LR patches are created **on-the-fly via bicubic downsampling** from HR patches

### 2. Run Training (Debug Mode — default)

```bash
python -m src.train --config configs/train_config.yaml
```

Debug mode runs ~2,000 iterations per phase on CPU for quick validation. To run the full training:

```bash
python -m src.train --config configs/train_config.yaml --no-debug
```

### Two-Phase Training

| Phase | Description | Loss |
|-------|-------------|------|
| Phase 1 | PSNR Pretraining (generator only) | L1 pixel loss |
| Phase 2 | GAN Fine-tuning (generator + discriminator) | L1 + VGG Perceptual + RaGAN adversarial |

Checkpoints are saved to `checkpoints/` at configurable intervals.

Sample training log format:

```text
[Iter 1000] L1: 0.023 | Perceptual: 0.84 | GAN: 0.12
```

Example (brief debug pretraining curve):

![Training loss curve](assets/training_loss_curve.png)

---

## Inference

### Single Image

```bash
python -m src.inference \
  --config configs/inference_config.yaml \
  --input path/to/low_res.jpg \
  --output results/ \
  --checkpoint checkpoints/final.pth
```

### Batch Processing (Directory)

```bash
python -m src.inference \
  --config configs/inference_config.yaml \
  --input path/to/lr_images/ \
  --output results/ \
  --checkpoint checkpoints/final.pth
```

### Tiled Inference

Large images are automatically processed in overlapping tiles to avoid memory issues and visible seams. Configure in `configs/inference_config.yaml`:

```yaml
tiling:
  enabled: true
  tile_size: 128       # LR tile size in pixels
  tile_overlap: 16     # Overlap in pixels to blend tiles
```

---

## Architecture Details

### Generator (RRDBNet)

- **No Batch Normalization** anywhere in the network
- **23 RRDB blocks**, each containing 3 Residual Dense Blocks
- **64 base channels**, 32 growth channels per dense layer
- **Residual scaling = 0.2** at every residual connection
- **Upsampling**: nearest-neighbor interpolation + Conv2d × 2 (total 4× scale)
- **LeakyReLU (α=0.2)** throughout
- **Kaiming Normal initialization**

### Discriminator

- VGG-style feature extractor
- No Batch Normalization
- Outputs raw scalar logit per image
- Used in Relativistic average GAN (RaGAN) framework

### Loss Functions

| Loss | Description |
|------|-------------|
| L1 | Pixel-wise mean absolute error |
| Perceptual | VGG19 conv5_4 feature distance (pre-activation, frozen) |
| Adversarial | Relativistic average GAN (RaGAN) using BCEWithLogitsLoss |

Perceptual loss behavior:

- VGG19 ImageNet weights are **automatically downloaded by torchvision** on first use
- Approximate download size: **~548 MB**
- Weights are cached locally (torch cache), so subsequent runs reuse them

---

## Evaluation (PSNR / SSIM)

You can compute full-reference metrics with `src/utils.py`:

```python
from src.utils import load_image, compute_psnr, compute_ssim

sr = load_image("results/example_sr.png")      # (C,H,W), float in [0,1]
hr = load_image("data/gt/example_hr.png")      # must match SR size

psnr = compute_psnr(sr, hr, max_val=1.0)
ssim = compute_ssim(sr, hr, max_val=1.0)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
```

---

## Testing

```bash
pytest tests/ -v
```

All tests run on CPU and complete in under 2 minutes.

---

## Notebooks

- `notebooks/demo_upscale.ipynb` — Load a model, run inference, and visualize results
- `notebooks/astronomy_pipeline.ipynb` — End-to-end super-resolution pipeline for telescope imagery

---

## Example Outputs

Examples below were generated from a brief debug pretraining run (CPU) and then inference on held-out synthetic samples.

### Standard Image (LR → SR)

![Standard example comparison](assets/examples/standard_comparison.png)

### Astronomy Example (LR → SR)

![Astronomy example comparison](assets/examples/astronomy_comparison.png)

---

## Limitations

- Training requires a dataset of HR images (minimum ~100 images recommended)
- Full training (50,000+ GAN iterations) requires a GPU for reasonable speed
- Perceptual loss downloads VGG19 weights (~548 MB) on first use
- 4× upscaling only; extend `RRDBNet` for other scale factors

---

## References

- Wang, X. et al. (2018). *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*. [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
- Lim, B. et al. (2017). *Enhanced Deep Residual Networks for Single Image Super-Resolution*. CVPR Workshop.
- Jolicoeur-Martineau, A. (2018). *The relativistic discriminator: a key element missing from standard GAN*. [arXiv:1807.00734](https://arxiv.org/abs/1807.00734)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
