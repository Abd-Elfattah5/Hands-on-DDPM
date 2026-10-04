# Hands-on Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, from-scratch implementation and benchmarking pipeline for continuous Gaussian Denoising Diffusion Probabilistic Models (DDPM) in PyTorch. Developed for the deep generative vision modeling benchmark (`GenCV003`).

---

## Overview

This repository implements a ground-up **Denoising Diffusion Probabilistic Model (DDPM)** on **CIFAR-10** ($32 \times 32 \times 3$). Following Ho et al. (2020, *Denoising Diffusion Probabilistic Models*), generation is a 1,000-step reverse Markov chain that iteratively denoises pure Gaussian noise into an image.

### Key Features
* **Forward Diffusion ($q(x_t | x_0)$)**: Closed-form noise injection with `linear` or `cosine` variance schedules over $T = 1000$ steps.
* **Simplified Noise Objective ($L_{\text{simple}}$)**: The network predicts the added noise $\epsilon$ instead of pixels, avoiding the $L_2$ pixel-averaging blur of the VAE.
* **Time-Conditioned U-Net (16.06 M parameters)**: Channels $[64, 128, 256]$, residual blocks with sinusoidal timestep embeddings, Group Normalization, and 4-head self-attention at $16 \times 16$ and $8 \times 8$.
* **EMA Weights**: An exponential moving average of the weights (decay 0.9999) is used for sampling and evaluation.
* **Reverse Sampling (Algorithm 2)**: Seeded, deterministic ancestral sampling from $x_T \sim \mathcal{N}(0, I)$ to $x_0$, with denoising strips.
* **Shared Benchmark Protocol**: FID and Inception Score over 5,000 generated vs. 5,000 real CIFAR-10 test images, identical to the sibling repository `Hands-on VAE`.
* **Measured 3 GB GPU Budget**: Every workflow fits in 3 GB on a 4 GB Quadro T2000 (see `specs/001-create-ddpm/research/gpu_memory_results.md`).
* **Local CLI + Colab**: Train locally with the `ddpm` CLI, or run the same CLI on a faster Colab GPU through `notebooks/ddpm_colab_training.ipynb`.

---

## Quantitative Benchmark Results (CIFAR-10)

Shared protocol with `Hands-on VAE`: 5,000 generated images vs. the first 5,000 CIFAR-10 test images, torchvision Inception-v3, IS over 10 splits. Full analysis: [`docs/reports/001-baseline-ddpm-report.md`](docs/reports/001-baseline-ddpm-report.md).

| Model | FID ↓ | IS ↑ | Parameters | Passes per image |
|---|---|---|---|---|
| VAE baseline (`Hands-on VAE` 001) | 169.02 | 2.11 ± 0.03 | 1.60 M | 1 |
| VAE enhanced (`Hands-on VAE` 002) | 181.00 | 1.68 ± 0.04 | 2.25 M | 1 |
| **DDPM (this repo)** | **39.69** | **5.18 ± 0.14** | 16.06 M | 1,000 |

DDPM details:
- Trained for 80 epochs (28,080 optimizer steps, effective batch 128) on a **Tesla T4** (Google Colab) in 5.25 h.
- Evaluated with **raw weights** on a **Quadro T2000**. The EMA weights had not converged at 80 epochs; see the report, §4.
- Test noise-prediction loss 0.0309.
- 2.16 s per image to sample (2.99 h for the 5,000-image benchmark).
- Samples spread over 322 distinct Inception classes.
- No memorization: nearest training-image L2 distance ≥ 4.75.

| Samples (raw weights, seed 42) | Reverse process (t = 200 → 0) |
|---|---|
| ![samples](docs/reports/figures/sample_grid.png) | ![strip](docs/reports/figures/denoise_strip_t200_to_0.png) |

---

## Project Structure

```text
├── configs/
│   ├── cifar10_baseline.yaml       # Local: T=1000 linear, UNet [64,128,256], batch 32 × accumulation 4
│   └── cifar10_colab.yaml          # Colab: same model and optimization, batch 128 × accumulation 1
├── docs/
│   ├── adr/
│   │   └── 0001-deviations-from-hands-on-vae-conventions.md # Justified differences from Hands-on VAE
│   └── reports/
│       └── 001-baseline-ddpm-report.md # GenCV003 technical report (written after training)
├── notebooks/
│   └── ddpm_colab_training.ipynb   # Colab driver: mount Drive → install → verify → train --resume
├── specs/001-create-ddpm/          # Spec, plan, research, data model, CLI contract, quickstart
├── src/
│   ├── cli/main.py                 # CLI: verify, train, evaluate, sample, denoise-strip, benchmark
│   ├── configs/schema.py           # Typed YAML schema validator
│   ├── data/cifar10.py             # CIFAR-10 loader, [-1, 1] transforms, seeded train/val split
│   ├── evaluation/
│   │   ├── evaluator.py            # Test noise-prediction loss
│   │   ├── metrics.py              # Fréchet Inception Distance (FID) & Inception Score (IS)
│   │   └── visualizer.py           # Sample grids, denoising strips, nearest-neighbor panel
│   ├── models/
│   │   ├── base.py, types.py       # Abstract base classes, typed output contracts
│   │   ├── registry.py             # Component registries & factory
│   │   ├── schedules.py            # Linear & cosine noise schedules
│   │   ├── embeddings.py           # Sinusoidal timestep embeddings
│   │   ├── attention.py            # Multi-head spatial self-attention
│   │   ├── resnet.py               # Time-conditioned residual block
│   │   ├── unet.py                 # Conditional U-Net backbone
│   │   └── gaussian_diffusion.py   # q(x_t|x_0), L_simple, Algorithm 2 sampling
│   ├── training/
│   │   ├── ema.py                  # EMA weights
│   │   ├── checkpoint.py           # Checkpoint save/restore (raw + EMA weights, history)
│   │   └── trainer.py              # Training loop: accumulation, warmup, clipping, EMA
│   └── utils/                      # Seeding, logging, GPU memory measurement
├── tests/unit/                     # Shape, schedule, diffusion, sampling, metric, memory tests
├── ARCHITECTURE_DEEP_DIVE.md       # Design rationale for every decision
├── CONTEXT.md                      # Domain glossary
├── HANDOFF.md                      # Implementation handoff and mathematical guide
├── pyproject.toml                  # Editable package configuration (`ddpm` console script)
└── README.md                       # This file
```

---

## Installation & Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Abd-Elfattah5/Hands-on-DDPM.git
cd Hands-on-DDPM

# Create and activate virtual environment
python3 -m venv DDPM
source DDPM/bin/activate

# Install PyTorch with CUDA 12.4 support (matches the tested environment)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Verify Architecture & Tensor Integrity

Before training, run the pre-training integrity gate:

```bash
ddpm verify --config configs/cifar10_baseline.yaml
```
*Checks schedule correctness, forward-process statistics, U-Net output shapes, gradient flow, a short reverse pass, and that one training step stays within the 3 GB GPU budget. Use `--skip-memory` on CPU-only machines.*

---

## CLI Reproduction Guide

The `ddpm` command-line interface provides the complete workflow. Run `ddpm --help` or `ddpm <command> --help` for every option. Sampling commands default to EMA weights; the reported results use `--weights raw` (see the results section above).

### 1. Train the Baseline DDPM (Local GPU)

```bash
ddpm train --config configs/cifar10_baseline.yaml
```
*Trains for 100 epochs at an effective batch of 128 (32 × 4 gradient accumulation), about 10 h on a Quadro T2000 (~6 min per epoch, measured). Emits `latest.pt` (every epoch), `best_checkpoint.pt`, `final_checkpoint.pt`, `metrics.json`, `loss_curve.png`, `train.log` and `samples/epoch_NNN.png` to `artifacts/runs/cifar10_baseline/`.*

A fresh run refuses to start in a directory that already holds checkpoints (so a finished run is never overwritten by accident); pass `--overwrite` to start over deliberately. Resume after an interruption:

```bash
ddpm train --config configs/cifar10_baseline.yaml --resume artifacts/runs/cifar10_baseline/latest.pt
```

### 2. Evaluate Held-Out Test Split

```bash
ddpm evaluate --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt
```
*Writes the test noise-prediction loss (EMA weights) to `artifacts/runs/cifar10_baseline/eval_metrics.json`.*

### 3. Quantitative Benchmarking (FID & Inception Score)

```bash
ddpm benchmark --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --weights raw --num-samples 5000
```
*Generates 5,000 images (about 2.8 h on a T2000), computes FID against the first 5,000 CIFAR-10 test images and IS over 10 splits, and writes `artifacts/eval/benchmark_metrics.json`.*

### 4. Generate High-Resolution Sample Grids (+ Memorization Check)

```bash
ddpm sample \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --num-samples 64 \
  --out artifacts/samples/sample_grid_1024.png \
  --upscale 4 \
  --seed 42 \
  --weights raw \
  --nearest
```
*Outputs a $1024 \times 1024$ grid of 64 samples. `--nearest` also writes `sample_grid_1024_nearest.png`: each generated image next to its 3 closest training images, to check that the model is not copying training data.*

### 5. Visualize the Reverse Denoising Process

```bash
ddpm denoise-strip \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --num-images 8 \
  --out artifacts/strips/denoise_strip.png \
  --seed 42 \
  --weights raw
```
*Rows are images, columns are timesteps $t = 1000, 800, 600, 400, 200, 100, 50, 0$.*

### 6. Reproduce from the Published Checkpoint

The reported checkpoint is attached to release [`v0.1.0-ddpm-baseline`](https://github.com/Abd-Elfattah5/Hands-on-DDPM/releases/tag/v0.1.0-ddpm-baseline):

```bash
gh release download v0.1.0-ddpm-baseline -R Abd-Elfattah5/Hands-on-DDPM -D artifacts/runs/cifar10_baseline
mv artifacts/runs/cifar10_baseline/ddpm_cifar10_baseline_ep080.pt artifacts/runs/cifar10_baseline/best_checkpoint.pt
ddpm benchmark --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --weights raw --num-samples 5000 --seed 42
```
*Reported results use `--weights raw` (EMA not yet converged after 80 epochs). Expected: FID ≈ 39.7, IS ≈ 5.2 on the same hardware and software; other GPUs should match within ±5% FID / ±0.3 IS.*

### 7. Run Test Suite

```bash
pytest tests/ -v
```
*Runs 90 tests covering tensor shapes, schedule and forward-process formulas, gradient flow, sampling determinism, EMA, checkpoint/resume, FID/IS calculations, the CLI contract and the 3 GB memory budget (CUDA only).*

---

## Training on Google Colab (Faster GPU)

The official training run uses a Colab GPU (e.g., T4), which has more compute and memory bandwidth than the local T2000. The notebook contains no model code; it runs the same `ddpm` CLI.

1. Open `notebooks/ddpm_colab_training.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Add a Colab Secret named `GH_TOKEN` (a GitHub token with read access to this private repository), or upload a zip of the repository to Google Drive.
3. **Run all**. The notebook mounts Google Drive, installs the package, runs `ddpm verify`, then trains with:
   ```bash
   ddpm train --config configs/cifar10_colab.yaml \
              --output-dir /content/drive/MyDrive/hands-on-ddpm/runs/cifar10_colab
   ```
4. If the session disconnects, reconnect and **Run all** again: the notebook detects `latest.pt` on Drive and adds `--resume` automatically (at most one epoch is lost).

---

## License

This project is licensed under the MIT License.
