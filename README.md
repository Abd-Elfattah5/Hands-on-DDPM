# Hands-on Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, from-scratch implementation and benchmarking pipeline for continuous Gaussian Denoising Diffusion Probabilistic Models (DDPM) in PyTorch. Developed for the deep generative vision modeling benchmark (`GenCV003`).

> **Status**: Specified and planned (`specs/001-create-ddpm/`); implementation in progress. The commands below follow the CLI contract in `specs/001-create-ddpm/contracts/cli.md` and become available as each phase lands.

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

The `ddpm` command-line interface provides the complete workflow. Run `ddpm --help` or `ddpm <command> --help` for every option.

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
ddpm benchmark --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --num-samples 5000
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
  --nearest
```
*Outputs a $1024 \times 1024$ grid of 64 samples. `--nearest` also writes `sample_grid_1024_nearest.png`: each generated image next to its 3 closest training images, to check that the model is not copying training data.*

### 5. Visualize the Reverse Denoising Process

```bash
ddpm denoise-strip \
  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt \
  --num-images 8 \
  --out artifacts/strips/denoise_strip.png \
  --seed 42
```
*Rows are images, columns are timesteps $t = 1000, 800, 600, 400, 200, 100, 50, 0$.*

### 6. Run Test Suite

```bash
pytest tests/ -v
```
*Verifies tensor shapes, schedule and forward-process formulas, gradient flow, sampling determinism, EMA, checkpoint/resume, FID/IS calculations, and the 3 GB memory budget (CUDA only).*

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
