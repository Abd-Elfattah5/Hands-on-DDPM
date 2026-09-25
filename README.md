# Hands-on Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, from-scratch implementation and benchmarking pipeline for continuous Gaussian Denoising Diffusion Probabilistic Models (DDPM) in PyTorch. Developed for the deep generative vision modeling benchmark (`GenCV003`).

---

## Overview

This repository implements a ground-up **Denoising Diffusion Probabilistic Model (DDPM)** on **CIFAR-10** ($32 \times 32 \times 3$). Following Ho et al. (2020, *Denoising Diffusion Probabilistic Models*), generation is formulated as a 1,000-step reverse Markov chain that iteratively denoises pure Gaussian noise into photorealistic images.

### Key Architectural Concepts
* **Forward Diffusion ($q(x_t | x_0)$)**: Analytical closed-form noise injection using linear or cosine variance schedules across $T = 1000$ steps.
* **Simplified Noise Objective ($L_{\text{simple}}$)**: The neural network is trained to predict the added noise vector $\epsilon \in \mathbb{R}^{B \times 3 \times 32 \times 32}$, bypassing the pixel-averaging blurriness of $L_2$ reconstruction losses.
* **Conditional U-Net Backbone**: Downsampling and upsampling with time-conditioned ResNet blocks, Group Normalization, and multi-head spatial self-attention at $16 \times 16$ and $8 \times 8$ resolutions.
* **Reverse Sampling Engine (Algorithm 2)**: Reverses the diffusion trajectory from $x_T \sim \mathcal{N}(0, I)$ down to clean image $x_0$.
* **Coherent Ecosystem**: Mirrors the modular architecture, CLI interface, and benchmarking conventions established in the sibling repository `Hands-on VAE`.

---

## Project Structure

```text
├── configs/
│   └── cifar10_baseline.yaml       # T=1000, linear schedule, UNet [64, 128, 256]
├── docs/
│   ├── adr/                         # Architecture Decision Records
│   └── reports/
│       └── 001-baseline-ddpm-report.md # Formal DDPM Technical Report
├── specs/                           # Feature specs and implementation tasks
├── src/
│   ├── cli/
│   │   └── main.py                 # CLI: verify, train, evaluate, sample, denoise-strip, benchmark
│   ├── configs/
│   │   └── schema.py               # Typed YAML schema validator
│   ├── data/
│   │   ├── cifar10.py              # CIFAR-10 data loader & transforms [-1, 1]
│   │   └── mnist.py                # MNIST data loader & transforms
│   ├── evaluation/
│   │   ├── evaluator.py            # Test loss evaluation
│   │   ├── metrics.py              # Fréchet Inception Distance (FID) & Inception Score (IS)
│   │   └── visualizer.py           # Reverse step-by-step strips (x_1000 -> x_0) & 1024x1024 grids
│   ├── models/
│   │   ├── base.py                 # Abstract base classes
│   │   ├── registry.py             # Component registries & factory
│   │   ├── embeddings.py           # Sinusoidal timestep embeddings
│   │   ├── attention.py            # Spatial multi-head self-attention
│   │   ├── resnet.py               # Time-conditioned ResNet block
│   │   ├── unet.py                 # Complete conditional U-Net backbone
│   │   └── gaussian_diffusion.py   # Variance schedules, q(x_t|x_0), p(x_{t-1}|x_t), sampling
│   ├── training/
│   │   ├── checkpoint.py           # Checkpoint serializer/restorer with EMA
│   │   └── trainer.py              # Training loop with EMA tracking and loss plotting
│   └── utils/
│       ├── logging.py              # Structured ISO-8601 logging
│       └── seeding.py              # Global random seeding
├── tests/
│   └── unit/                       # Shape tests, diffusion formula tests, sampling tests
├── HANDOFF.md                      # Complete implementation handoff and mathematical guide
├── pyproject.toml                  # Editable package configuration
└── README.md                       # This file
```

---

## Detailed Implementation Handoff

For the complete mathematical derivations, architecture specifications, GPU memory budget, and phased task breakdown, refer to **`HANDOFF.md`**.

---

## License

This project is licensed under the MIT License.
