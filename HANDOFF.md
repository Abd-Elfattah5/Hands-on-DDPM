# HANDOFF: Hands-on DDPM (Denoising Diffusion Probabilistic Models)

**Repository**: `Hands-on DDPM` (`/home/amousa1/Projects/Hands-on DDPM`)  
**Assignment Reference**: `GenCV003` (Part 2: Denoising Diffusion Generative Modeling Benchmark)  
**Target Architecture**: Continuous Gaussian Denoising Diffusion Probabilistic Model (DDPM)  
**Target Dataset**: CIFAR-10 ($32 \times 32 \times 3$, normalized to $[-1, 1]$)  
**Target Hardware**: Local NVIDIA Quadro T2000 (4GB GDDR6 VRAM, Compute Capability 7.5)  
**Language/Runtime**: Python 3.12+, PyTorch 2.6+cu124  
**Date**: September 2026  
**Status**: Initialized & Structured; Ready for Phased Implementation  

---

## 1. Project Mission & Objective

This repository is the sibling project to `Hands-on VAE`. In the VAE project, we proved both mathematically and empirically that continuous single-stage Variational Autoencoders inherently suffer from the **$L_2$ conditional mean smoothing trap**—producing soft, blurred reconstructions because the pixel-wise loss forces the model to predict the conditional expectation $\mathbb{E}[x|z]$.

The objective of **Hands-on DDPM** is to implement a **from-scratch Denoising Diffusion Probabilistic Model (DDPM)** on CIFAR-10 based on Ho et al. (2020, *Denoising Diffusion Probabilistic Models*) without relying on external generative library wrappers (such as HuggingFace `diffusers`).

By modeling generation as a 1,000-step iterative reverse denoising Markov chain and training the neural network to predict the **injected noise vector $\epsilon$** rather than raw image pixels, DDPM eliminates the $L_2$ pixel-averaging blur, achieving crisp edges and photorealistic visual synthesis.

---

## 2. Complete Mathematical Formulation

```
                               DIFFUSION PROCESS OVERVIEW
                               
  Forward Diffusion Process q(x_t | x_{t-1}) (Deterministic Gaussian Noise Injection, No Learning)
  ─────────────────────────────────────────────────────────────────────────────────────────────────►
  x₀ (Clean Image) ──► x₁ ──► ... ──► x_{t-1} ──► x_t ──► ... ──► x_T ~ N(0, I) (Pure Noise)
  ◄─────────────────────────────────────────────────────────────────────────────────────────────────
  Reverse Denoising Process p_θ(x_{t-1} | x_t) (Learned U-Net Noise Estimator ε_θ(x_t, t))
```

### 2.1 The Forward Diffusion Process $q(x_t | x_{t-1})$
Given a clean data sample $x_0 \sim q(x_0)$ bounded in $[-1, 1]$, the forward process adds Gaussian noise across $T = 1000$ discrete timesteps according to a pre-defined variance schedule $\beta_1 < \beta_2 < \dots < \beta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$
$$q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

### 2.2 Closed-Form Marginal Distribution $q(x_t | x_0)$
A crucial property of Gaussian diffusion is that $x_t$ can be sampled directly at any arbitrary timestep $t$ in closed form without iterating through intermediate steps.
Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
$$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 2.3 Variance Schedules
1. **Linear Schedule (Ho et al., 2020)**:
   $$\beta_1 = 10^{-4}, \quad \beta_T = 0.02, \quad T = 1000$$
   Linearly spaced: $\beta_t = \beta_1 + \frac{t - 1}{T - 1} (\beta_T - \beta_1)$.
2. **Cosine Schedule (Nichol & Dhariwal, 2021)**:
   Prevents excessive noise destruction in small images:
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)^2, \quad s = 0.008$$
   $$\beta_t = \text{clip}\left( 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \text{max}=0.999 \right)$$

### 2.4 True Reverse Posterior $q(x_{t-1} | x_t, x_0)$
When conditioned on the original clean image $x_0$, the reverse step is tractable and Gaussian:
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)$$
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

### 2.5 Parametrization: Noise Prediction $\epsilon_\theta(x_t, t)$
Substituting $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$ into the posterior mean yields:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
Instead of predicting $x_0$ or the mean directly, the neural network is parameterized to predict the noise vector $\epsilon_\theta(x_t, t) \approx \epsilon$.

### 2.6 Simplified Training Objective ($L_{\text{simple}}$)
Ho et al. showed that discarding the complex variational weights produces substantially higher sample quality:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|_2^2 \right]$$

### 2.7 Reverse Sampling Algorithm (Algorithm 2 in Ho et al.)
To generate an image from pure noise:
1. Sample $x_T \sim \mathcal{N}(0, I)$.
2. For $t = T, T-1, \dots, 1$:
   - Sample $z \sim \mathcal{N}(0, I)$ if $t > 1$, else $z = 0$.
   - Compute:
     $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
     where $\sigma_t = \sqrt{\tilde{\beta}_t}$ (or $\sqrt{\beta_t}$).
3. Return clean sample $x_0$ clamped to $[-1, 1]$.

---

## 3. U-Net Architecture Blueprint for CIFAR-10 ($32 \times 32$)

```
                       U-NET CONDITIONAL DENOISING BACKBONE
  Input Noisy Image x_t [B, 3, 32, 32]
  Input Timestep t ∈ [1, 1000] ──► Sinusoidal Embedding [B, 128] ──► MLP ──► Time Embedding [B, 256]
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Downsampling Stage 1 (32 x 32)                             │
  │ • Conv2d(3, 64)                                            │
  │ • 2x ResNetBlock(64) + Time Injection                      │ ──┐ (Skip 1 to Decoder)
  │ • Downsample2d (Conv2d, k4, s2, p1) → [B, 64, 16, 16]      │   │
  └─────────────────────────────────────────────────────────────┘   │
          │                                                         │
          ▼                                                         │
  ┌─────────────────────────────────────────────────────────────┐   │
  │ Downsampling Stage 2 (16 x 16)                             │   │
  │ • 2x ResNetBlock(128) + Time Injection                     │ ──┼──┐ (Skip 2 to Decoder)
  │ • Spatial Self-Attention (128 channels, 4 heads)           │   │  │
  │ • Downsample2d (Conv2d, k4, s2, p1) → [B, 128, 8, 8]       │   │  │
  └─────────────────────────────────────────────────────────────┘   │  │
          │                                                         │  │
          ▼                                                         │  │
  ┌─────────────────────────────────────────────────────────────┐   │  │
  │ Downsampling Stage 3 (8 x 8)                               │   │  │
  │ • 2x ResNetBlock(256) + Time Injection                     │ ──┼──┼──┐ (Skip 3 to Decoder)
  │ • Spatial Self-Attention (256 channels, 4 heads)           │   │  │  │
  └─────────────────────────────────────────────────────────────┘   │  │  │
          │                                                         │  │  │
          ▼                                                         │  │  │
  ┌─────────────────────────────────────────────────────────────┐   │  │  │
  │ Middle Block / Bottleneck (8 x 8)                          │   │  │  │
  │ • ResNetBlock(256) + Time Injection                        │   │  │  │
  │ • Spatial Self-Attention (256 channels, 4 heads)           │   │  │  │
  │ • ResNetBlock(256) + Time Injection                        │   │  │  │
  └─────────────────────────────────────────────────────────────┘   │  │  │
          │                                                         │  │  │
          ▼                                                         │  │  │
  ┌─────────────────────────────────────────────────────────────┐   │  │  │
  │ Upsampling Stage 3 (8 x 8)                                 │   │  │  │
  │ • Concat(Skip 3) → [B, 512, 8, 8]                          │◄──┘  │  │
  │ • 2x ResNetBlock(256) + Spatial Self-Attention(256)        │      │  │
  │ • Upsample2d (ConvTranspose2d or Nearest+Conv) → [16, 16]   │      │  │
  └─────────────────────────────────────────────────────────────┘      │  │
          │                                                            │  │
          ▼                                                            │  │
  ┌─────────────────────────────────────────────────────────────┐      │  │
  │ Upsampling Stage 2 (16 x 16)                               │      │  │
  │ • Concat(Skip 2) → [B, 256, 16, 16]                        │◄─────┘  │
  │ • 2x ResNetBlock(128) + Spatial Self-Attention(128)        │         │
  │ • Upsample2d → [32, 32]                                    │         │
  └─────────────────────────────────────────────────────────────┘         │
          │                                                               │
          ▼                                                               │
  ┌─────────────────────────────────────────────────────────────┐         │
  │ Upsampling Stage 1 (32 x 32)                               │         │
  │ • Concat(Skip 1) → [B, 128, 32, 32]                        │◄────────┘
  │ • 2x ResNetBlock(64) + Time Injection                      │
  └─────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Output Head                                                 │
  │ • GroupNorm(32, 64) + SiLU + Conv2d(64, 3, k3, s1, p1)      │
  └─────────────────────────────────────────────────────────────┘
          │
          ▼
  Predicted Noise ε_θ(x_t, t) ∈ ℝ^{B x 3 x 32 x 32}
```

---

## 4. Hardware Feasibility on NVIDIA Quadro T2000 (4GB VRAM)

* **VRAM Ceiling**: 4.0 GB physical memory ($\approx 3.5\text{ GB}$ safe working limit).
* **Attention Token Sizing**:
  - Spatial attention at $32 \times 32$ requires computing attention matrices of size $1024 \times 1024$ per head per sample $\to$ high memory.
  - **Golden Rule**: Restrict self-attention blocks strictly to $16 \times 16$ ($N = 256$ tokens) and $8 \times 8$ ($N = 64$ tokens). Do **not** place attention at $32 \times 32$.
* **Batch Size & Gradient Accumulation**:
  - `batch_size: 64` fits comfortably on 4GB VRAM ($\approx 2.1\text{ GB}$ peak VRAM).
  - Use `gradient_accumulation_steps: 2` to achieve an effective batch size of $128$.
* **Exponential Moving Average (EMA)**:
  - DDPM generation quality depends heavily on an EMA of model weights ($\beta_{\text{EMA}} = 0.9999$). Maintain an EMA shadow model on CPU or evaluate with EMA weights during sampling.

---

## 5. Architectural Consistency with Hands-on VAE

To maintain a coherent multi-repo ecosystem, the structure of `Hands-on DDPM` mirrors `Hands-on VAE`:

```text
Hands-on DDPM/
├── configs/
│   ├── cifar10_baseline.yaml       # T=1000, linear schedule, UNet [64, 128, 256]
│   └── mnist_baseline.yaml         # T=1000, in_channels=1
├── docs/
│   ├── adr/                         # Architecture Decision Records
│   └── reports/
│       └── 001-baseline-ddpm-report.md # Formal DDPM Report (Deliverable a)
├── specs/
│   └── 001-create-ddpm/            # Specification, plan, contracts, and tasks
├── src/
│   ├── cli/
│   │   └── main.py                 # Typer CLI: verify, train, sample, denoise-strip, benchmark
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
├── pyproject.toml                  # Editable package: `pip install -e .`, cli: `ddpm`
└── README.md                       # Comprehensive setup and reproduction guide
```

---

## 6. Phased Implementation Roadmap for the Next Agent

When you start implementing with the coding agent, execute the project in these logical phases:

### Phase 1: Setup & Packaging
- Scaffold `pyproject.toml` with entry point `ddpm = "src.cli.main:app"`.
- Set up `src/utils/seeding.py` and `src/utils/logging.py`.
- Configure `configs/cifar10_baseline.yaml` ($T = 1000$, linear $\beta = [10^{-4}, 0.02]$, batch size 64).

### Phase 2: Foundational Diffusion Mathematics (`src/models/gaussian_diffusion.py`)
- Implement variance schedules: `linear_beta_schedule`, `cosine_beta_schedule`.
- Precompute diffusion constants: $\alpha_t$, $\bar{\alpha}_t$, $\sqrt{\bar{\alpha}_t}$, $\sqrt{1 - \bar{\alpha}_t}$, posterior variance $\tilde{\beta}_t$.
- Implement closed-form forward sampling: `q_sample(x_0, t, noise=None)`.
- Implement unit tests verifying forward noise convergence to pure Gaussian noise $\mathcal{N}(0, I)$ as $t \to T$.

### Phase 3: U-Net Subcomponents
- `src/models/embeddings.py`: Sinusoidal positional embeddings + 2-layer MLP with SiLU.
- `src/models/attention.py`: Multi-head spatial self-attention with GroupNorm for $16 \times 16$ and $8 \times 8$ resolutions.
- `src/models/resnet.py`: ResNet block injecting timestep embeddings via additive or scale-shift modulation.

### Phase 4: U-Net Assembly & Training Objective
- `src/models/unet.py`: Assembles downsampling stages, bottleneck, skip connections, and upsampling stages.
- `src/models/gaussian_diffusion.py`: Implement training loss $L_{\text{simple}}(\theta) = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$.
- Pre-training gate: `ddpm verify --config configs/cifar10_baseline.yaml` verifying gradient flow through all timesteps.

### Phase 5: Reverse Denoising Sampling Engine
- Implement Algorithm 2 in `gaussian_diffusion.py`: `p_sample` and `p_sample_loop`.
- Implement step-by-step trajectory capture: records frames at $t = 1000, 800, 600, 400, 200, 100, 50, 0$.
- `src/evaluation/visualizer.py`: Render horizontal reverse diffusion transition strips showing clean images emerging from pure static noise.

### Phase 6: Training Loop & EMA Checkpointing
- Implement `DDPMAgentTrainer` in `src/training/trainer.py`.
- Integrate Exponential Moving Average (EMA) with decay $0.9999$.
- Save checkpoints (`best_checkpoint.pt`, `final_checkpoint.pt`, `latest.pt`) with full metadata.

### Phase 7: Quantitative Benchmarking (FID & Inception Score)
- Adapt `src/evaluation/metrics.py` to evaluate 5,000 synthesized DDPM samples using Inception-v3.
- Target Benchmark: Compare against VAE Baseline (FID $169.02$) and Enhanced VAE (FID $181.00$). DDPM on CIFAR-10 typically achieves FID $< 50.0$.

### Phase 8: Deliverable Documentation & Technical Report
- Author `docs/reports/001-baseline-ddpm-report.md`.
- Include the final **Comparative Synthesis Table** answering the core mandate of `GenCV003`:
  - Detailed theoretical and empirical comparison between VAE and DDPM.
  - Why DDPM avoids blurriness (loss on noise vs. loss on pixels).
  - Sampling speed vs. sample realism trade-off (1 step vs. 1,000 steps).

---

## 7. Comparative Theoretical Table: VAE vs. DDPM (For Final Deliverable)

| Dimension | Variational Autoencoder (VAE) | Denoising Diffusion (DDPM) |
| :--- | :--- | :--- |
| **Generative Paradigm** | Single-step latent variable projection | Multi-step iterative stochastic denoising |
| **Latent Space** | Compressed low-dimensional bottleneck ($d=32, 128$) | Full observation dimension ($3 \times 32 \times 32 = 3072$) |
| **Target of Loss** | Placed directly on **reconstructed pixels** $\|x - \hat{x}\|^2$ | Placed on **predicted noise vector** $\|\epsilon - \epsilon_\theta\|^2$ |
| **Why Blurry vs. Sharp** | **Blurry**: $L_2$ conditional mean averaging over ambiguous edges | **Sharp**: Iterative local denoising reverses noise step-by-step |
| **Inference Speed** | **Fast** (Single forward pass: $\sim 1\text{ ms}$) | **Slow** (1,000 sequential U-Net forward passes: $\sim 2\text{ s}$) |
| **Sampling Mechanism** | Single evaluation: $\hat{x} = \text{Decoder}(z), z \sim \mathcal{N}(0, I)$ | Markov chain walk: $x_T \to x_{T-1} \to \dots \to x_0$ |
| **Training Stability** | Can suffer from posterior collapse or variance inflation | Extremely stable MSE regression on noise vector $\epsilon$ |
| **FID Score on CIFAR-10**| Baseline: **$169.02$**, Enhanced: **$181.00$** | Typical DDPM: **$<50.0$** (Vastly sharper and photorealistic) |

---

## 8. Resume Prompt for the Next Session / Agent

When you start the DDPM session in `/home/amousa1/Projects/Hands-on DDPM`, provide this exact prompt:

> *"We are starting the Hands-on DDPM project in `/home/amousa1/Projects/Hands-on DDPM` to complete Part 2 of `GenCV003`. Please read `HANDOFF.md` completely. Notice the mathematical formulation, U-Net architecture blueprint, and the phased roadmap. Proceed directly to executing **Phase 1 (Setup & Packaging)** and **Phase 2 (Foundational Diffusion Mathematics)**."*
