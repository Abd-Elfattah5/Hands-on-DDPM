# Architecture Deep Dive: Top-Down Technical & Mathematical Rationale

This document gives the top-down technical rationale for every architectural decision, hyperparameter and mathematical formulation chosen for the baseline Denoising Diffusion Probabilistic Model (DDPM). It is the sibling of `Hands-on VAE/ARCHITECTURE_DEEP_DIVE.md`, which ends with the motivation for this project (§13 there: "The Transition to DDPM").

> **Status**: Final. Design decisions, measured hardware numbers and empirical results (§11) are complete; full analysis in `docs/reports/001-baseline-ddpm-report.md`.

---

## 1. Why DDPM After the VAE?

The VAE project showed empirically and theoretically that single-stage Gaussian VAEs produce soft images regardless of capacity (d = 512, 8.8 M parameters, full covariance and GMM priors all stayed blurry):

- **The $L_2$ conditional-mean trap**: A pixel-space Gaussian likelihood makes the optimal decoder output $\mathbb{E}[x|z]$. When several sharp images are plausible for one $z$, the decoder outputs their average, which is a blur.
- **One-shot generation**: The decoder must turn a 32-number vector into 3,072 pixels in a single pass.

DDPM changes both:

1. **Generation becomes 1,000 small steps.** Each step only has to remove a little noise, a much easier local task than producing a whole image at once.
2. **The loss is placed on the noise, not on the pixels.** The network predicts the Gaussian noise $\epsilon$ that was added. The target is a single well-defined value for each $(x_0, \epsilon, t)$, so there is no averaging across alternative sharp images within one step. Sharpness emerges from the chain of stochastic steps.

---

## 2. Dataset Selection & Preprocessing: CIFAR-10

### 2.1 Why CIFAR-10 ($32 \times 32 \times 3$)?
- **Parity with the VAE**: The same dataset, splits and preprocessing make FID/IS directly comparable with the VAE results (FID 169.02 baseline, 181.00 enhanced).
- **The standard DDPM benchmark**: Ho et al. (2020) report CIFAR-10 results, so the architecture and hyperparameters have a well-tested reference point.
- **Hardware fit**: At $32 \times 32$ the full Ho-style U-Net trains within the 3 GB budget of the local Quadro T2000 (measured, §9).

### 2.2 Normalization to $[-1, 1]$
- The forward process mixes the image with $\mathcal{N}(0, I)$ noise. Centering the data at 0 with a scale comparable to unit variance means $x_T$ really is indistinguishable from pure noise when $\bar{\alpha}_T \approx 0$.
- Generated images are clamped to $[-1, 1]$ at the end of sampling, then mapped to $[0, 1]$ for saving and for Inception.

### 2.3 Splits and Augmentation
- 45,000 train / 5,000 validation (seeded `random_split`, as in the VAE) / 10,000 test.
- The validation loss (EMA weights) selects `best_checkpoint.pt`; the test split supplies the 5,000 real images for FID.
- Random horizontal flip is the only augmentation, matching Ho et al. on CIFAR-10. It doubles effective data diversity without changing image statistics.

---

## 3. The Forward Diffusion Process

### 3.1 Markov Chain Definition
$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big), \qquad t = 1, \dots, T,\ T = 1000$$
The process has no learnable parameters. It is fully defined by the noise schedule $\beta_1, \dots, \beta_T$.

### 3.2 Closed-Form Marginal (Why Training Is Cheap)
With $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:
$$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t) I\big) \;\Rightarrow\; x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$$
Any timestep can be reached in **one** operation, so every training example uses a random $t$ without simulating the chain. This is why a 1,000-step model trains at the cost of a normal image-to-image network.

### 3.3 Noise Schedules
| Schedule | Definition | Why |
|---|---|---|
| `linear` (default) | $\beta_t$ linear from $10^{-4}$ to $0.02$ | The Ho et al. CIFAR-10 setting; reproduces the reference results |
| `cosine` | $\bar{\alpha}_t = f(t)/f(0)$, $f(t) = \cos^2\!\big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\big)$, $s = 0.008$, $\beta_t \le 0.999$ | Destroys information more gradually at small $t$ (Nichol & Dhariwal); available for comparison |

- All schedule quantities are precomputed once in float64 (avoiding cumulative-product drift over 1,000 multiplications) and stored as float32 buffers.
- The cosine $\beta_t$ is clipped at 0.999 so the last steps never divide by values near zero.
- `ddpm verify` checks that $\bar{\alpha}_t$ is strictly decreasing and that $\bar{\alpha}_T < 10^{-3}$, so $x_T$ is effectively pure noise.

### 3.4 True Reverse Posterior
Given $x_0$, the reverse step is a Gaussian with closed-form parameters:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\tilde{\mu}_t,\ \tilde{\beta}_t I\big), \quad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$
The learned reverse process approximates this posterior, and $\tilde{\beta}_t$ is one of the two variance options in sampling (§6.2).

---

## 4. Training Objective

### 4.1 Noise-Prediction Parametrization
The network predicts $\epsilon_\theta(x_t, t)$. The reverse mean is then
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)$$
Predicting $\epsilon$ (instead of $x_0$ or $\tilde{\mu}_t$) gives a target with the same unit scale at every $t$. Ho et al. found this gives the best sample quality.

### 4.2 Simplified Objective $L_{\text{simple}}$
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{1,T\},\ x_0,\ \epsilon \sim \mathcal{N}(0,I)}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\ t\big)\big\|^2\Big]$$
- This is the variational bound with its per-timestep weights removed. Dropping the weights puts relatively more emphasis on the harder, noisier timesteps, which improves sample quality.
- **Contrast with the VAE**: The VAE loss is an MSE on *pixels* (which leads to the conditional-mean blur). Here the MSE is on *noise*, which the network can predict without averaging over alternative images.

### 4.3 Optimization Settings
| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW, lr $2\times10^{-4}$, weight decay 0 | Ho et al. CIFAR-10 setting |
| Warmup | Linear over 5,000 optimizer steps, then constant | Prevents early instability at effective batch 128 |
| Gradient clipping | Norm 1.0 | Ho et al.; guards against rare large-loss batches at small $t$ |
| Effective batch | 128 = 32 × 4 accumulation (local), 128 × 1 (Colab) | Same optimization on both machines (§9) |
| Precision | fp32 default; bf16/fp16 opt-in | fp16 was 4× slower and produced NaN on the T2000 (measured) |
| Epochs | 100 (≈ 35k optimizer steps) | Fits the time budget; increase if FID has not converged, record it in the report |

### 4.4 EMA Weights
After every optimizer step:
$$\theta_{\text{EMA}} \leftarrow 0.9999\,\theta_{\text{EMA}} + 0.0001\,\theta$$
- This averages over roughly the last 10,000 steps, smoothing out batch-to-batch jitter in the weights.
- Sampling runs the network 1,000 times in a row, so small weight errors compound. EMA weights produce noticeably cleaner samples and lower FID.
- EMA weights are used for validation (to pick `best_checkpoint.pt`), sampling and benchmarking. Raw weights remain selectable with `--weights raw`.

---

## 5. Denoising Network: Time-Conditioned U-Net

### 5.1 Overall Topology
```text
x_t [B,3,32,32] ─► Conv3x3(3→64)
  Level 32×32: 2× ResBlock(64)                          ──skips──┐
  Down (stride-2 conv) → 16×16
  Level 16×16: 2× [ResBlock(128) + Attention(128, 4 heads)] ──skips──┤
  Down → 8×8
  Level 8×8:   2× [ResBlock(256) + Attention(256, 4 heads)] ──skips──┤
  Middle 8×8:  ResBlock(256) → Attention(256) → ResBlock(256)        │
  Up path: 3 ResBlocks per level (one per skip), attention at 8×8 and 16×16,
           nearest-neighbor ×2 + Conv3x3 upsampling                ◄┘
  Output: GroupNorm(32) → SiLU → Conv3x3(64→3)  ═► ε_θ(x_t, t) [B,3,32,32]
t ─► sinusoidal(64) ─► Linear(64→256) ─► SiLU ─► Linear(256→256) ─► injected into every ResBlock
```
Measured size: **16.06 M parameters**.

### 5.2 Why a U-Net?
- The output (noise) has the same shape as the input, and good noise prediction needs both global context (what object is this?) and pixel-exact detail.
- The encoder–decoder path provides context at 8×8, and the skip connections carry full-resolution detail directly to the output.

### 5.3 Channel Widths $[64, 128, 256]$
- Widths double each time resolution halves, keeping the information capacity per level roughly constant.
- This is a reduced version of Ho et al.'s CIFAR-10 model ($[128, 256, 256, 256]$, 35.7 M parameters), sized to train within the 3 GB budget while keeping the same structure.

### 5.4 Residual Blocks and Time Injection
`GroupNorm → SiLU → Conv3x3 → (+ Linear(time embedding)) → GroupNorm → SiLU → Dropout(0.1) → Conv3x3`, with a 1×1 skip conv when channel counts change.
- **Time injection** adds a per-channel bias computed from the timestep embedding, so the same weights behave differently at different noise levels: heavy denoising at $t \approx 1000$, fine detail at $t \approx 1$.
- **Zero-initialized** last conv in each block and in the output head: the network starts close to an identity mapping, which stabilizes early training.
- **Dropout 0.1**, as in Ho et al. for CIFAR-10, reduces overfitting on 45k images.

### 5.5 Why GroupNorm (32 groups) Instead of BatchNorm?
- Normalization is per sample, so results don't depend on batch composition or size.
- This is essential here: the network is trained with micro-batches of 32 (accumulated to 128) and sampled with batches of 256. BatchNorm statistics would differ across these, and also across the very different noise levels in one batch.

### 5.6 Self-Attention at 16×16 and 8×8 Only
- Convolutions see local neighborhoods; attention lets every position use every other position, which helps global coherence (object shape, symmetric layout).
- **Memory rule**: Attention cost grows with the square of the token count. 32×32 = 1,024 tokens gives a 1,024 × 1,024 matrix per head per image, which is too expensive for 4 GB. 16×16 (256 tokens) and 8×8 (64 tokens) are affordable, and Ho et al. also use 16×16.
- **Implementation**: The block (GroupNorm, 1×1 projections to Q/K/V, 4 heads, output projection, residual) is written from primitives, and the softmax product uses PyTorch's fused `scaled_dot_product_attention` kernel. Measured: ~170 MiB less memory and ~6% faster than an explicit einsum. A unit test checks it against the einsum reference.

### 5.7 Down/Upsampling
- Downsampling with a stride-2 3×3 conv is learnable and keeps detail better than pooling.
- Upsampling with nearest-neighbor ×2 followed by a 3×3 conv avoids the checkerboard artifacts that transposed convolutions often produce, which would be amplified over 1,000 sampling steps.

### 5.8 Sinusoidal Timestep Embedding
$\text{emb}(t)_{2i} = \sin(t / 10000^{2i/d})$, $\text{emb}(t)_{2i+1} = \cos(t / 10000^{2i/d})$, then a 2-layer MLP with SiLU.
- Gives each timestep a unique, smooth code (nearby timesteps get similar codes), so the network can interpolate between noise levels instead of memorizing 1,000 separate cases.

---

## 6. Reverse Sampling (Algorithm 2)

### 6.1 Algorithm
```text
x_T ~ N(0, I)
for t = T, …, 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
return clamp(x_0, −1, 1)
```

### 6.2 Reverse Variance $\sigma_t^2$
| Option | Value | Notes |
|---|---|---|
| `fixed_large` (default) | $\beta_t$ | Ho et al. found it slightly better on CIFAR-10 |
| `fixed_small` | $\tilde{\beta}_t$ | Exact posterior variance; available for comparison |
Learned variances (Nichol & Dhariwal) are out of scope for this feature.

### 6.3 Determinism and Cost
- Sampling uses a dedicated random generator seeded from `--seed`, so the same checkpoint and seed give identical images on the same hardware.
- Cost: 1,000 network passes per batch. Measured on the T2000: ~500 s per batch of 256, so ~2.8 h for the 5,000 benchmark images. Finished batches are saved immediately, so an interrupted benchmark can be resumed.

### 6.4 Denoising Strip
- Frames are captured at $t = 1000, 800, 600, 400, 200, 100, 50, 0$ and shown as a strip, illustrating that coarse structure appears first (around $t \approx 400$–$200$) and fine detail last.
- Label rule: $t = 1000$ is the initial noise $x_T$ before any reverse step; label $k < 1000$ is the image after the reverse step at zero-based index $k$, so $t = 0$ is the final clamped output.

---

## 7. Evaluation Design

### 7.1 Shared Benchmark Protocol (Parity with the VAE)
- The VAE's `metrics.py` is ported with unchanged maths: torchvision Inception-v3 (`DEFAULT` weights), bilinear resize to 299, inputs rescaled to $[-1, 1]$, 2048-d pool features, 1000-way softmax.
- **FID**: The first 5,000 CIFAR-10 test images (real) vs. 5,000 generated images, using the mean and covariance of the features and `scipy.linalg.sqrtm`.
- **IS**: The 5,000 generated images split into 10 groups of 500; reported as mean ± std.
- These numbers are comparable with the VAE but not with published 50k-sample FIDs (5k samples give a slightly higher, noisier FID).

### 7.2 Memorization Check (Nearest-Neighbor Panel)
- `sample --nearest` finds, for each of the first 64 generated images (`--nearest-rows`), the 3 closest training images by pixel L2 distance and shows them side by side.
- Near-identical pairs indicate memorization; similar-but-different images indicate generalization. A low FID alone can't distinguish the two.

### 7.3 Computational Complexity and Distribution Coverage (Required in the Report)
- The constitution requires the report to cover both.
- **Computational complexity**: parameter count, 1,000 network evaluations per generated image (vs. 1 for the VAE), measured seconds per image and per 5,000 images, training time and peak GPU memory.
- **Distribution coverage**: `ddpm benchmark` stores a `class_coverage` block (number of distinct Inception top-1 classes, entropy of the marginal class distribution, the 20 most frequent classes) computed from the same probabilities used for IS. The report combines it with a visual check that all 10 CIFAR-10 object types appear and with the nearest-neighbor findings.

### 7.4 Test Loss
- $L_{\text{simple}}$ on the test split with fixed-seed timesteps and noise, so the value is repeatable between runs.

---

## 8. Software Architecture

### 8.1 Decorator Registry (as in the VAE)
- `@register_schedule("linear" | "cosine")`, `@register_denoiser("unet")` and `@register_diffusion("gaussian")`.
- `build_diffusion_from_config` validates the YAML, builds schedule → denoiser → diffusion and returns the model.
- Adding a new schedule or network later (e.g., an enhanced feature) requires no change to the training or CLI code.

### 8.2 Typed Output Contracts
- `DiffusionOutput` (x_t, t, noise, predicted_noise), `LossOutput` and `SamplingOutput` (samples, trajectory, timing).
- These replace loose tuples, as `VAEOutput` did in the VAE.

### 8.3 Checkpoints and Resume
- `latest.pt` every epoch, `best_checkpoint.pt` on EMA validation improvement, `final_checkpoint.pt` at the end.
- Each contains raw and EMA weights, optimizer, scheduler, history and RNG states.
- `train --resume` continues exactly where training stopped. Unlike the VAE, the metric history is restored too.

### 8.4 One Code Path, Two Machines
- Local training and Colab training run the **same** `ddpm train` command.
- The Colab notebook only mounts Google Drive, installs the package and calls the CLI with `--output-dir` on Drive (resuming automatically after disconnects).
- No model or training code lives in the notebook, which keeps the constitution's rule against notebook-only training loops.
- The two configs differ only in per-step batch, accumulation steps and data workers; model, diffusion and optimization settings and the effective batch of 128 are identical.

### 8.5 Recorded Deviations from Hands-on VAE
- Every place where this repository intentionally differs from `Hands-on VAE` (no MNIST, two configs, tracked Colab notebook, JSON only, manual VAE comparison, EMA-based checkpoint selection, index-based split identical to the VAE's, history-restoring resume, published checkpoint) is justified in `docs/adr/0001-deviations-from-hands-on-vae-conventions.md`, as the constitution requires.

---

## 9. Hardware Budget (Measured, Not Assumed)

Measured on the NVIDIA Quadro T2000 (4 GB, budget 3,072 MiB per process), full architecture, fresh process per scenario (`specs/001-create-ddpm/research/gpu_memory_results.md`):

| Workload | Configuration | Total GPU memory | Speed |
|---|---|---|---|
| Training (default) | 32 × 4 accumulation, fp32, AdamW + EMA on GPU | 1,625 MiB | 0.84 s per optimizer step; ~6 min per epoch incl. validation (~10 h / 100 epochs, measured smoke run) |
| Training | 64 × 2 | 2,925 MiB | 0.87 s per step (too close to the budget) |
| Training | 96 × 1 / 128 × 1 | 3,915 MiB / > 4 GB | Does not fit |
| Sampling | batch 256 | 2,061 MiB | ~2.8 h for 5,000 images |
| Inception (FID/IS) | batch 64 | 2,269 MiB | — |

Training speed is limited by compute (TFLOPS) and memory bandwidth, not by memory size. 32 × 4 and 64 × 2 take the same time per update on the T2000. The official training run therefore used a Colab T4 with 128 × 1, which gives the same effective batch. **Measured on the T4: 236 s per epoch (≈ 1.5× faster than the T2000's ~6 min), 5.25 h for 80 epochs, 5.8 GB peak.**

---

## 10. VAE vs. DDPM: Design-Level Comparison

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              GENERATIVE MODEL PARADIGM SHIFT                           │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Variational Autoencoder (VAE)  │ Denoising Diffusion Probabilistic Model (DDPM)        │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ • Latent bottleneck z          │ • No bottleneck: x_t has full image dimension (3,072) │
│   (d = 32 / 128)               │                                                       │
│ • Learned encoder + decoder    │ • Fixed forward process + one learned denoiser        │
│ • Loss on pixels (ELBO, MSE)   │ • Loss on noise (L_simple)                            │
│ • 1 network pass per image     │ • 1,000 network passes per image                      │
│ • ~1.6–2.3 M parameters        │ • 16.06 M parameters                                  │
│ • Risk: posterior collapse,    │ • Stable MSE regression; risk: slow sampling,         │
│   blur                         │   memorization (checked with the NN panel)            │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

The quantitative comparison (FID, IS, seconds per image, training time) is added to `docs/reports/001-baseline-ddpm-report.md` once results exist.

---

## 11. Results (80 epochs, raw weights)

| Metric | DDPM | VAE baseline / enhanced |
|---|---|---|
| FID ↓ (5,000 vs 5,000) | **39.69** | 169.02 / 181.00 |
| Inception Score ↑ | **5.18 ± 0.14** | 2.11 / 1.68 |
| Test $L_{\text{simple}}$ | 0.0309 | — |
| Sampling | 2.16 s/image (T2000, batch 256) | 1 pass |
| Class coverage | 322 Inception classes, entropy 5.85 nats, max share 5.2% | — |
| Memorization | none (nearest training L2 ≥ 4.75) | — |

Observations that refine the design discussion above:

1. **The EMA trade-off (§4.4) depends on training length.** At 28,080 steps, EMA 0.9999 still retains
   $0.9999^{28080}\approx6\%$ of the random initialization. Its single-step loss looked fine (0.0390), but over 1,000
   reverse steps the error compounded and 57% of sample pixels saturated. The raw weights (1.7% saturation) were used for
   all results (ADR 0001 D11). For short runs, a smaller decay (e.g. 0.999) or a warm-up of the decay would be preferable.
2. **Coarse-to-fine generation (§6.4) is confirmed.** Frames stay noise-like down to $t\approx400$
   ($\sqrt{\bar\alpha_t}=0.44$); layout and colour appear near $t\approx200$ (0.81); the last 50 steps add edges and texture.
3. **Noise prediction removes the VAE's blur (§1, §4.2).** At similar 32×32 resolution, objects have sharp silhouettes and
   consistent backgrounds; the remaining failures are abstract or ambiguous samples rather than averaged ones.
4. **The cost is in sampling, not training stability.** The raw loss converged within ~15 epochs without any instability,
   but generation needs 1,000 network evaluations per image.

