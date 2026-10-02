# Generative Modeling & Benchmarking Context

The domain of mathematical from-scratch generative vision modeling (Denoising Diffusion Probabilistic Models), deterministic benchmarking, and distribution evaluation. Sibling glossary to `Hands-on VAE/CONTEXT.md`.

## Language

### Core Diffusion Structures

**Forward Diffusion Process ($q(x_t|x_{t-1})$)**:
The fixed, non-learned Markov chain that gradually adds Gaussian noise to a clean image over $T$ steps until it becomes pure noise.
_Avoid_: Encoder, noising network

**Reverse Denoising Process ($p_\theta(x_{t-1}|x_t)$)**:
The learned Markov chain that removes noise one step at a time, turning pure Gaussian noise into an image.
_Avoid_: Decoder, generator network

**Timestep ($t$)**:
The position in the diffusion chain. Paper notation runs $t = 1 \dots T$ ($T = 1000$); code uses zero-based indices $0 \dots T-1$, and $t = 0$ in visualizations denotes the final clean image.
_Avoid_: Epoch, iteration, step count

**Noise Schedule ($\beta_t$)**:
The ordered sequence of per-step noise variances that defines how fast the forward process destroys the signal; `linear` (Ho et al.) or `cosine` (Nichol & Dhariwal).
_Avoid_: Learning-rate schedule, noise level

**Cumulative Signal Level ($\bar{\alpha}_t$)**:
The product $\prod_{s=1}^{t}(1-\beta_s)$; the fraction of the original image's variance that survives at step $t$. It decreases from ≈ 1 to ≈ 0.
_Avoid_: Alpha, signal ratio

**Closed-Form Forward Marginal ($q(x_t|x_0)$)**:
Direct one-shot noising of a clean image to any step: $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$.
_Avoid_: Iterative noising, forward pass

**True Reverse Posterior ($q(x_{t-1}|x_t, x_0)$)**:
The exact Gaussian distribution of the previous step given the current noisy image and the clean image; its variance is the posterior variance $\tilde{\beta}_t$.
_Avoid_: Reverse process, denoiser output

**Denoising Network ($\epsilon_\theta(x_t, t)$)**:
The time-conditioned U-Net that receives a noisy image and its timestep and predicts the noise that was added.
_Avoid_: Generator, decoder, score model

**Timestep Embedding**:
A sinusoidal encoding of $t$ passed through a small MLP and injected into every residual block so that one network handles all noise levels.
_Avoid_: Positional encoding (outside the attention context), time feature

**Group Normalization**:
A channel-group normalization applied independently per sample, so the network behaves identically for any batch size during training, accumulation and sampling.
_Avoid_: Batch normalization, layer scaling

### Architecture & Extension Patterns

**Component Registry**:
A decoupled factory pattern binding string identifiers in configuration schemas to modular schedules, denoisers and diffusion processes.
_Avoid_: Hardcoded switch, dynamic loader

**DiffusionOutput**:
A strongly-typed container holding the noised input, sampled timesteps, target noise and predicted noise emitted by a training forward pass.
_Avoid_: Output tuple, model prediction

**EMA Weights**:
An exponential moving average copy of the network weights (decay 0.9999) updated after every optimizer step and used for sampling and evaluation.
_Avoid_: Averaged model, smoothed checkpoint, shadow model (in prose)

**Effective Batch**:
The number of images contributing to one optimizer update: per-step batch × gradient-accumulation steps (32 × 4 = 128 locally, 128 × 1 on Colab).
_Avoid_: Batch size (when the per-step micro-batch is meant)

**Gradient Accumulation**:
Summing gradients over several per-step batches before one optimizer update, trading time for GPU memory.
_Avoid_: Micro-batching (as a synonym for the whole technique)

**Checkpoint Convention**:
`latest.pt` every epoch, `best_checkpoint.pt` on EMA validation-loss improvement, `final_checkpoint.pt` at the end; any of them can be passed to `train --resume`.
_Avoid_: Snapshot, save state

### Mathematical Operations & Objectives

**Noise-Prediction Parametrization ($\epsilon$-prediction)**:
Training the network to predict the injected noise $\epsilon$ instead of the clean image or the posterior mean.
_Avoid_: Denoising objective, reconstruction

**Simplified Objective ($L_{\text{simple}}$)**:
The mean squared error $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ with $t$ sampled uniformly; the reweighted variational bound used by Ho et al.
_Avoid_: ELBO, reconstruction loss, total loss

**Ancestral Sampling (Algorithm 2)**:
Generation by starting from $x_T \sim \mathcal{N}(0, I)$ and applying $T$ reverse steps, adding fresh noise $\sigma_t z$ at every step except the last.
_Avoid_: Inference, decoding, DDIM

**Reverse Variance ($\sigma_t^2$)**:
The noise variance added in each reverse step: `fixed_large` ($\beta_t$, default) or `fixed_small` ($\tilde{\beta}_t$).
_Avoid_: Learned variance (out of scope)

**Denoising Trajectory**:
The intermediate images of one reverse chain captured at selected timesteps (1000, 800, 600, 400, 200, 100, 50, 0) and rendered as a denoising strip.
_Avoid_: Interpolation, latent traversal

### Evaluation & Benchmarking

**Fréchet Inception Distance (FID)**:
The distance between the Gaussian statistics (mean, covariance) of Inception-v3 2048-d features for real and generated images; lower is better.
_Avoid_: Accuracy, quality score

**Inception Score (IS)**:
$\exp(\mathbb{E}_x[\mathrm{KL}(p(y|x)\,\|\,p(y))])$ over Inception-v3 class probabilities of generated images, reported as mean ± std over 10 splits; higher is better.
_Avoid_: Classification accuracy

**Shared Benchmark Protocol**:
5,000 generated images vs. the first 5,000 CIFAR-10 test images, torchvision Inception-v3 weights, bilinear resize to 299, IS over 10 splits of 500; identical to `Hands-on VAE`, so the numbers are comparable across the two repositories (not with 50k-sample published FIDs).
_Avoid_: Standard FID, official FID

**Nearest-Neighbor Panel**:
A memorization check (`sample --nearest`) showing each generated image next to its closest CIFAR-10 training images by pixel L2 distance.
_Avoid_: Retrieval, similarity search

**Memorization**:
A failure mode where generated images are near-copies of training images rather than novel samples.
_Avoid_: Overfitting (when referring to samples), mode collapse
