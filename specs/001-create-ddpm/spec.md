# Feature Specification: From-Scratch Denoising Diffusion Probabilistic Model (DDPM)

**Feature Branch**: `001-create-ddpm`

**Created**: 2026-10-02

**Status**: Ready for Planning

**Input**: User description: "Implement a from-scratch DDPM (Ho et al. 2020) on CIFAR-10: linear/cosine
variance schedules with T=1000, time-conditioned U-Net [64,128,256] with attention at 16×16 and 8×8,
the simplified noise-prediction loss, EMA training, Algorithm 2 sampling, FID/IS evaluation matching
the VAE protocol, and a CLI (verify, train, evaluate, sample, denoise-strip, benchmark)." Scoped
against assignment `GenCV003` (Part 2: DDPM) and the sibling `Hands-on VAE` project.

### Assignment Traceability (GenCV003)

| GenCV003 Requirement | Covered By |
|----------------------|------------|
| Implement DDPM from the ground up, without pre-existing full implementations | User Stories 1–3, FR-001–FR-012 |
| Train on a publicly available dataset | User Story 2, FR-013–FR-018 (CIFAR-10) |
| Quantitative analysis with appropriate metrics (e.g., FID, Inception Score) | User Story 4, FR-024–FR-027 |
| Qualitative analysis: diversity, realism, thematic consistency | User Stories 3–4, FR-019–FR-023, FR-028 |
| Comprehensive comparison of the two models (VAE vs. DDPM) | User Story 5, FR-027–FR-030 |
| Deliverable a: detailed report (implementation, explanations, results, conclusions, main difference) | User Story 5, FR-029–FR-030 |
| Deliverable b: GitHub repository with README explaining how to reproduce results | User Story 5, FR-031 |

## Clarifications

### Session 2026-10-02

- Q: What is the GPU memory ceiling for every workflow? → A: 3 GB (3072 MiB) total GPU memory per
  process, as reported by the GPU driver, verified by measurement rather than estimate. Probe
  measurements of the full architecture are in `research/gpu_memory_results.md`.
- Q: Which per-step training batch should the baseline use within the 3 GB budget? → A: 32 images
  × 4 gradient-accumulation steps (effective batch 128; measured 1,625 MiB, 0.84 s per update).
- Q: How is a crash mid-training recovered? → A: Same convention as `Hands-on VAE`: `latest.pt`
  every epoch, `best_checkpoint.pt` on validation improvement, `final_checkpoint.pt` at the end,
  and `train --resume <checkpoint>` to continue.
- Q: Where does the official training run for reported results happen? → A: Verify locally; train
  on a free Colab T4 with the same config and seed, checkpoints on Google Drive, and `--resume`
  across disconnects; fp32 by default, mixed precision optional.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify the Diffusion Model Before Training (Priority: P1)

A researcher wants to confirm that the hand-written diffusion mathematics and the denoising network
are correct before spending hours of GPU time. They run a single verification command against the
baseline configuration and get a pass/fail report on shapes, schedule correctness, forward-noising
behavior, gradient flow, numerical stability, and memory fit on the local 4 GB GPU.

**Why this priority**: The constitution's Pre-Training Gate makes verification a hard prerequisite
for training. A silent bug in the noise schedule or time conditioning wastes an entire training run.

**Independent Test**: Run the verify command on the baseline configuration with no trained
checkpoint present; it completes and reports every check as passed or failed with a non-zero exit
code on any failure.

**Acceptance Scenarios**:

1. **Given** the baseline configuration (T = 1000, linear schedule β from 1e-4 to 0.02), **When**
   verification runs, **Then** it confirms the cumulative signal level ᾱ_t decreases strictly
   monotonically and that ᾱ_T is close to zero (forward process ends in near-pure Gaussian noise).
2. **Given** a batch of noisy images and random timesteps, **When** the denoising network performs a
   forward pass, **Then** the predicted noise has exactly the same shape as the input image batch
   (B × 3 × 32 × 32).
3. **Given** the simplified noise-prediction loss on a batch, **When** a backward pass runs, **Then**
   every trainable parameter receives a finite, non-zero gradient.
4. **Given** the cosine schedule is selected instead of linear, **When** verification runs, **Then**
   the same checks pass for the cosine schedule.
5. **Given** the baseline batch size, **When** one training step runs on the local GPU, **Then**
   the measured total GPU memory of the process stays at or below 3 GB (3072 MiB) and the
   measured value is reported.

---

### User Story 2 - Train the Baseline DDPM on CIFAR-10 (Priority: P1)

A researcher wants to train the baseline DDPM on CIFAR-10 from a single configuration file. Training
teaches the network to predict the noise that was added to a clean image at a randomly chosen
timestep. An exponential moving average (EMA) of the weights is maintained for generation. Losses,
learning rate, and periodic sample grids are recorded every epoch, and training can resume after an
interruption.

**Why this priority**: Training produces the model that every later story (sampling, evaluation,
comparison) depends on, and it satisfies GenCV003's "train on a public dataset" requirement.

**Independent Test**: Train for a short run (e.g., 2 epochs) on the baseline configuration and
confirm that the training loss decreases, per-epoch metrics are written to machine-readable files,
and checkpoints containing both raw and EMA weights are saved and reloadable.

**Acceptance Scenarios**:

1. **Given** CIFAR-10 images scaled to [-1, 1] and the baseline configuration, **When** training
   runs, **Then** each step samples a timestep uniformly from 1..T, noises the image in closed form,
   and minimizes the mean squared error between the true and predicted noise.
2. **Given** training is in progress, **When** each epoch ends, **Then** training loss, validation
   loss, learning rate, and elapsed time are appended to a machine-readable metrics log and an EMA
   sample grid is saved at the configured interval.
3. **Given** training reaches a checkpoint interval or finishes, **When** a checkpoint is saved,
   **Then** it contains the raw weights, EMA weights, optimizer state, epoch, seed, and full resolved
   configuration (latest, best-by-validation-loss, and final checkpoints are kept).
4. **Given** an interrupted run, **When** training is restarted with `--resume` pointing at
   `latest.pt` (or `best_checkpoint.pt`), **Then** it continues from the next epoch with identical
   optimizer and EMA state, losing at most one epoch of work.
5. **Given** two training runs with the same seed and configuration on the same hardware, **When**
   both complete the same number of steps, **Then** their recorded loss curves match.

---

### User Story 3 - Generate Images and Visualize the Reverse Process (Priority: P2)

A researcher wants to create new images by starting from pure random noise and repeatedly removing
noise over 1,000 steps (Ho et al. Algorithm 2) using the EMA weights. They also want a "denoising
strip" that shows the same image at selected steps from pure noise to the final clean image, to
explain how diffusion works in the report.

**Why this priority**: Sampling is the generative output that is evaluated and shown in the
qualitative analysis; the denoising strip is the main explanatory visual for the report.

**Independent Test**: Load a trained (or briefly trained) checkpoint, generate a fixed-seed 8 × 8
sample grid and a denoising strip, and confirm the artifacts exist, have valid pixel ranges, and are
identical across two runs with the same seed.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint, **When** sampling runs with seed S and count 64, **Then** an 8 × 8
   image grid and the individual images are saved to the samples artifact directory.
2. **Given** the same seed S on two consecutive runs, **When** sampling completes, **Then** the
   generated images are identical.
3. **Given** a trained checkpoint, **When** the denoising-strip command runs, **Then** it saves a
   horizontal strip per image showing frames at t = 1000, 800, 600, 400, 200, 100, 50, and 0.
4. **Given** sampling completes, **When** outputs are inspected, **Then** all pixel values lie in
   the valid image range and no output contains invalid numeric values.
5. **Given** a sampling run, **When** it completes, **Then** the total and per-image generation time
   are recorded in a machine-readable file.

---

### User Story 4 - Evaluate the DDPM Quantitatively and Qualitatively (Priority: P2)

A researcher wants to score the trained DDPM with the same metrics and protocol used for the VAE so
that the numbers are directly comparable. They also want qualitative artifacts that support a
written assessment of realism, diversity, and thematic consistency.

**Why this priority**: GenCV003 explicitly requires both quantitative metrics (FID, Inception Score)
and qualitative analysis; protocol parity with the VAE is what makes the comparison valid.

**Independent Test**: Run evaluation on a checkpoint and confirm a timestamped metrics summary is
written containing FID, Inception Score (mean and standard deviation), test noise-prediction loss,
sample count, and generation time, computed with the VAE's protocol.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint and the CIFAR-10 test split, **When** evaluation runs, **Then** it
   reports the average noise-prediction loss on the test split.
2. **Given** the EMA weights, **When** benchmarking runs, **Then** it computes FID and Inception
   Score over 5,000 generated images against 5,000 real CIFAR-10 test images, using the same feature
   extractor, image preprocessing, and 10-split Inception Score method as the `Hands-on VAE` project.
3. **Given** evaluation completes, **When** results are saved, **Then** a timestamped summary in
   JSON (and CSV) records the metrics, sample count, seed, checkpoint identity, and wall-clock time.
4. **Given** a trained checkpoint, **When** qualitative artifacts are generated, **Then** the system
   produces a large sample grid and a nearest-neighbor panel that pairs generated images with their
   closest real training images, to check for memorization.

---

### User Story 5 - Compare DDPM Against the VAE and Deliver the Report (Priority: P3)

A reviewer of the GenCV003 assignment wants a written report and a reproducible repository that
explain how the DDPM was built, show its results next to the baseline and enhanced VAE, and clearly
state the main difference between the two approaches and their trade-offs.

**Why this priority**: These are the two official GenCV003 deliverables, but they depend on all
earlier stories being complete.

**Independent Test**: Read the report and README; confirm the report contains every required section
and comparison table, and that following the README commands on a clean machine reproduces the
reported metrics within the stated tolerance.

**Acceptance Scenarios**:

1. **Given** DDPM and VAE metrics exist, **When** the benchmark command runs, **Then** it produces a
   side-by-side comparison table (FID, Inception Score, generation time per image, parameter count,
   training time) for DDPM, baseline VAE, and enhanced VAE.
2. **Given** the report is complete, **When** a reviewer reads it, **Then** it documents
   implementation steps, mathematical explanations, experimental results, qualitative assessment
   (diversity, realism, thematic consistency), and conclusions, and states the main difference
   between VAE and DDPM.
3. **Given** a clean environment, **When** a reviewer follows the README, **Then** they can set up
   the environment, download data, train, sample, and evaluate using only the documented commands.

---

### Edge Cases

- **Timestep boundaries**: Sampling the final step (t = 1) adds no noise; training timesteps at
  t = 1 and t = T both produce valid, finite losses.
- **Cosine schedule clipping**: Per-step noise levels from the cosine schedule are capped (at 0.999)
  so the final steps do not become numerically singular.
- **GPU unavailable**: All commands fall back to CPU with a clear warning; verification still
  completes within the CPU-mode time budget.
- **Out-of-memory during training or sampling**: The run stops with a clear message that suggests
  a smaller batch size or gradient accumulation, and the latest checkpoint stays intact.
- **Non-finite loss during training**: Training halts at the first NaN/Inf loss, records the step
  and epoch, and does not overwrite the best checkpoint.
- **Missing or incompatible checkpoint**: Sampling/evaluation fails fast with a message naming the
  missing file or the configuration mismatch (e.g., schedule or channel widths differ).
- **Missing VAE metrics**: The benchmark command still reports DDPM results and marks the VAE
  columns as unavailable instead of failing.
- **Colab session disconnect**: Because checkpoints are written to persistent storage every epoch,
  a disconnect loses at most one epoch; rerunning `train --resume` on a new session continues the run.
- **Interrupted sampling for benchmarking**: Long sampling jobs save generated images in batches so
  an interruption does not discard completed work.
- **Dataset not yet downloaded**: The first data-using command downloads CIFAR-10 automatically, or
  fails with a clear message when offline.

## Requirements *(mandatory)*

### Functional Requirements

**Diffusion process (from scratch)**

- **FR-001**: System MUST implement the forward diffusion process, the noise-prediction objective,
  and the reverse sampling procedure in project-authored code, without using pre-existing full
  diffusion implementations or pretrained diffusion models.
- **FR-002**: System MUST provide a linear variance schedule (β from 1e-4 to 0.02) and a cosine
  variance schedule (offset s = 0.008, per-step β capped at 0.999), selectable by configuration,
  with T = 1000 steps by default.
- **FR-003**: System MUST precompute all schedule-derived quantities (α_t, ᾱ_t, their square roots,
  and the reverse-posterior variance) once per run and reuse them for training and sampling.
- **FR-004**: System MUST noise a clean image to any timestep in a single closed-form step,
  x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε with ε drawn from a standard Gaussian.
- **FR-005**: System MUST train with the simplified objective: the mean squared error between the
  injected noise ε and the predicted noise, with timesteps drawn uniformly from 1..T.
- **FR-006**: System MUST generate images with Ho et al. Algorithm 2: start from pure Gaussian noise
  and apply T reverse steps, adding fresh noise at every step except the last.
- **FR-007**: System MUST support the reverse-step variance choice σ_t² = β_t (default) and
  σ_t² = β̃_t (posterior variance), selectable by configuration.

**Denoising network**

- **FR-008**: System MUST provide a time-conditioned U-Net denoising network for 32 × 32 × 3 images
  with stage channel widths [64, 128, 256], two residual blocks per stage, skip connections between
  matching encoder and decoder stages, and a middle bottleneck block.
- **FR-009**: System MUST condition every residual block on the timestep through a sinusoidal
  timestep embedding passed through a small learned projection.
- **FR-010**: System MUST apply multi-head spatial self-attention (4 heads) only at the 16 × 16 and
  8 × 8 resolutions and in the bottleneck, and MUST NOT apply attention at 32 × 32.
- **FR-011**: System MUST use group normalization and dropout (default 0.1) in residual blocks.
- **FR-012**: System MUST register schedules, network components, and the diffusion model in
  component registries so alternatives can be selected through configuration without code changes,
  mirroring the `Hands-on VAE` registry pattern.

**Data, training, and reproducibility**

- **FR-013**: System MUST load CIFAR-10 (50,000 train / 10,000 test images), scale pixels to
  [-1, 1], and hold out 10% of the training set as a fixed-seed validation split.
- **FR-014**: System MUST apply random horizontal flips as the only training augmentation by default.
- **FR-015**: System MUST maintain an EMA copy of the network weights (default decay 0.9999), use
  the EMA weights for sampling and evaluation by default, and allow raw weights to be selected.
- **FR-016**: System MUST support gradient clipping (default norm 1.0), gradient accumulation, and a
  configurable learning rate (default 2e-4). The default is a per-step batch of 32 images with 4
  accumulation steps (effective batch 128); both values MUST be configurable so larger GPUs can use
  bigger per-step batches with fewer accumulation steps at the same effective batch.
- **FR-017**: System MUST follow the `Hands-on VAE` checkpoint convention: save `latest.pt` every
  epoch, `best_checkpoint.pt` whenever validation loss improves, and `final_checkpoint.pt` at the end,
  each containing raw weights, EMA weights, optimizer state, epoch, global step, seed, and the full
  resolved configuration. The `train` command MUST accept a `--resume <checkpoint>` option that
  restores all of this state and continues from the next epoch.
- **FR-018**: System MUST seed every random source from a single configuration value and record the
  seed, configuration, and software/hardware environment with every run.
- **FR-018a**: System MUST run the same configuration unchanged on the local GPU and on a Google
  Colab GPU (T4), with the run output directory (checkpoints, metrics, samples) configurable so it
  can point to persistent storage such as Google Drive, allowing `train --resume` to continue across
  Colab session disconnects.
- **FR-018b**: System MUST train in full precision (fp32) by default and offer mixed precision as an
  opt-in setting; if mixed precision produces a non-finite loss, training MUST stop with a message
  recommending fp32, without overwriting the best checkpoint.

**Sampling and visualization**

- **FR-019**: System MUST generate a requested number of images from a checkpoint with a given seed
  and save both a sample grid and individual images.
- **FR-020**: System MUST produce reverse-process denoising strips showing frames at
  t = 1000, 800, 600, 400, 200, 100, 50, and 0.
- **FR-021**: System MUST clip final generated images to the valid pixel range before saving or
  scoring.
- **FR-022**: System MUST save training loss curves and periodic EMA sample grids during training.
- **FR-023**: System MUST produce a nearest-neighbor panel pairing generated images with their
  closest real training images for memorization checks.

**Evaluation and comparison**

- **FR-024**: System MUST report the average noise-prediction loss on the CIFAR-10 test split.
- **FR-025**: System MUST compute FID and Inception Score with the `Hands-on VAE` protocol:
  5,000 generated images vs. 5,000 real test images, the same pretrained Inception feature extractor
  and image preprocessing, and Inception Score averaged over 10 splits (mean and standard deviation).
- **FR-026**: System MUST write every evaluation result as a timestamped JSON and CSV summary
  including metric values, sample count, seed, checkpoint identity, and wall-clock times.
- **FR-027**: System MUST produce a comparison table of DDPM, baseline VAE, and enhanced VAE
  covering FID, Inception Score, generation time per image, parameter count, and training time,
  reading VAE results from their recorded metric files.

**Interface and deliverables**

- **FR-028**: System MUST expose `verify`, `train`, `evaluate`, `sample`, `denoise-strip`, and
  `benchmark` commands through a single command-line tool that takes a configuration file plus
  per-command overrides, prints human-readable progress, and returns a non-zero exit code on failure.
- **FR-029**: System MUST include a technical report documenting implementation steps, the
  mathematical formulation, experimental setup, quantitative results, a qualitative assessment of
  diversity, realism, and thematic consistency, and conclusions.
- **FR-030**: The report MUST explicitly explain the main difference between VAE and DDPM
  (single-step latent decoding with a pixel-space loss vs. multi-step iterative denoising with a
  noise-prediction loss) and the resulting trade-offs in sharpness, diversity, training stability,
  and generation speed.
- **FR-031**: The repository README MUST document environment setup and the exact commands needed
  to reproduce every reported result, both locally and on Google Colab (including mounting Drive
  and resuming an interrupted run), and MUST state which GPU produced the reported results.

### Key Entities

- **Experiment Configuration**: The single source of run settings: seed, device, data settings,
  network widths and attention placement, diffusion settings (T, schedule, β range, reverse
  variance), and training settings (epochs, learning rate, EMA decay, clipping, accumulation,
  intervals).
- **Noise Schedule**: The ordered per-step noise levels for T steps and all quantities derived from
  them; defined by schedule type and parameters.
- **Denoising Network**: The time-conditioned U-Net that takes a noisy image and timestep and
  predicts the noise; identified by its architecture settings and parameter count.
- **Checkpoint**: A saved training state (raw weights, EMA weights, optimizer state, epoch, step,
  seed, configuration); tagged latest, best, or final.
- **Training Run**: One execution of training, with its configuration, seed, per-epoch metrics log,
  loss curves, sample grids, and checkpoints, stored in its own run directory.
- **Sample Set**: Images generated from a checkpoint with a given seed and count, along with
  generation timing.
- **Denoising Trajectory**: The sequence of intermediate images recorded at selected timesteps for
  one generated image.
- **Evaluation Report**: A timestamped record of quantitative metrics (test loss, FID, Inception
  Score), sample count, seed, checkpoint identity, and timing.
- **Model Comparison**: A table combining DDPM, baseline VAE, and enhanced VAE metrics for the
  report.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Verification of the baseline configuration completes in under 2 minutes on the local
  GPU (under 10 minutes on CPU) and reports 100% of checks passing before any training run.
- **SC-002**: Every workflow (verify, train, sample, denoise-strip, evaluate, benchmark) runs to
  completion on the local GPU with measured total process GPU memory at or below 3 GB (3072 MiB).
- **SC-003**: Training loss falls by at least 50% between the first epoch and the final epoch, and
  no run produces invalid (NaN/Inf) losses.
- **SC-004**: The final DDPM achieves an FID below 50 on CIFAR-10 under the shared protocol,
  improving on both the baseline VAE (169.02) and the enhanced VAE (181.00).
- **SC-005**: The final DDPM achieves an Inception Score higher than both VAE models under the
  shared protocol.
- **SC-006**: Repeating any sampling command with the same checkpoint and seed produces identical
  images in 100% of trials.
- **SC-007**: The 5,000-image benchmark (sampling plus scoring) completes in under 3.5 hours on the
  local GPU (measured sampling estimate: 2.8–3.1 hours).
- **SC-008**: A reviewer following only the README can reproduce the reported FID within ±5% and
  Inception Score within ±0.3 using the published checkpoint and seed.
- **SC-009**: The report contains all required GenCV003 sections and the DDPM vs. VAE comparison
  table, and a reviewer can state the main difference between the two approaches after reading it.
- **SC-010**: In the nearest-neighbor panel, no generated image is a near-duplicate of a training
  image, as judged by visual inspection of at least 64 samples.

## Assumptions

- **Dataset**: CIFAR-10 is the public dataset for both models, matching `Hands-on VAE`; it is
  downloaded automatically on first use.
- **Scope**: This feature covers an unconditional baseline DDPM only. Class-conditional generation,
  classifier/classifier-free guidance, accelerated samplers (e.g., DDIM), learned variances, and
  latent diffusion are out of scope and may be added in later features (e.g., `002-enhanced-ddpm`).
- **Hardware**: Development and verification run on the local NVIDIA Quadro T2000 (4 GB physical,
  3 GB usable budget per SC-002). The official training run for reported results runs on a free
  Google Colab T4 (15 GB) with the same configuration and seed (estimated ~3.5–4 h in fp32 versus
  ~8.5 h measured locally; the Colab estimate is unmeasured). The 3 GB budget remains the default so
  every workflow still runs locally. Mixed precision is opt-in only, because the local probe showed
  it was slower and produced non-finite losses on the T2000; its benefit on the T4 is unverified.
- **Cross-hardware reproducibility**: Bitwise determinism (SC-006) is guaranteed only on the same
  hardware and software; results reproduced on a different GPU are expected to match within the
  SC-008 tolerances.
- **Training budget**: The default baseline trains for 100 epochs with per-step batch 32 and 4-step
  gradient accumulation (effective batch 128); if this cannot reach SC-004 within the hardware budget, the epoch count is
  increased and the change is recorded in the report rather than altering the architecture.
- **VAE reference metrics**: Baseline VAE (FID 169.02) and enhanced VAE (FID 181.00) results are
  taken from the recorded `Hands-on VAE` metric files and were produced with the same 5,000-sample
  protocol; the VAE models are not retrained in this feature.
- **Protocol parity**: "Matching the VAE protocol" means the same sample count, real-image source
  (test split), Inception feature extractor, preprocessing, and Inception Score splitting; FID values
  are therefore comparable across both repositories but not to published 50,000-sample FID numbers.
- **Evaluation weights**: Pretrained Inception weights are used only for metric computation, which
  the constitution explicitly permits.
- **Report location**: The report is a Markdown document in the repository's reports folder, and the
  final cross-model comparison is written there, mirroring `Hands-on VAE`.
