# Research & Design Decisions: From-Scratch DDPM

**Feature**: `001-create-ddpm`
**Date**: 2026-10-02
**Inputs**: [spec.md](./spec.md), `.specify/memory/constitution.md` (v1.0.0), `HANDOFF.md`,
sibling repository `Hands-on VAE` (conventions inventory), GPU probe
([research/gpu_memory_results.md](./research/gpu_memory_results.md)).

Every decision below uses the format **Decision / Rationale / Alternatives considered**.
All Technical Context unknowns are resolved; no `NEEDS CLARIFICATION` items remain.

---

## 1. Dataset & Preprocessing

- **Decision**: CIFAR-10 via `torchvision.datasets.CIFAR10`, `ToTensor()` then
  `Normalize((0.5,)*3, (0.5,)*3)` to [-1, 1]; `RandomHorizontalFlip` on train only; 45,000/5,000
  train/validation split with `torch.utils.data.random_split` and a `Generator` seeded from
  `experiment.seed`; 10,000-image test split for evaluation and FID real features.
- **Rationale**: Identical to `Hands-on VAE/src/data/cifar10.py`, so FID real-image statistics and
  splits are directly comparable (Constitution III). Horizontal flip is the augmentation Ho et al. use
  on CIFAR-10.
- **Alternatives considered**: No augmentation (lower sample diversity per epoch); full 50k train set
  without validation (breaks the `best_checkpoint.pt` convention, which needs a validation loss).

## 2. Noise Schedules

- **Decision**: Register `linear` (β from 1e-4 to 0.02, `torch.linspace`, float64) and `cosine`
  (Nichol & Dhariwal, s = 0.008, β clipped to ≤ 0.999) schedules. Precompute all buffers once in
  float64, then cast to float32 and register them as non-persistent module buffers: `betas`, `alphas`,
  `alphas_cumprod`, `alphas_cumprod_prev`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`,
  `sqrt_recip_alphas`, `posterior_variance`, `posterior_log_variance_clipped`, `posterior_mean_coef1/2`.
- **Rationale**: Computing in float64 avoids cumulative-product drift over 1,000 steps. Precomputing
  buffers satisfies FR-003 and makes `verify` checks (monotonic ᾱ_t, ᾱ_T ≈ 0) trivial.
- **Alternatives considered**: Computing on the fly per step (slower, duplicated logic); sigmoid or
  continuous SDE schedules (out of scope; the registry leaves room for them in a later feature).

## 3. Timestep Indexing Convention

- **Decision**: Internally use zero-based indices t ∈ {0, …, T−1}, where index 0 is paper step 1.
  User-facing artifacts (denoising strips, docs) use paper notation t ∈ {1000, …, 1} plus t = 0 for
  the final clean image.
- **Rationale**: Zero-based indexing maps directly to buffer lookups (`betas[t]`) and avoids
  off-by-one errors; the strip renderer translates indices to paper labels.
- **Alternatives considered**: One-based internal indices (requires `t-1` at every lookup, a common
  source of bugs).

## 4. Denoising Network (U-Net)

- **Decision**: Ho et al.-style U-Net: input conv 3→64; three resolution levels with widths
  [64, 128, 256] (32×32, 16×16, 8×8); 2 residual blocks per level on the down path and 3 on the up
  path (one per skip connection); 4-head self-attention after each residual block at 16×16 and 8×8;
  middle block Res → Attn → Res at 8×8; downsampling with a stride-2 3×3 conv; upsampling with
  nearest-neighbor ×2 followed by a 3×3 conv; output GroupNorm(32) → SiLU → conv 64→3, zero-initialized.
  Residual blocks: GroupNorm(32) → SiLU → conv → add time projection → GroupNorm → SiLU → Dropout(0.1)
  → conv (zero-initialized) with a 1×1 skip when channel counts change. Time embedding: sinusoidal
  (dim 64) → Linear(64, 256) → SiLU → Linear(256, 256).
- **Rationale**: This exact layout was measured by the GPU probe: 16.06 M parameters,
  1,625 MiB peak at 32 × 4 (well under 3 GB). Zero-initializing the last conv of each block and the
  output head makes the network start close to an identity mapping, which stabilizes early training.
  Nearest+conv upsampling avoids the checkerboard artifacts of transposed convolutions.
- **Alternatives considered**: Scale-shift (FiLM) time conditioning (marginal gain, deferred to an
  enhanced feature); attention at 32×32 (forbidden by the memory budget, FR-010); transposed-conv
  upsampling (checkerboard artifacts).

## 5. Attention Implementation

- **Decision**: Implement multi-head attention from primitives (GroupNorm → 1×1 conv to Q, K, V →
  attention → 1×1 output conv, residual add), calling `torch.nn.functional.scaled_dot_product_attention`
  for the softmax(QKᵀ/√d)V product.
- **Rationale**: The probe measured the fused kernel at 170 MiB less memory and ~6% faster than an
  explicit einsum/softmax. It is a tensor primitive, not a pre-packaged model, so it complies with
  Constitution I. A unit test compares it with an explicit einsum reference to document the maths.
- **Alternatives considered**: `nn.MultiheadAttention` (a higher-level wrapper that hides the
  projection layout); explicit einsum (kept only as the test reference).

## 6. Training Objective, Optimizer & EMA

- **Decision**: L_simple = MSE(ε, ε_θ(x_t, t)), with t ~ Uniform{0..T−1} sampled per image. AdamW
  (lr 2e-4, weight decay 0), linear learning-rate warmup over the first 5,000 optimizer steps then
  constant, gradient clipping at norm 1.0. Per-step batch 32 × 4 accumulation steps = effective batch
  128. EMA with decay 0.9999, updated after every optimizer step, kept on the GPU (≈ 64 MB, included
  in the probe measurement). EMA weights are used for sampling and evaluation by default; the
  validation loss that selects `best_checkpoint.pt` is computed with the EMA weights.
- **Rationale**: Matches Ho et al. (lr 2e-4, warmup 5,000, clip 1.0, EMA 0.9999, batch 128). Using
  EMA weights for validation makes "best" track the weights that are actually used for generation.
- **Alternatives considered**: Cosine LR decay as in the VAE (Ho et al. use a constant rate; decay
  interacts badly with EMA warm-up); EMA on CPU (slower, unnecessary within the budget); no warmup
  (early instability at batch 128).

## 7. Precision & Determinism

- **Decision**: fp32 by default; `training.mixed_precision: none` with opt-in `bf16`/`fp16` autocast
  plus GradScaler for fp16. A non-finite loss stops training with exit code 2 and does not overwrite
  `best_checkpoint.pt` (FR-018b). Seeding uses the VAE `seed_everything` (Python, NumPy, torch, CUDA,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`). Sampling uses a dedicated `torch.Generator`
  seeded from `--seed`, so results do not depend on prior RNG consumption.
- **Rationale**: The local probe showed fp16 was 4× slower and produced NaN on the T2000. A dedicated
  sampling generator gives bitwise-identical samples for the same seed on the same hardware (SC-006).
- **Alternatives considered**: AMP by default (unsafe on the T2000); global RNG for sampling (results
  change when unrelated code consumes random numbers).

## 8. Reverse Sampling (Algorithm 2)

- **Decision**: `p_sample_loop` runs indices T−1 … 0: predict ε, compute
  μ = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) · ε), add σ_t·z for t > 0 (z = 0 at t = 0), where σ_t² = β_t
  (`fixed_large`, default) or β̃_t (`fixed_small`), selectable via `diffusion.variance_type`. The final
  output is clamped to [-1, 1]. An optional `return_trajectory` captures frames at configured indices.
  Generation runs in batches (default 256, measured 2,061 MiB) and writes each finished batch to disk.
- **Rationale**: This is a direct transcription of Algorithm 2. Batch-wise writing satisfies the
  interrupted-sampling edge case. The measured throughput at batch 256 gives ~2.8 h for 5,000 images
  on the T2000 (SC-007 < 3.5 h).
- **Alternatives considered**: Predicting x₀ and clipping at every step (deviates from Algorithm 2;
  possible enhancement); DDIM (out of scope).

## 9. Quantitative Evaluation Protocol (Parity with VAE)

- **Decision**: Port `Hands-on VAE/src/evaluation/metrics.py` unchanged in behavior:
  `InceptionFeatureExtractor` (torchvision `Inception_V3_Weights.DEFAULT`, bilinear resize to 299,
  [0,1]→[-1,1], 2048-d pool features, 1000-way softmax), `calculate_frechet_distance` (scipy `sqrtm`),
  and `calculate_inception_score(splits=10)`. Real features come from the first 5,000 CIFAR-10 test
  images; 5,000 EMA samples are generated with a fixed seed. Output keys extend the VAE schema
  (JSON only):
  `fid`, `inception_score_mean`, `inception_score_std`, `benchmark_samples`, plus DDPM-specific
  `test_loss`, `sampling_seconds_total`, `sampling_seconds_per_image`, `num_parameters`.
- **Rationale**: Identical extractor, preprocessing, sample count and splits are the definition of
  "matching the VAE protocol" in the spec, so FID 169.02 / 181.00 are comparable to the DDPM result.
  The code is copied rather than imported, because the repositories are independent.
- **Alternatives considered**: `pytorch-fid` / `torchmetrics` (different Inception weights, so not
  comparable with the VAE results); 50k-sample FID (not comparable with the VAE; ~28 h of sampling on
  the T2000).

## 10. VAE Comparison (Manual, in the Final Report)

- **Decision**: The repository does not read, copy or tabulate VAE metrics. `ddpm benchmark` records
  every DDPM value needed for the comparison (FID, IS mean ± std, seconds per image, parameter count,
  training time taken from the run's `metrics.json`). The author writes the DDPM vs. VAE table
  manually in `docs/reports/001-baseline-ddpm-report.md`.
- **Rationale**: User decision (plan review, 2026-10-02). It keeps the two repositories independent
  and avoids comparison code that is used exactly once.
- **Alternatives considered**: A tracked `benchmarks/vae_reference_metrics.json` plus generated
  `comparison.md` (rejected by the user as unnecessary).

## 11. CLI Framework & Command Set

- **Decision**: Typer app `ddpm` (console script `ddpm = "src.cli.main:app"`, already in
  `pyproject.toml`) with commands `verify`, `train`, `evaluate`, `sample`, `denoise-strip` and
  `benchmark`, plus `--version`. It mirrors VAE option names (`-c/--config`, `-k/--checkpoint`,
  `-r/--resume`, `-s/--seed`, `-n/--num-samples`, `-o/--out`, `--upscale`) and adds DDPM overrides
  (`--output-dir`, `--grad-accum`, `--mixed-precision`, `--weights ema|raw`). The `sample --nearest`
  flag produces the nearest-neighbor memorization panel (FR-023); it is an option of `sample`
  rather than a seventh command, because it post-processes the images `sample` just generated. Exit codes: 0 success, 1
  configuration/input error, 2 runtime error (OOM, NaN), which fixes a gap in the VAE where code 2
  was documented but never returned.
- **Rationale**: Constitution V requires a CLI for every workflow; keeping the VAE's option vocabulary
  makes the two repositories feel like one ecosystem.
- **Alternatives considered**: argparse (inconsistent with the VAE); a separate `nearest-neighbors`
  command (the spec fixes the command list at six).

## 12. Training Artifacts, Logging & Resume

- **Decision**: Follow the VAE layout in `<output_dir>/`: `latest.pt` (every epoch),
  `best_checkpoint.pt` + `best.pt` (EMA validation loss improved), `epoch_NNN.pt` (every
  `save_every`), `final_checkpoint.pt`, `metrics.json` (list of epoch records), `loss_curve.png`,
  `train.log`, and `samples/epoch_NNN.png` (EMA 8×8 grid every
  `sample_every` epochs). Resume restores model, EMA, optimizer, scheduler, scaler, epoch,
  `global_step`, `best_val_loss`, **history** and RNG states.
- **Rationale**: Matches the VAE convention the user requested. JSON only (user decision; the
  constitution's "JSON/CSV" is satisfied by JSON). Restoring history fixes the VAE issue where
  resuming overwrote `metrics.json` with only the resumed epochs.
- **Alternatives considered**: TensorBoard/W&B (an extra dependency, not used by the VAE).

## 13. Colab Training Path

- **Decision**: Add `notebooks/ddpm_colab_training.ipynb`, a thin driver with no model code:
  (1) check the GPU; (2) mount Google Drive; (3) get the code by cloning the private repository using
  a GitHub token stored in Colab Secrets (`GH_TOKEN`), or fall back to a repository zip on Drive;
  (4) `pip install -e .`; (5) download CIFAR-10 to local `/content/data` (fast disk, not Drive);
  (6) `ddpm verify`; (7) `ddpm train --config configs/cifar10_colab.yaml --output-dir
  /content/drive/MyDrive/hands-on-ddpm/runs/cifar10_baseline`, adding `--resume .../latest.pt`
  automatically when it exists; (8) optional `ddpm benchmark` and `ddpm denoise-strip`; (9) display
  the loss curve and samples. `configs/cifar10_colab.yaml` is identical to the baseline except for
  `batch_size: 128`, `grad_accum_steps: 1` and `num_workers: 2` (same effective batch 128). The
  inherited `*.ipynb` ignore rule is removed (only `.ipynb_checkpoints/` stays ignored) so the
  notebook is tracked in git.
- **Rationale**: Constitution "Prohibited Patterns" forbids training loops entangled in notebooks
  without tested library modules, so the notebook only calls the tested CLI. Checkpoints on Drive plus
  the `--resume` convention survive free-tier disconnects (FR-018a). Same effective batch keeps the
  optimization identical to the local run.
- **Alternatives considered**: A self-contained notebook like `vae_colab_diagnostic.ipynb`
  (duplicates the model and breaks Constitution I/IV traceability); TPU (needs `torch_xla` and is slow
  for Python sampling loops).

## 14. Memory Verification in `verify`

- **Decision**: `ddpm verify` runs one full training step (forward, backward, optimizer and EMA
  update) at the configured per-step batch on CUDA and reports `torch.cuda.max_memory_reserved()` plus
  the CUDA context overhead. It fails when the total exceeds `verify.memory_budget_mib` (default 3072).
  A pytest test (`tests/unit/test_memory_budget.py`, skipped without CUDA) asserts the same for train,
  sample (batch 256) and Inception extraction (batch 64).
- **Rationale**: Turns the 3 GB clarification into an automated gate (SC-002), using the probe
  methodology from `research/gpu_memory_probe.py`.
- **Alternatives considered**: A one-off manual measurement only (does not catch regressions).

## 15. Repository Parity Gap Analysis (vs. Hands-on VAE)

Status after the plan review on 2026-10-02 (every deviation below is justified in
`docs/adr/0001-deviations-from-hands-on-vae-conventions.md`):

| VAE artifact | DDPM decision |
|---|---|
| `README.md` (overview, install, verify, CLI guide, tests) | Rewritten in P8 with the same sections plus a Colab section; every command must match `ddpm --help` |
| `HANDOFF.md` | Present; updated at the end of each phase |
| `CONTEXT.md` (domain glossary) | **Created** in this planning step |
| `ARCHITECTURE_DEEP_DIVE.md` | **Created** in this planning step; results added in P8 |
| `specs/.../contracts/cli.md` | **Created** in this planning step |
| `docs/adr/` | ADR 0001 records all deviations (required by the constitution; added 2026-10-03) |
| `docs/reports/` | `001-baseline-ddpm-report.md` written after results exist (P8), with the manual VAE comparison |
| `configs/cifar10_baseline.yaml` | Present; updated to 32 × 4 and the new schema keys in P1 |
| `configs/mnist_baseline.yaml` + `src/data/mnist.py` | Not created (user decision; ADR 0001 D1); HANDOFF now carries a note |
| `configs/cifar10_colab.yaml` | New (P7) |
| `src/**`, `tests/**` | Created during implementation (P1–P9) |
| Colab notebook | `notebooks/ddpm_colab_training.ipynb`, tracked in git (P7) |
| `.gemini/`, `GEMINI.md`, `graphify-out/` | Not needed now; the user runs graphify after implementation |
| VAE benchmark metrics | Not copied; added manually to the final report |
| `pyproject.toml` | `numpy` added explicitly |
| `.gitignore` | `*.ipynb` rule removed so notebooks are tracked |
| `artifacts/` layout | Mirrors the VAE: `runs/`, `samples/`, `eval/`, plus `strips/` |
