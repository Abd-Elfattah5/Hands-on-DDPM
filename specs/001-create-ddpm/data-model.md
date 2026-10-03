# Data Model: From-Scratch DDPM

**Feature**: `001-create-ddpm`
**Date**: 2026-10-02
**Source**: [spec.md](./spec.md) Key Entities, [research.md](./research.md)

The structure mirrors `Hands-on VAE/specs/001-create-vae/data-model.md`.

---

## 1. Core Domain Entities

| Entity | Represents | Key attributes | Relationships |
|---|---|---|---|
| Experiment Configuration | Single source of run settings | `experiment`, `data`, `model`, `diffusion`, `training`, `sampling`, `evaluation`, `verify` sections | Drives every other entity; embedded in every Checkpoint |
| Noise Schedule | Per-step noise levels and derived quantities | `name` (`linear`/`cosine`), `timesteps` T, β range or cosine offset `s`, precomputed buffers | Owned by the Gaussian Diffusion process |
| Gaussian Diffusion | Forward noising, loss, reverse sampling | schedule, `variance_type`, `loss_type` | Wraps the Denoising Network |
| Denoising Network | Time-conditioned U-Net ε_θ(x_t, t) | `channels`, `num_res_blocks`, `attention_resolutions`, `num_heads`, `dropout`, `time_embed_dim`, parameter count | Owned by Gaussian Diffusion; has an EMA copy |
| EMA Weights | Slow moving average of network weights | `decay`, `num_updates`, shadow state dict | Saved in every Checkpoint; used for sampling and evaluation |
| Checkpoint | Saved training state | see §4 | Belongs to one Training Run; tagged latest / best / epoch / final |
| Training Run | One training execution | `output_dir`, history records, artifacts | Produces Checkpoints, metrics, sample grids |
| Sample Set | Generated images | `seed`, `num_samples`, `weights` (`ema`/`raw`), timing | Produced from a Checkpoint |
| Denoising Trajectory | Intermediate frames of one reverse chain | frame indices (paper t = 1000…0), frames tensor `[F, B, 3, 32, 32]` | Produced from a Checkpoint |
| Evaluation Report | Quantitative results | see §5 | Produced from a Checkpoint |
| Model Comparison | DDPM vs VAE table in the final report | rows per model, metric columns | Written manually from the Evaluation Report and the VAE results |

---

## 2. In-Memory Tensor Contracts (`src/models/types.py`)

### 2.1 `DiffusionOutput` (training forward pass)

| Field | Shape / Type | Meaning |
|---|---|---|
| `x_t` | `[B, C, H, W]` float32 | Noised input at sampled timesteps |
| `t` | `[B]` int64, values in `[0, T-1]` | Sampled timestep indices (zero-based) |
| `noise` | `[B, C, H, W]` | Target ε ~ N(0, I) |
| `predicted_noise` | `[B, C, H, W]` | ε_θ(x_t, t) |
| `extra` | `dict[str, Tensor]` | Optional diagnostics |

### 2.2 `LossOutput`

| Field | Type | Meaning |
|---|---|---|
| `loss` | scalar Tensor (requires grad) | L_simple, mean over batch and pixels |
| `metrics` | `dict[str, float]` | Detached scalars for logging (e.g., `mse`) |

### 2.3 `SamplingOutput`

| Field | Shape / Type | Meaning |
|---|---|---|
| `samples` | `[N, C, H, W]` clamped to [-1, 1] | Final x₀ |
| `trajectory` | `[F, N, C, H, W]` or `None` | Frames at the requested indices |
| `trajectory_steps` | `list[int]` (paper t labels) | e.g. `[1000, 800, 600, 400, 200, 100, 50, 0]` |
| `seconds` | float | Wall-clock sampling time |

---

## 3. Configuration Entities & Schema (`src/configs/schema.py`)

Dataclass sections; unknown keys raise `ConfigError`. Defaults correspond to `configs/cifar10_baseline.yaml`.

| Section | Field | Default | Validation |
|---|---|---|---|
| `experiment` | `name` | `cifar10_baseline` | non-empty |
| | `seed` | `42` | int ≥ 0 |
| | `device` | `auto` | `auto` / `cuda` / `cpu` |
| | `output_dir` | `artifacts/runs/cifar10_baseline` | path |
| `data` | `dataset` | `cifar10` | `cifar10` |
| | `data_dir` | `data` | path |
| | `batch_size` | `32` | int ≥ 1 (per-step micro-batch) |
| | `num_workers` | `4` | int ≥ 0 |
| | `val_split` | `0.1` | 0 < x < 1 |
| | `random_flip` | `true` | bool |
| `model` | `name` | `unet` | registered network name |
| | `in_channels` / `out_channels` | `3` / `3` | ≥ 1, equal |
| | `image_size` | `32` | divisible by 2^(len(channels)−1) |
| | `channels` | `[64, 128, 256]` | each divisible by `num_groups` |
| | `num_res_blocks` | `2` | ≥ 1 |
| | `attention_resolutions` | `[16, 8]` | subset of the U-Net resolutions; must not include `image_size` (FR-010) |
| | `num_heads` | `4` | divides every attended channel width |
| | `num_groups` | `32` | ≥ 1 |
| | `dropout` | `0.1` | 0 ≤ x < 1 |
| | `time_embed_dim` | `256` | ≥ 1 |
| `diffusion` | `name` | `gaussian` | registered diffusion name (added at implementation, T005, so the registry selects the process like `model.name`) |
| | `schedule` | `linear` | registered schedule name |
| | `timesteps` | `1000` | ≥ 1 |
| | `beta_start` / `beta_end` | `1e-4` / `0.02` | 0 < start < end < 1 (linear) |
| | `cosine_s` | `0.008` | > 0 (cosine) |
| | `variance_type` | `fixed_large` | `fixed_large` / `fixed_small` |
| `training` | `epochs` | `100` | ≥ 1 |
| | `lr` | `2e-4` | > 0 |
| | `weight_decay` | `0.0` | ≥ 0 |
| | `warmup_steps` | `5000` | ≥ 0 (optimizer steps) |
| | `grad_accum_steps` | `4` | ≥ 1 (effective batch = batch_size × grad_accum_steps) |
| | `gradient_clip_val` | `1.0` | > 0 |
| | `ema_decay` | `0.9999` | 0 < x < 1 |
| | `mixed_precision` | `none` | `none` / `bf16` / `fp16` |
| | `save_every` | `10` | ≥ 1 epochs |
| | `sample_every` | `5` | ≥ 1 epochs (EMA grid) |
| `sampling` | `batch_size` | `256` | ≥ 1 |
| | `strip_steps` | `[1000, 800, 600, 400, 200, 100, 50, 0]` | descending, within [0, T] |
| `evaluation` | `num_samples` | `5000` | ≥ 10 and divisible by `is_splits` |
| | `batch_size` | `64` | ≥ 1 (Inception extraction) |
| | `is_splits` | `10` | ≥ 1 |
| `verify` | `memory_budget_mib` | `3072` | > 0 |

---

## 4. Persistent Checkpoint Schema (`.pt`)

| Key | Type | Notes |
|---|---|---|
| `format_version` | str | `"1.0"` |
| `epoch` | int | Last completed epoch (resume starts at `epoch + 1`) |
| `global_step` | int | Optimizer steps (not micro-batches) |
| `model_state_dict` | dict | Full `GaussianDiffusion.state_dict()` with raw weights (keys prefixed `denoiser.`; schedule buffers are non-persistent and rebuilt from `config`) |
| `ema_state_dict` | dict | `{decay, num_updates, shadow}`; `shadow` is the **denoiser-only** EMA state dict, loaded with `model.denoiser.load_state_dict` |
| `optimizer_state_dict` | dict | AdamW |
| `scheduler_state_dict` | dict or None | Warmup scheduler |
| `scaler_state_dict` | dict or None | Only when fp16 |
| `best_val_loss` | float | EMA validation loss of the best epoch so far |
| `history` | list[dict] | All epoch records (restored on resume) |
| `rng_state` | dict | Python, NumPy, torch CPU, and CUDA RNG states |
| `config` | dict | Full resolved configuration |
| `metrics` | dict | Record of this epoch |
| `provenance` | dict | `torch_version`, `cuda_available`, `device_name`, `timestamp` (UTC ISO), `seed`, `git_commit` (if available) |

Tags and files: `latest.pt` (every epoch), `best_checkpoint.pt` and `best.pt` (on improvement),
`epoch_NNN.pt` (every `save_every`), `final_checkpoint.pt` (end of training).

### Epoch history record (`metrics.json` list item)

`epoch, global_step, lr, train_loss, val_loss, val_loss_ema, epoch_seconds, peak_memory_mib`

---

## 5. Evaluation Report Schema (JSON)

| Key | Type | Producer |
|---|---|---|
| `timestamp` | str (UTC ISO) | all |
| `checkpoint` | str | all |
| `weights` | `ema` / `raw` | all |
| `seed` | int | all |
| `test_loss` | float | `evaluate`, `benchmark` |
| `total_samples` | int (test images) | `evaluate`, `benchmark` |
| `fid` | float (2 dp) | `benchmark` |
| `inception_score_mean` / `inception_score_std` | float | `benchmark` |
| `benchmark_samples` | int (5000) | `benchmark` |
| `sampling_seconds_total` / `sampling_seconds_per_image` | float | `benchmark`, `sample` |
| `class_coverage` | object: `distinct_top1_classes` (int), `marginal_entropy_nats` (float), `top20` (list of `[imagenet_index, count]`) | `benchmark` (distribution coverage for the report, FR-029) |
| `num_parameters` | int | `benchmark` |
| `training_seconds` | float (from the run's `metrics.json`, if present) | `benchmark` |
| `device_name` | str | all |

These keys are everything the author needs for the manual DDPM vs. VAE table in the final report.

---

## 6. Validation Rules & State Transitions

- Training run states: `initialized → training ⇄ (interrupted → resumed) → completed | failed(nan|oom)`.
  - `failed(nan)` keeps `best_checkpoint.pt` untouched; `latest.pt` is the last finite epoch.
  - Resume requires the checkpoint config to match the current config on model and diffusion
    sections; a mismatch causes exit code 1 with the differing keys listed.
- Timestep indices are always in `[0, T-1]`; paper labels are `index + 1`, plus `0` for the clean image.
- All saved images are clamped to [-1, 1] before unnormalizing to [0, 1].
- `evaluation.num_samples % evaluation.is_splits == 0` (5000 / 10 = 500 per split).
