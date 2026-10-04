---

description: "Task list for feature 001-create-ddpm"
---

# Tasks: From-Scratch Denoising Diffusion Probabilistic Model (DDPM)

**Input**: Design documents from `/specs/001-create-ddpm/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/cli.md`, `contracts/component-interfaces.md`, `quickstart.md`, `research/gpu_memory_results.md`

**Tests**: Automated unit tests for tensor shapes, diffusion formulas, gradient flow, numerical stability and sampling determinism are **mandated by Constitution Principle IV**, so test tasks are included and MUST be written before the implementation they cover.

**Organization**: Tasks are grouped by user story (spec.md US1–US5) so each story can be implemented and validated independently. The plan's phases P1–P9 map to these phases as follows: P1→Phase 1, P2→Phase 2, P3→Phase 3, P4→Phase 4 (and Colab tooling), P5→Phase 5, P6→Phase 6, P7+P8→Phase 7, P9→Phase 8.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1–US5)
- Every task names its exact file path

## Conventions for every task (read once)

- **Python 3.12, PyTorch 2.6**, venv `DDPM/` (`source DDPM/bin/activate`). Run tests with `pytest tests/ -v`.
- **Mirror `Hands-on VAE`** (`/home/amousa1/Projects/Hands-on VAE`) for style: module docstrings, type hints, `from __future__`-free Python 3.12 typing (`list[int]`, `X | None`), Typer CLI, `get_logger` logging, `├──`/`└──` console trees. Where a task says **(port)**, copy the named VAE file and apply only the listed changes.
- **No `src/__init__.py`** at the top level (as in the VAE); every subpackage (`src/cli`, `src/configs`, `src/data`, `src/models`, `src/training`, `src/evaluation`, `src/utils`) has an empty `__init__.py`. Imports are absolute: `from src.models.registry import ...`.
- **Timesteps**: internal indices are zero-based `t ∈ [0, T-1]` (index 0 = paper step 1). Paper labels in strips/docs are `index + 1`, plus `0` for the final clean image (research §3).
- **Metrics are JSON only** (no CSV). **No MNIST**, **no VAE metric files or comparison code** (user decisions; see spec Clarifications). Every deviation from `Hands-on VAE` conventions is justified in `docs/adr/0001-deviations-from-hands-on-vae-conventions.md`; add a row there before introducing a new one.
- **Forbidden**: `diffusers`, `pytorch-fid`, `torchmetrics`, `nn.MultiheadAttention`, or any pretrained diffusion weights (Constitution I). Pretrained Inception-v3 is allowed only in `src/evaluation/metrics.py`.
- **Memory budget**: 3072 MiB total per process (spec SC-002). fp32 by default.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Package layout, configs, and ported utilities.

- [X] T001 Create the package skeleton: empty `__init__.py` files in `src/cli/`, `src/configs/`, `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/utils/`, plus directories `tests/unit/` (no `__init__.py`, as in the VAE) and `notebooks/` (add `notebooks/.gitkeep`); verify `pyproject.toml` keeps `ddpm = "src.cli.main:app"`, `numpy>=1.26`, and `[tool.setuptools.packages.find] include = ["src*"]`, then run `pip install -e ".[dev]"` in `pyproject.toml`
- [X] T002 [P] (port) Copy `Hands-on VAE/src/utils/seeding.py` to `src/utils/seeding.py` keeping `seed_everything(seed: int = 42) -> None` unchanged (Python `random`, `PYTHONHASHSEED`, NumPy, torch, CUDA, `cudnn.deterministic=True`, `cudnn.benchmark=False`), and add `make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator` returning a `torch.Generator(device=device).manual_seed(seed)` used for all sampling (research §7)
- [X] T003 [P] (port) Copy `Hands-on VAE/src/utils/logging.py` to `src/utils/logging.py`, changing only the default logger name from `"vae"` to `"ddpm"` (format `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`, `datefmt="%Y-%m-%dT%H:%M:%S%z"`, stdout + optional append-mode file handler, handlers added once)
- [X] T004 [P] Implement GPU memory helpers in `src/utils/memory.py`: `reset_peak(device)` (synchronize, `empty_cache`, `reset_peak_memory_stats`), `peak_reserved_mib(device) -> int`, `process_total_mib() -> int | None` (parse `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` for `os.getpid()`, fall back to `nvidia-smi --query-gpu=memory.used`; return `None` if `nvidia-smi` is unavailable), and `measured_total_mib(device) -> int` = `process_total_mib()` if available else `peak_reserved_mib + 300` (CUDA context estimate). Reuse the methodology of `specs/001-create-ddpm/research/gpu_memory_probe.py`
- [X] T005 Implement the typed config schema in `src/configs/schema.py` following `Hands-on VAE/src/configs/schema.py` (`ConfigError(ValueError)`, `load_config(path) -> dict` with `yaml.safe_load`, `FileNotFoundError` on missing file, `ConfigError` if top level is not a mapping). Define dataclasses `ExperimentSectionConfig`, `DataSectionConfig`, `ModelSectionConfig`, `DiffusionSectionConfig`, `TrainingSectionConfig`, `SamplingSectionConfig`, `EvaluationSectionConfig`, `VerifySectionConfig`, `ExperimentConfig` with exactly the fields/defaults in data-model.md §3, and `validate_config(config: dict) -> ExperimentConfig` enforcing verbatim: `seed` "int ≥ 0"; `device` "`auto` / `cuda` / `cpu`"; `dataset` "`cifar10`"; `batch_size` "int ≥ 1 (per-step micro-batch)"; `num_workers` "int ≥ 0"; `val_split` "0 < x < 1"; `in_channels`/`out_channels` "≥ 1, equal"; `image_size` "divisible by 2^(len(channels)−1)"; `channels` "each divisible by `num_groups`"; `num_res_blocks` "≥ 1"; `attention_resolutions` "subset of the U-Net resolutions; must not include `image_size` (FR-010)"; `num_heads` "divides every attended channel width"; `dropout` "0 ≤ x < 1"; `timesteps` "≥ 1"; `beta_start`/`beta_end` "0 < start < end < 1 (linear)"; `cosine_s` "> 0 (cosine)"; `variance_type` "`fixed_large` / `fixed_small`"; `epochs` "≥ 1"; `lr` "> 0"; `weight_decay` "≥ 0"; `warmup_steps` "≥ 0"; `grad_accum_steps` "≥ 1"; `gradient_clip_val` "> 0"; `ema_decay` "0 < x < 1"; `mixed_precision` "`none` / `bf16` / `fp16`"; `save_every`, `sample_every` "≥ 1"; `strip_steps` "descending, within [0, T]"; `evaluation.num_samples` "≥ 10 and divisible by `is_splits`"; `memory_budget_mib` "> 0". Unknown keys MUST raise `ConfigError` naming the key. Also provide `apply_overrides(config: dict, overrides: dict[str, Any]) -> dict` for dotted keys (e.g. `"data.batch_size"`), returning a deep copy
- [X] T006 [P] Rewrite `configs/cifar10_baseline.yaml` to match data-model.md §3 exactly: `experiment` (name `cifar10_baseline`, seed 42, device auto, output_dir `artifacts/runs/cifar10_baseline`), `data` (cifar10, `data`, batch_size 32, num_workers 4, val_split 0.1, random_flip true), `model` (name unet, in/out 3, image_size 32, channels [64,128,256], num_res_blocks 2, attention_resolutions [16,8], num_heads 4, num_groups 32, dropout 0.1, time_embed_dim 256), `diffusion` (schedule linear, timesteps 1000, beta_start 0.0001, beta_end 0.02, cosine_s 0.008, variance_type fixed_large), `training` (epochs 100, lr 0.0002, weight_decay 0.0, warmup_steps 5000, grad_accum_steps 4, gradient_clip_val 1.0, ema_decay 0.9999, mixed_precision none, save_every 10, sample_every 5), `sampling` (batch_size 256, strip_steps [1000,800,600,400,200,100,50,0]), `evaluation` (num_samples 5000, batch_size 64, is_splits 10), `verify` (memory_budget_mib 3072)
- [X] T007 [P] Create `tests/conftest.py` with fixtures: `device` (cuda if available else cpu), `tiny_config` (a valid config dict copied from the baseline with `model.channels: [32, 64]`, `attention_resolutions: [16]`, `num_res_blocks: 1`, `num_groups: 8`, `time_embed_dim: 64`, `diffusion.timesteps: 50`, `data.batch_size: 4`, `training.grad_accum_steps: 2`, `training.warmup_steps: 2`, `experiment.output_dir` set to a `tmp_path`), `synthetic_image_batch` (`[4, 3, 32, 32]` uniform in [-1, 1], seeded), and `fake_cifar_loaders` (in-memory `TensorDataset` loaders of 16/8/8 random images with labels, so no test downloads CIFAR-10)

**Checkpoint**: `pip install -e ".[dev]"` succeeds, `pytest --collect-only` runs, and `validate_config(load_config("configs/cifar10_baseline.yaml"))` returns without error.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Contracts, registries, schedules, and the data pipeline every story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Tests for Foundational ⚠️

- [X] T008 [P] Write `tests/unit/test_foundations.py`: `DiffusionOutput`, `LossOutput`, `SamplingOutput` construct with the fields of data-model.md §2; the base classes in `src/models/base.py` cannot be instantiated; `validate_config` accepts the baseline YAML and `tiny_config`; it raises `ConfigError` for an unknown key, for `attention_resolutions: [32]` with `image_size: 32`, for `channels: [64, 100]` with `num_groups: 32`, for `beta_start >= beta_end`, and for `evaluation.num_samples: 5001`; `apply_overrides` sets dotted keys without mutating the input
- [X] T009 [P] Write `tests/unit/test_registry.py`: registering a non-subclass raises `TypeError`; `get_schedule("missing")` raises `KeyError` whose message lists available names; `"linear"`, `"cosine"`, `"unet"`, `"gaussian"` resolve after import; the `"unet"` and `"gaussian"` assertions go in a separate test marked `@pytest.mark.xfail(strict=False)` until T024/T025 land, and the marker is removed in T029
- [X] T010 [P] Write `tests/unit/test_schedules.py`: linear β has length T, `β[0]==1e-4`, `β[-1]==0.02` (atol 1e-12, float64); cosine β ≤ 0.999 and > 0; for both, `alphas_cumprod` strictly decreasing, `alphas_cumprod[-1] < 1e-3` for T=1000 (linear) and `< 1e-2` (cosine), `alphas_cumprod[0] > 0.99`
- [X] T011 [P] Write `tests/unit/test_data.py` using monkeypatched `torchvision.datasets.CIFAR10` (a fake dataset of 100 random uint8 images) so nothing is downloaded: train/val sizes are 90/10 for `val_split=0.1`; the split is identical for the same seed and different for another seed; train and val index sets are disjoint; **split parity with the VAE (ADR 0001 D7)**: the train/val indices equal `torch.utils.data.random_split(range(N), [N−n_val, n_val], generator=torch.Generator().manual_seed(seed))` for the same seed; batches are in [-1, 1]; validation and test transforms contain no random flip; `unnormalize` maps -1→0 and 1→1

### Implementation for Foundational

- [X] T012 [P] Implement `src/models/types.py` with `@dataclass` `DiffusionOutput(x_t: Tensor, t: Tensor, noise: Tensor, predicted_noise: Tensor, extra: dict[str, Tensor] = field(default_factory=dict))`, `LossOutput(loss: Tensor, metrics: dict[str, float] = field(default_factory=dict))`, `SamplingOutput(samples: Tensor, trajectory: Tensor | None, trajectory_steps: list[int], seconds: float)` with the shapes documented in data-model.md §2 (`t` int64 in `[0, T-1]`; `samples` clamped to [-1, 1]; `trajectory` `[F, N, C, H, W]`)
- [X] T013 [P] Implement `src/models/base.py` with ABCs `BaseNoiseSchedule` (attribute `timesteps: int`, abstract `betas() -> Tensor` returning float64 `[T]` in (0, 1)), `BaseDenoiser(nn.Module, ABC)` (abstract `forward(x_t, t) -> Tensor`), `BaseDiffusion(nn.Module, ABC)` (abstract `q_sample`, `forward`, `compute_loss`, `p_sample`, `sample`) with the exact signatures in `contracts/component-interfaces.md` §1
- [X] T014 Implement `src/models/registry.py` per `contracts/component-interfaces.md` §2: `SCHEDULE_REGISTRY`, `DENOISER_REGISTRY`, `DIFFUSION_REGISTRY`; decorators `register_schedule`, `register_denoiser`, `register_diffusion` that raise `TypeError` for non-subclasses; getters that call `_ensure_default_components()` (lazy-import `src.models.schedules`, `src.models.unet`, `src.models.gaussian_diffusion`, ignoring `ImportError` only for modules not yet implemented) and raise `KeyError("Schedule 'x' not found. Available schedules: [...]")`; and `build_diffusion_from_config(config: dict) -> BaseDiffusion` that runs `validate_config`, builds the schedule from `config["diffusion"]`, the denoiser from `config["model"]` (dropping `name`), and the diffusion via `get_diffusion("gaussian")(schedule=..., denoiser=..., variance_type=...)`, storing `config` on the returned model as `model.config` (depends on T012, T013)
- [X] T015 Implement `src/models/schedules.py`: `@register_schedule("linear") class LinearSchedule(BaseNoiseSchedule)` with `__init__(timesteps=1000, beta_start=1e-4, beta_end=0.02, **_)` and `betas()` = `torch.linspace(beta_start, beta_end, T, dtype=torch.float64)`; `@register_schedule("cosine") class CosineSchedule(BaseNoiseSchedule)` with `__init__(timesteps=1000, cosine_s=0.008, **_)` computing `f(t)=cos²(((t/T)+s)/(1+s)·π/2)` for `t=0..T`, `ᾱ_t=f(t)/f(0)`, `β_t = min(1 − ᾱ_t/ᾱ_{t−1}, 0.999)` in float64 (research §2) (depends on T014)
- [X] T016 (port) Implement `src/data/cifar10.py` from `Hands-on VAE/src/data/cifar10.py`: keep `get_cifar10_transforms()` (`ToTensor` + `Normalize((0.5,)*3, (0.5,)*3)`), `unnormalize(t) = (t*0.5+0.5).clamp(0,1)`, and the bounds-checking collate (rename `vae_collate_fn` → `ddpm_collate_fn`); add `get_cifar10_train_transforms(random_flip: bool)` that prepends `RandomHorizontalFlip()` when true. **Change**: build two `CIFAR10(train=True)` instances (flip transform for training, plain transform for validation) and split *indices* with `perm = torch.randperm(50000, generator=torch.Generator().manual_seed(seed))`, train = `perm[:45000]`, val = `perm[45000:]` (generally `perm[:N−n_val]` / `perm[N−n_val:]` with `n_val = int(N·val_split)`), as `Subset`s so validation is never flipped. This is exactly the permutation `random_split` draws for the same seed and lengths (verified 2026-10-03), so the partition is identical to `Hands-on VAE` (ADR 0001 D7, Constitution III); test set uses the plain transform. Expose `get_cifar10_datasets(data_dir, val_split, seed, random_flip, download)` and `get_cifar10_dataloaders(data_dir, batch_size, val_split, num_workers, seed, random_flip, download, pin_memory=None)` (train: shuffle, `drop_last=True`; val/test: no shuffle), plus `get_dataloaders_from_config(config, batch_size=None, data_dir=None)`. On download failure raise `RuntimeError("CIFAR-10 download failed ... (offline?)")`

**Checkpoint**: `pytest tests/unit/test_foundations.py tests/unit/test_registry.py tests/unit/test_schedules.py tests/unit/test_data.py -v` passes.

---

## Phase 3: User Story 1 - Verify the Diffusion Model Before Training (Priority: P1)

**Goal**: Build the U-Net and the forward/loss half of Gaussian diffusion, and expose `ddpm verify` as the pre-training gate (FR-001–FR-005, FR-008–FR-011, SC-001, SC-002).

**Independent Test**: `ddpm verify --config configs/cifar10_baseline.yaml` prints 7/7 ✓, ~16.06 M parameters, measured GPU memory ≤ 3072 MiB, exits 0 in < 2 minutes; a copy of the config with `diffusion.schedule: cosine` also passes.

### Tests for User Story 1 ⚠️

- [X] T017 [P] [US1] Write `tests/unit/test_shapes.py`: `SinusoidalTimestepEmbedding(64)` maps `t` `[B]` → `[B, 64]`, distinct timesteps give distinct rows; `TimestepMLP` → `[B, 256]`; `ResBlock(64→128)` preserves H×W and outputs 128 channels; `SpatialSelfAttention(128, 4)` preserves shape and its SDPA output matches an explicit einsum/softmax reference within `atol=1e-5`; `UNet` from the baseline config maps `[2,3,32,32]` + `t` `[2]` → `[2,3,32,32]` and has 16.06 M ± 0.05 M parameters; `UNet` construction with attention at 32×32 is rejected by `validate_config`
- [X] T018 [P] [US1] Write `tests/unit/test_forward_process.py` (tiny and baseline schedules): buffers `sqrt_alphas_cumprod² + sqrt_one_minus_alphas_cumprod² == 1` (atol 1e-6); `q_sample(x0, t, noise)` equals `√ᾱ_t·x0 + √(1−ᾱ_t)·noise` exactly; at `t = T−1` on a 512-image batch of constant images the output has |mean| < 0.05 and |std − 1| < 0.05; posterior variance `β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t)` with `β̃_0 = 0` handled by `posterior_log_variance_clipped` using `β̃_1`; `posterior_mean_coef1/2` match their closed forms
- [X] T019 [P] [US1] Write `tests/unit/test_loss.py`: `forward(x0, generator)` returns a `DiffusionOutput` with `t` in `[0, T−1]` and noise shape = input shape; `compute_loss` returns a finite scalar equal to `F.mse_loss(predicted_noise, noise)`; after `loss.backward()` on `tiny_config`, every parameter with `requires_grad` has a finite, non-zero gradient norm; the same generator seed gives the same `t` and noise
- [X] T020 [P] [US1] Write `tests/unit/test_memory_budget.py` (module-level `pytest.mark.skipif(not torch.cuda.is_available())`): in a **subprocess per scenario** (`subprocess.run([sys.executable, "-c", ...])`, so allocator state is fresh), using the baseline config, assert `measured_total_mib` ≤ 3072 for (a) one full training step at batch 32 with 4 accumulation micro-steps, AdamW and an EMA copy on GPU, (b) 5 reverse steps of sampling at batch 256 under `torch.no_grad()`, (c) Inception-v3 feature extraction at batch 64 (use `inception_v3(weights=None, aux_logits=True, init_weights=False)` to avoid a download; same size as the pretrained model)

### Implementation for User Story 1

- [X] T021 [P] [US1] Implement `src/models/embeddings.py`: `SinusoidalTimestepEmbedding(dim)` computing `half=dim//2`, `freqs = exp(−ln(10000)·arange(half)/half)`, `emb = cat(sin(t·freqs), cos(t·freqs))` with `t` as float; `TimestepMLP(in_dim=64, out_dim=256)` = `Linear → SiLU → Linear` (research §4)
- [X] T022 [P] [US1] Implement `src/models/attention.py`: `SpatialSelfAttention(channels, num_heads=4, num_groups=32)` = `GroupNorm → Conv1x1(C→3C)` reshaped to `[B, heads, HW, C/heads]` Q/K/V → `F.scaled_dot_product_attention(q, k, v)` → reshape → `Conv1x1(C→C)` (zero-initialized) → residual add; assert `channels % num_heads == 0`; include a module-level `reference_attention(q, k, v)` einsum implementation used only by tests (research §5)
- [X] T023 [P] [US1] Implement `src/models/resnet.py`: `ResBlock(in_ch, out_ch, time_dim, dropout=0.1, num_groups=32)` = `GroupNorm → SiLU → Conv3x3` + `Linear(SiLU(time_emb))` broadcast as a per-channel bias → `GroupNorm → SiLU → Dropout → Conv3x3` (zero-initialized weights and bias) + skip (`Conv1x1` if `in_ch != out_ch` else identity); `Downsample(ch)` = `Conv3x3(stride=2, padding=1)`; `Upsample(ch)` = nearest ×2 + `Conv3x3`
- [X] T024 [US1] Implement `src/models/unet.py`: `@register_denoiser("unet") class UNet(BaseDenoiser)` with `__init__(in_channels=3, out_channels=3, image_size=32, channels=(64,128,256), num_res_blocks=2, attention_resolutions=(16,8), num_heads=4, num_groups=32, dropout=0.1, time_embed_dim=256, **_)`. Layout (must match the probe in `research/gpu_memory_probe.py`, which measured 16.06 M params): input `Conv3x3(in→channels[0])`; down path per level `num_res_blocks` × [`ResBlock` + `SpatialSelfAttention` if current resolution ∈ `attention_resolutions`], pushing every output (including the input conv and each `Downsample`) onto a skip stack, `Downsample` between levels; middle `ResBlock → SpatialSelfAttention → ResBlock`; up path per level (reversed) `num_res_blocks + 1` × [concat(skip.pop()) → `ResBlock` + attention if resolution matches], `Upsample` between levels; head `GroupNorm → SiLU → Conv3x3(channels[0]→out)` zero-initialized; time path `SinusoidalTimestepEmbedding(channels[0]) → TimestepMLP(channels[0], time_embed_dim)`. Expose `num_parameters()` (depends on T021–T023)
- [X] T025 [US1] Implement the forward/loss half of `src/models/gaussian_diffusion.py`: `@register_diffusion("gaussian") class GaussianDiffusion(BaseDiffusion)` with `__init__(schedule, denoiser, variance_type="fixed_large")`. Compute in float64, then register as **non-persistent** float32 buffers (research §2): `betas`, `alphas`, `alphas_cumprod`, `alphas_cumprod_prev` (prepend 1.0), `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `sqrt_recip_alphas`, `posterior_variance`, `posterior_log_variance_clipped` (log of `posterior_variance` with index 0 replaced by index 1), `posterior_mean_coef1 = β_t·√ᾱ_{t−1}/(1−ᾱ_t)`, `posterior_mean_coef2 = (1−ᾱ_{t−1})·√α_t/(1−ᾱ_t)`. Add `_extract(buf, t, shape)` (gather → `[B,1,1,1]`), `q_sample(x0, t, noise=None)` (FR-004), `forward(x0, generator=None)` sampling `t ~ U{0..T−1}` and `noise ~ N(0,I)` with the generator, returning `DiffusionOutput`, and `compute_loss(output)` returning `LossOutput(loss=mse, metrics={"mse": float})` (FR-005). Leave `p_sample`/`sample` raising `NotImplementedError` until T045 (depends on T014, T015, T024)
- [X] T026 [US1] Implement `ddpm verify` and the app skeleton in `src/cli/main.py`: `app = typer.Typer(name="ddpm", help="Modular From-Scratch Denoising Diffusion Probabilistic Model CLI for CIFAR-10.", add_completion=False)`; import every component module at the top so decorators register; `@app.callback()` with eager `--version` printing `Hands-on DDPM version: 0.1.0`; `verify(config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True), skip_memory: bool = typer.Option(False, "--skip-memory"))` running the 7 checks of `contracts/cli.md` §2.1 in order (schema; schedule monotonic with ᾱ_T < 1e-3; forward statistics at t=T−1 within 0.05; U-Net output shape; finite loss and non-zero gradients for all parameters; 10-step reverse pass finite and in [-1, 1] after clamp — use a local loop over `t = T−1 … T−10` with the posterior formulas until T045 lands, then switch to `p_sample`; one training step at the configured batch with `measured_total_mib` ≤ `verify.memory_budget_mib`, skipped with a warning on CPU or with `--skip-memory`). Print `✓`/`✗` per check, the parameter count, measured MiB and total elapsed seconds (SC-001 target < 120 s on GPU, printed as a note, not a failing check); exit with `typer.Exit(code=1)` on any failure. Wrap unexpected exceptions: `ConfigError`/`FileNotFoundError` → exit 1, `torch.cuda.OutOfMemoryError`/`RuntimeError` → exit 2 (depends on T004, T005, T025)
- [X] T027 [US1] Add a shared helper `_resolve_device(config) -> torch.device` (`auto` → cuda if available else cpu, printing a yellow warning on CPU fallback) and `_handle_errors` decorator implementing the exit-code policy (0 / 1 config-input / 2 runtime) of `contracts/cli.md` §1 in `src/cli/main.py`, and apply it to `verify`
- [X] T028 [US1] Run `ddpm verify --config configs/cifar10_baseline.yaml` and a cosine variant (`/tmp/cosine.yaml` from `apply_overrides`), record the printed parameter count, MiB and elapsed seconds (SC-001: < 120 s) in `specs/001-create-ddpm/research/gpu_memory_results.md` under a new "Verify gate (implementation)" section
- [X] T029 [US1] Remove the temporary `xfail` markers added in T009 from `tests/unit/test_registry.py` and confirm `pytest tests/unit -v` passes

**Checkpoint**: User Story 1 is fully functional: the verify gate passes on both schedules within 3 GB.

**Implementation notes (US1, 2026-10-03)**:
- *Gradient-flow check on a perturbed copy*: zero-initialized output convolutions (research §4) make the untrained network an identity mapping, so upstream layers receive exactly zero gradient at initialization. `test_loss.py` and `ddpm verify` therefore check gradient flow on a copy whose all-zero parameters are filled with N(0, 1e-3) noise (`perturb_zero_init_` in `src/models/gaussian_diffusion.py`); the real model keeps its zero initialization.
- *`p_sample` implemented early*: T045's `p_sample` was implemented during T026 so the verify reverse-pass check uses the real Algorithm 2 step instead of a temporary loop. T045 remains the owner of its tests (T043).
- *Config key `diffusion.name`*: added in T005 (default `gaussian`) so the registry selects the diffusion process the same way `model.name` selects the network; recorded in data-model.md §3.

---

## Phase 4: User Story 2 - Train the Baseline DDPM on CIFAR-10 (Priority: P1) 🎯 MVP

**Goal**: EMA, checkpoints, resumable training with gradient accumulation, `ddpm train`, and the Colab training path (FR-015–FR-018b, FR-022, SC-003).

**Independent Test**: Quickstart §4: `ddpm train --epochs 2 --output-dir artifacts/runs/smoke`, then `--epochs 3 --resume .../latest.pt` produces `latest.pt`, `best_checkpoint.pt`, `final_checkpoint.pt`, `metrics.json` with **3** records, `loss_curve.png`, `train.log`, `samples/epoch_NNN.png`, `resolved_config.yaml`; the resumed run starts at epoch 3 with `global_step` continuing.

### Tests for User Story 2 ⚠️

- [X] T030 [P] [US2] Write `tests/unit/test_ema.py`: after one `update` with decay 0.9, each shadow parameter equals `0.9·old + 0.1·new`; buffers are copied, not averaged; `copy_to(model)` overwrites model weights; `state_dict()`/`load_state_dict()` round-trips `decay`, `num_updates` and shadow tensors; the EMA model has `requires_grad=False`
- [X] T031 [P] [US2] Write `tests/unit/test_checkpoint.py`: `save_checkpoint` writes every key of data-model.md §4 (`format_version`, `epoch`, `global_step`, `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `best_val_loss`, `history`, `rng_state`, `config`, `metrics`, `provenance` with `torch_version`, `cuda_available`, `device_name`, `timestamp`, `seed`, `git_commit`); `restore_model_from_checkpoint(path, weights="ema")` loads EMA weights and `weights="raw"` loads raw weights (they differ after one update); `load_checkpoint` on a non-checkpoint file raises `ValueError`
- [X] T032 [P] [US2] Write `tests/unit/test_trainer.py` on CPU with `tiny_config` and `fake_cifar_loaders`: (a) 2 epochs produce the files listed in the Independent Test and `metrics.json` records with keys `epoch, global_step, lr, train_loss, val_loss, val_loss_ema, epoch_seconds, peak_memory_mib`; (b) `global_step` counts optimizer steps = `floor(len(train_loader)/grad_accum_steps)` per epoch; (c) gradient-accumulation equivalence: one optimizer step with batch 4 × accum 2 gives the same parameter update (atol 1e-5, dropout 0) as batch 8 × accum 1 on the same data, timesteps and noise; (d) resuming from `latest.pt` after epoch 1 and training to epoch 2 yields a `history` of 2 records and the same `global_step` as an uninterrupted run; (e) a monkeypatched loss returning NaN makes `train()` raise `NonFiniteLossError` and leaves `best_checkpoint.pt` unchanged; (f) resuming with a config whose `model.channels` differ raises `ConfigError` listing the differing keys; (g) **determinism** (spec US2 scenario 5): two fresh 1-epoch runs with the same seed and config produce identical `train_loss`/`val_loss` values in `metrics.json`

### Implementation for User Story 2

- [X] T033 [P] [US2] Implement `src/training/ema.py`: `class EMA` with `__init__(model: nn.Module, decay: float = 0.9999)` storing a `copy.deepcopy` of the model (eval mode, `requires_grad_(False)`, same device), `num_updates`; `@torch.no_grad() update(model)` doing `shadow.lerp_(param, 1 − decay)` for floating-point parameters and `copy_` for buffers; `copy_to(model)`; `module` property; `state_dict()` → `{"decay", "num_updates", "shadow": shadow.state_dict()}`; `load_state_dict(state)` (research §6)
- [X] T034 [P] [US2] Implement `src/training/checkpoint.py` (port + extend `Hands-on VAE/src/training/checkpoint.py`): `save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, history, config, metrics, seed) -> Path` writing every key of data-model.md §4 (`rng_state` = Python `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` if CUDA; `provenance.git_commit` from `git rev-parse HEAD` or `None`), saving atomically (write `path.with_suffix(".tmp")` then `os.replace`); `load_checkpoint(path, map_location="cpu")` with `weights_only=False` and a `ValueError` if `model_state_dict` is missing; `restore_model_from_checkpoint(path, map_location="cpu", weights="ema") -> (model, ckpt)` rebuilding via `build_diffusion_from_config(ckpt["config"])`, where `model_state_dict` is the **full** `GaussianDiffusion.state_dict()` (keys prefixed `denoiser.`; schedule buffers are non-persistent so they are rebuilt, not loaded) and `ema_state_dict["shadow"]` is the **denoiser-only** state dict of the EMA copy: for `"raw"` call `model.load_state_dict(ckpt["model_state_dict"])`, for `"ema"` call `model.denoiser.load_state_dict(ckpt["ema_state_dict"]["shadow"])`; return the model in `eval()` mode; `restore_rng_state(state)`
- [X] T035 [US2] Implement `src/training/trainer.py` class `DDPMTrainer` per `contracts/component-interfaces.md` §3: `__init__(config, model=None, train_loader=None, val_loader=None, device=None)` calls `seed_everything`, resolves the device, creates `output_dir` and `output_dir/"samples"`, logger `get_logger("ddpm.trainer", log_file=output_dir/"train.log")`, writes `resolved_config.yaml`, builds the model with `build_diffusion_from_config` if not given, `EMA(model.denoiser, ema_decay)`, `torch.optim.AdamW(lr, weight_decay)`, a `LambdaLR` warmup `min(1, (step+1)/warmup_steps)` stepped per optimizer step (constant afterwards), and a `torch.amp.GradScaler` enabled only for `mixed_precision == "fp16"`; state `history=[]`, `best_val_loss=inf`, `global_step=0`, `training_seconds=0.0` (depends on T033, T034)
- [X] T036 [US2] Implement `DDPMTrainer.train_epoch(epoch)` in `src/training/trainer.py`: tqdm bar `"Epoch 001/100 [Train]"`; for each micro-batch compute the loss under `torch.autocast` when `mixed_precision` is `bf16`/`fp16`, divide by `grad_accum_steps`, backward (via scaler when fp16); every `grad_accum_steps` micro-batches: unscale, `clip_grad_norm_(gradient_clip_val)`, optimizer step, scheduler step, `zero_grad(set_to_none=True)`, `ema.update`, `global_step += 1`; drop a trailing incomplete accumulation group; raise `NonFiniteLossError(epoch, step)` (defined in this module) on the first non-finite loss, with the message recommending fp32 when mixed precision is on (FR-018b); return `{"train_loss", "lr"}` averaged over micro-batches; track `peak_memory_mib` with `src/utils/memory.py` when on CUDA
- [X] T037 [US2] Implement `DDPMTrainer.validate(epoch, use_ema=True)` (`@torch.no_grad()`) in `src/training/trainer.py`: compute `L_simple` on the validation loader with a generator seeded by `experiment.seed` (identical timesteps/noise every epoch, so values are comparable), once with the raw model (`val_loss`) and once with EMA weights (`val_loss_ema`, built by swapping the EMA denoiser into the diffusion wrapper)
- [X] T038 [US2] Implement `DDPMTrainer.train(start_epoch=1)` and `resume_from_checkpoint(path) -> int` in `src/training/trainer.py`: per epoch run train → validate → append the record `{epoch, global_step, lr, train_loss, val_loss, val_loss_ema, epoch_seconds, peak_memory_mib}` → save `latest.pt` → if `val_loss_ema < best_val_loss` save `best_checkpoint.pt` and `best.pt` → if `epoch % save_every == 0` save `epoch_{epoch:03d}.pt` → if `epoch % sample_every == 0` save an 8×8 EMA grid to `samples/epoch_{epoch:03d}.png` (call `generate_sample_grid` once T047 exists; until then guard with `if hasattr(self.model, "sample")` and catch `NotImplementedError`) → rewrite `metrics.json` (JSON list, indent 2). After the loop save `final_checkpoint.pt` and call `_plot_loss_curves()` (matplotlib Agg, train/val/val-EMA loss vs epoch, `loss_curve.png`, dpi 150). On `NonFiniteLossError` or `torch.cuda.OutOfMemoryError`: log, keep `latest.pt` from the last finite epoch, never touch `best_checkpoint.pt`, re-raise. `resume_from_checkpoint` checks that `config["model"]` and `config["diffusion"]` match the checkpoint (else `ConfigError` with the differing keys), restores model, EMA, optimizer, scheduler, scaler, `global_step`, `best_val_loss`, **`history`** and RNG state, and returns `epoch + 1`. Return `{best_val_loss, final_epoch, history, best_checkpoint, latest_checkpoint}`
- [X] T039 [US2] Implement `ddpm train` in `src/cli/main.py` with the options of `contracts/cli.md` §2.2: `-c/--config` (required), `-r/--resume`, `-s/--seed`, `--epochs`, `--batch-size`, `--grad-accum`, `--output-dir`, `--mixed-precision` (`none`/`bf16`/`fp16`, same values as the config). Apply overrides with `apply_overrides` **before** `validate_config`, build loaders with `get_dataloaders_from_config`, construct `DDPMTrainer`, call `resume_from_checkpoint` when `--resume` is given (print `Resuming training from checkpoint: <path>`), then `train(start_epoch)`. Print a summary tree (best EMA val loss, final epoch, paths). Exit 2 with a message suggesting `--batch-size 16 --grad-accum 8` on OOM, or fp32 on NaN
- [X] T040 [P] [US2] Create `configs/cifar10_colab.yaml`: identical to `configs/cifar10_baseline.yaml` except `experiment.name: cifar10_colab`, `data.batch_size: 128`, `data.num_workers: 2`, `training.grad_accum_steps: 1` (same effective batch 128); add a header comment explaining the equivalence (research §13)
- [X] T041 [US2] Create `notebooks/ddpm_colab_training.ipynb` (valid nbformat 4 JSON with a `"colab"` metadata block and `"accelerator": "GPU"`), cells in order: (1) markdown title + instructions (Runtime → T4 GPU, add `GH_TOKEN` Colab Secret, Run all, re-run after disconnects); (2) GPU check printing `torch.cuda.get_device_name(0)` and total memory, failing loudly without a GPU; (3) `from google.colab import drive; drive.mount("/content/drive")`; define `RUN_DIR = "/content/drive/MyDrive/hands-on-ddpm/runs/cifar10_baseline"`; (4) get the code: if `/content/Hands-on-DDPM` is missing, read `GH_TOKEN` via `google.colab.userdata.get`, `git clone https://$GH_TOKEN@github.com/Abd-Elfattah5/Hands-on-DDPM.git` (never print the token); fallback: unzip `/content/drive/MyDrive/hands-on-ddpm/Hands-on-DDPM.zip`; optional `BRANCH` variable for `git checkout`; (5) `pip install -e .` (Colab's preinstalled torch is used); (6) `ddpm verify --config configs/cifar10_colab.yaml --skip-memory` (memory check is for the 3 GB local budget); (7) train cell: build the command `ddpm train --config configs/cifar10_colab.yaml --output-dir $RUN_DIR`, append `--resume $RUN_DIR/latest.pt` if that file exists, run it with `!`; (8) optional `ddpm denoise-strip` and `ddpm benchmark` cells writing under `$RUN_DIR/../eval`, guarded by a `RUN_BENCHMARK = False` flag; (9) display `loss_curve.png` and the latest `samples/epoch_NNN.png`. No model, loss or training-loop code in any cell (Constitution Prohibited Patterns)
- [X] T042 [US2] Run quickstart §4 (2-epoch smoke train + resume to epoch 3) locally on CIFAR-10 and confirm the Independent Test; record per-epoch seconds and `peak_memory_mib` from `metrics.json` in `specs/001-create-ddpm/research/gpu_memory_results.md` ("Smoke training (implementation)")

**Checkpoint**: Training works end to end locally, resumes without losing history, and the Colab notebook drives the same CLI. This is the MVP.

**Implementation notes (US2, 2026-10-03)**:
- *Generation pulled forward*: `p_sample`, `sample`, `generate_sample_grid` and `render_denoise_strip` (T045–T047) were implemented with US2, so T038 calls `generate_sample_grid` directly (no temporary guard) and T050 reduced to the test assertion already in `test_trainer.py`.
- *Measured epoch time*: ~336–360 s per epoch including raw + EMA validation, so the full local run takes ≈ 10 h, not the 8.5 h planning estimate (see research/gpu_memory_results.md).

---

## Phase 5: User Story 3 - Generate Images and Visualize the Reverse Process (Priority: P2)

**Goal**: Algorithm 2 sampling with deterministic seeds, sample grids, denoising strips, `ddpm sample` and `ddpm denoise-strip` (FR-006, FR-007, FR-019–FR-021, SC-006).

**Independent Test**: Quickstart §5: two `ddpm sample -n 16 --seed 7` runs on the smoke checkpoint produce byte-identical PNGs; `ddpm denoise-strip --num-images 4` writes a strip with 8 labeled columns `t=1000 … t=0`; `<out_stem>_timing.json` exists.

### Tests for User Story 3 ⚠️

- [X] T043 [P] [US3] Write `tests/unit/test_sampling.py` with `tiny_config` (T=50) on CPU: `sample(4, device, generator)` returns `SamplingOutput` with `samples` `[4,3,32,32]` in [-1, 1] and finite; two calls with generators of the same seed give identical tensors, different seeds differ; `p_sample` at `t=0` adds no noise (monkeypatch `torch.randn` to return large values and assert the output equals the mean); `fixed_large` uses σ²=β_t and `fixed_small` uses σ²=β̃_t (check via a zero denoiser and known x_t); `trajectory_steps=[50, 25, 0]` returns `trajectory` `[3, 4, 3, 32, 32]` whose first frame is the initial noise `x_T` (label T = 50) and whose last frame equals `samples` (label 0); `batch_size=2` for 4 samples gives the same result as one batch of 4 for the same seed
- [X] T044 [P] [US3] Write `tests/unit/test_visualizer.py` (sampling part) using `tmp_path`: `generate_sample_grid` writes a PNG of size `(sqrt(n)·(32·upscale + 2·padding))` per side for n=16, upscale 2; `render_denoise_strip` writes a PNG whose width corresponds to 8 columns plus label margin; `upscale_tensor` output is in [0, 1]

### Implementation for User Story 3

- [X] T045 [US3] Implement `p_sample(x_t, t: int, generator=None)` in `src/models/gaussian_diffusion.py`: `eps = denoiser(x_t, full((B,), t))`; `mean = sqrt_recip_alphas[t]·(x_t − betas[t]/sqrt_one_minus_alphas_cumprod[t]·eps)`; `σ² = betas[t]` (`fixed_large`) or `posterior_variance[t]` (`fixed_small`); return `mean + σ·z` with `z = torch.randn(..., generator=generator)` for `t > 0`, else `mean` (FR-006, FR-007, research §8)
- [X] T046 [US3] Implement `sample(num_samples, device, generator=None, trajectory_steps=None, batch_size=None, on_batch=None)` (`@torch.no_grad()`) in `src/models/gaussian_diffusion.py`: process in chunks of `batch_size` (default all); each chunk starts from `x_T = torch.randn(..., generator=generator, device=device)` and loops `t = T−1 … 0`; when `trajectory_steps` (paper labels) is given, capture frames with this single rule: **label `T` (1000) = the initial `x_T` before any reverse step; label `k` for `0 ≤ k < T` = the state after the reverse step at zero-based index `k`** (so label 0 is the final output, clamped to [-1, 1]); a frame is captured when its label is in `trajectory_steps`; clamp the final output to [-1, 1]; call `on_batch(index, samples_chunk)` after each chunk so callers can write to disk; return `SamplingOutput(samples, trajectory, trajectory_steps, seconds)`. Then switch the reverse-pass check in `ddpm verify` (T026) to use `p_sample` (depends on T045)
- [X] T047 [P] [US3] Implement the generation part of `src/evaluation/visualizer.py` (port `upscale_tensor` and `generate_sample_grid` from `Hands-on VAE/src/evaluation/visualizer.py`): `upscale_tensor(t, factor=4)` bicubic + clamp [0, 1]; `generate_sample_grid(model, num_samples=64, device=None, out_path="artifacts/samples/sample_grid.png", seed=None, upscale=4, batch_size=256) -> Path` using `make_generator(seed, device)`, `unnormalize`, `vutils.save_image(..., nrow=isqrt(n), padding=2*upscale, normalize=False)`; `render_denoise_strip(trajectory, steps, out_path, upscale=4) -> Path` arranging rows = images and columns = steps, with a top label row `t=1000 … t=0` drawn with PIL `ImageDraw` (default font)
- [X] T048 [US3] Implement `ddpm sample` in `src/cli/main.py` per `contracts/cli.md` §2.4 (without `--nearest`, which is added in T058): `-k/--checkpoint` (required), `-n/--num-samples` 64, `-s/--seed` (default config seed), `-b/--batch-size` (default `sampling.batch_size`), `--weights` `ema|raw` (default ema), `-o/--out` `artifacts/samples/sample_grid.png`, `--upscale` 4, `--save-individual`. Load with `restore_model_from_checkpoint(weights=...)`, sample with `make_generator(seed, device)` and `on_batch` writing individual PNGs to `<out_stem>/NNNNN.png` when `--save-individual`, save the grid, write `<out_stem>_timing.json` (`num_samples`, `seed`, `sampling_seconds_total`, `sampling_seconds_per_image`, `device_name`); print a summary tree
- [X] T049 [US3] Implement `ddpm denoise-strip` in `src/cli/main.py` per `contracts/cli.md` §2.5: `-k/--checkpoint`, `--num-images` 8, `--steps` (comma-separated, default `sampling.strip_steps`, validated descending within [0, T]), `-s/--seed`, `--weights`, `-o/--out` `artifacts/strips/denoise_strip.png`, `--upscale` 4; call `sample(..., trajectory_steps=steps)` then `render_denoise_strip`
- [X] T050 [US3] Hook the epoch sample grid into training: replace the guard in `DDPMTrainer.train` (T038) with a direct call to `generate_sample_grid(ema_model, 64, seed=experiment.seed, out_path=output_dir/"samples"/f"epoch_{epoch:03d}.png")` in `src/training/trainer.py`, and assert in `tests/unit/test_trainer.py` that `samples/epoch_002.png` exists when `sample_every=2`
- [X] T051 [US3] Run quickstart §5 on the smoke checkpoint (`cmp` of two seeded runs prints `IDENTICAL`; strip rendered) and note the measured `sampling_seconds_per_image` at batch 256 in `specs/001-create-ddpm/research/gpu_memory_results.md` ("Sampling (implementation)")

**Checkpoint**: Seeded sampling is deterministic; grids and denoising strips render from any checkpoint.

---

## Phase 6: User Story 4 - Evaluate the DDPM Quantitatively and Qualitatively (Priority: P2)

**Goal**: Test loss, FID/IS with the exact VAE protocol, the nearest-neighbor memorization panel, `ddpm evaluate`, `ddpm benchmark`, and `sample --nearest` (FR-023–FR-027, SC-007, SC-010).

**Independent Test**: Quickstart §8 on the smoke checkpoint with `--num-samples 100`: `benchmark_metrics.json` contains every key of data-model.md §5; `eval_metrics.json` contains `test_loss`; `ddpm sample --nearest` writes `<out_stem>_nearest.png` and `<out_stem>_nearest.json`.

### Tests for User Story 4 ⚠️

- [X] T052 [P] [US4] (port) Copy `Hands-on VAE/tests/unit/test_metrics.py` to `tests/unit/test_metrics.py` (Fréchet distance 0 for identical statistics; known value for a mean shift; IS ≈ 1 for uniform predictions; IS high for diverse confident predictions) and add: `calculate_inception_score` with 5000 probs and `splits=10` uses splits of 500; `compute_fid_and_is` with a stub extractor (monkeypatch `InceptionFeatureExtractor` to return deterministic random features) and a tiny model returns keys `fid`, `inception_score_mean`, `inception_score_std`, `benchmark_samples`, `class_coverage` (with `distinct_top1_classes` ≤ 1000 and `marginal_entropy_nats` ≥ 0), writes sample batches to `sample_dir`, and with `reuse_samples=True` skips generation for batches already on disk
- [X] T053 [P] [US4] Extend `tests/unit/test_visualizer.py` with the nearest-neighbor panel: with a fake 50-image training tensor where image 7 is copied into the samples, `find_nearest_neighbors(samples, train_images, k=3)` returns index 7 with distance 0 as the first neighbor for that sample; `render_nearest_neighbors` writes a PNG with `rows × (1 + k)` tiles and a JSON with `indices` and `distances` per row
- [X] T054 [P] [US4] Write `tests/unit/test_evaluator.py`: `evaluate_model(model, loader, device, seed=42)` returns `test_loss` and `total_samples`, and two calls give identical `test_loss` (fixed-seed timesteps and noise)

### Implementation for User Story 4

- [X] T055 [P] [US4] (port) Implement `src/evaluation/metrics.py` from `Hands-on VAE/src/evaluation/metrics.py` with **unchanged maths**: `InceptionFeatureExtractor(device)` (torchvision `inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)`, input in [0,1] → bilinear resize to 299 with `align_corners=False` → `x*2−1` → manual forward to `avgpool`/`dropout` → features `[B,2048]`, `fc` + softmax → probs `[B,1000]`); `calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)` (scipy `sqrtm`, eps offset when non-finite, real part); `calculate_inception_score(probs, splits=10)`. Replace `compute_fid_and_is` with the DDPM signature `compute_fid_and_is(model, real_loader, num_samples=5000, batch_size=64, device=None, sample_batch_size=256, seed=42, sample_dir=None, reuse_samples=False) -> dict`: real features from the first `num_samples` test images (unnormalized); fake images generated with `model.sample(..., generator=make_generator(seed, device), batch_size=sample_batch_size, on_batch=...)` saving each batch as `sample_dir/batch_{i:04d}.pt` (uint8) and, when `reuse_samples`, loading existing batches instead of regenerating; features computed in chunks of `batch_size`; return `{"fid": round(fid, 2), "inception_score_mean", "inception_score_std", "benchmark_samples", "sampling_seconds_total", "sampling_seconds_per_image", "class_coverage"}` where `class_coverage = {"distinct_top1_classes": int, "marginal_entropy_nats": float, "top20": [[imagenet_index, count], ...]}` computed from the same fake-image probabilities (no extra forward passes; Constitution report gate "distribution coverage") (research §9)
- [X] T056 [P] [US4] Implement `src/evaluation/evaluator.py`: `@torch.no_grad() evaluate_model(model, data_loader, device, seed=42) -> dict` computing the sample-weighted mean `L_simple` over the loader using `model.forward(x0, generator=make_generator(seed, device))` and returning `{"test_loss", "total_samples"}`
- [X] T057 [US4] Implement the nearest-neighbor panel in `src/evaluation/visualizer.py` per `contracts/cli.md` §2.4.1: `load_train_images(data_dir) -> Tensor` (all 50,000 CIFAR-10 training images as float32 [0,1] `[50000,3,32,32]`, no augmentation); `find_nearest_neighbors(samples01, train01, k=3, chunk_size=5000) -> (indices [N,k], distances [N,k])` computing squared L2 on flattened pixels chunk by chunk on the samples' device (≈ 600 MB, within budget) and returning `sqrt` distances; `render_nearest_neighbors(samples01, train01, indices, distances, out_path, upscale=4) -> Path` writing a grid with one row per sample (generated image first, then k neighbors, a separator column of white pixels between them) and `<stem>.json` with `{"rows": [{"sample": i, "train_indices": [...], "distances": [...]}]}`
- [X] T058 [US4] Add `--nearest`, `--nearest-k` (default 3) and `--nearest-rows` (default 64, matching SC-010's "at least 64 samples") to `ddpm sample` in `src/cli/main.py`: after sampling, take the first `--nearest-rows` samples, call `load_train_images`, `find_nearest_neighbors`, `render_nearest_neighbors` → `<out_stem>_nearest.png` and `<out_stem>_nearest.json`; print the minimum distance found
- [X] T059 [US4] Implement `ddpm evaluate` in `src/cli/main.py` per `contracts/cli.md` §2.3: `-k/--checkpoint` (required), `--weights` (ema), `-b/--batch-size` 128, `-d/--data-dir`, `-o/--out` (default `<checkpoint_dir>/eval_metrics.json`); write JSON with `timestamp` (UTC ISO), `checkpoint`, `weights`, `seed`, `test_loss`, `total_samples`, `device_name`
- [X] T060 [US4] Implement `ddpm benchmark` in `src/cli/main.py` per `contracts/cli.md` §2.6: `-k/--checkpoint`, `-n/--num-samples` 5000, `-b/--batch-size` 64, `--sample-batch-size` 256, `-s/--seed`, `--weights`, `-o/--out` `artifacts/eval/benchmark_metrics.json`, `--reuse-samples DIR`. Steps: (1) `evaluate_model` on the test loader; (2–4) `compute_fid_and_is` with `sample_dir = --reuse-samples or artifacts/eval/samples_<seed>/`; (5) write JSON with every key of data-model.md §5: `timestamp`, `checkpoint`, `weights`, `seed`, `test_loss`, `total_samples`, `fid`, `inception_score_mean`, `inception_score_std`, `benchmark_samples`, `sampling_seconds_total`, `sampling_seconds_per_image`, `class_coverage`, `num_parameters`, `training_seconds` (sum of `epoch_seconds` from `metrics.json` next to the checkpoint, or `null`), `device_name`. Print the summary tree shown in `contracts/cli.md` §2.6. Never read VAE files
- [X] T061 [US4] Run quickstart §8 on the smoke checkpoint with `--num-samples 100` (and `sample --nearest`) to validate the pipeline end to end; confirm peak memory for the Inception step stays ≤ 3072 MiB

**Checkpoint**: All six CLI commands exist and every evaluation artifact is produced from any checkpoint.

---

## Phase 7: User Story 5 - Official Training, Results and Deliverables (Priority: P3)

**Goal**: Produce the reported results on Colab, benchmark them, and complete the README and report deliverables (FR-018a, FR-029–FR-031, SC-004, SC-005, SC-008, SC-009, SC-010).

**Independent Test**: Following only `README.md` on a clean clone reproduces setup, verify and the CLI workflow; `docs/reports/001-baseline-ddpm-report.md` contains every GenCV003 section and the manual DDPM vs. VAE table.

- [X] T062 [US5] Push the branch and run `notebooks/ddpm_colab_training.ipynb` on a Colab T4 for the full 100 epochs with `RUN_DIR` on Google Drive, resuming after any disconnect; afterwards copy `best_checkpoint.pt`, `final_checkpoint.pt`, `metrics.json`, `loss_curve.png` and `samples/` from Drive (`runs/cifar10_colab/`) to `artifacts/runs/cifar10_baseline/` locally (the README/quickstart commands use this path; code review R4 gave the Colab run its own directory on Drive). Record the measured T4 time per epoch in `specs/001-create-ddpm/research/gpu_memory_results.md` ("Colab T4 (measured)")
- [X] T063 [US5] Run on the final checkpoint: `ddpm evaluate`, `ddpm benchmark --num-samples 5000` (local T2000 or Colab; record which), `ddpm sample -n 64 --seed 42 --upscale 4 --out artifacts/samples/sample_grid_1024.png --nearest`, `ddpm denoise-strip --num-images 8 --seed 42`; check and record in `specs/001-create-ddpm/research/gpu_memory_results.md` ("Final results"): SC-003 (training loss falls ≥ 50% between the first and last record of `metrics.json`; no NaN/Inf), SC-004 (FID < 50), SC-005 (IS above the VAE values), SC-007 (`sampling_seconds_total` + scoring time of the 5,000-image benchmark < 3.5 h on the T2000; if the benchmark ran on Colab, also time it on the T2000 or note the deviation) and SC-010 (no near-duplicates among the 64 rows of the nearest-neighbor panel); if FID ≥ 50, continue training with `--resume` and `--epochs N` and record the change (spec Assumptions: Training budget)
- [X] T064 [US5] Update `README.md`: verify every command in Installation & Setup and the CLI Reproduction Guide against `ddpm <command> --help` and the real artifact paths; remove the "implementation in progress" status note; publish the reported `best_checkpoint.pt` as a GitHub Release asset (`gh release create v0.1.0-ddpm-baseline artifacts/runs/cifar10_baseline/best_checkpoint.pt --notes ...`, ADR 0001 D10) and add a "Reproduce from the published checkpoint" subsection with the download command and the `ddpm benchmark` call (SC-008); add a "Quantitative Benchmark Results (CIFAR-10)" section (FID, IS, test loss, seconds per image, parameters, training time, GPU used) populated from `artifacts/eval/benchmark_metrics.json`; state which GPU produced the results (FR-031)
- [X] T065 [US5] Write `docs/reports/001-baseline-ddpm-report.md` (GenCV003 Deliverable a, mirroring `Hands-on VAE/docs/reports/001-baseline-vae-report.md`): implementation steps; mathematical formulation (forward process, closed form, posterior, ε-parametrization, L_simple, Algorithm 2); experimental setup (hardware, effective batch, epochs, EMA); quantitative results from the benchmark JSON; qualitative assessment of diversity, realism and thematic consistency using the sample grid, denoising strip and nearest-neighbor panel; **computational complexity** (parameter count, 1,000 network evaluations per image, wall-clock seconds per image and per 5,000 images, training time, peak GPU memory, with the VAE's single forward pass for contrast); **distribution coverage** (the benchmark's `class_coverage` block: number of distinct Inception top-1 classes, entropy of the marginal class distribution p(y), and the 20 most frequent predicted classes over the 5,000 samples, plus a visual check of the 1024×1024 grid for all 10 CIFAR-10 object types, mode-collapse observations and the nearest-neighbor findings) (FR-029, Constitution quality gate 4); the **DDPM vs. baseline VAE vs. enhanced VAE table** with VAE values entered manually by the author (FID 169.02 / 181.00, IS 2.11 / 1.68 from `Hands-on VAE`); the main difference between VAE and DDPM and its trade-offs (FR-029, FR-030); conclusions
- [X] T066 [P] [US5] Append a "Results" section to `ARCHITECTURE_DEEP_DIVE.md` (measured FID/IS, sampling speed, training time, observations from the denoising strip) and update §9 with the measured Colab T4 timing
- [X] T067 [P] [US5] Update `HANDOFF.md` to the implemented state: completed phases, commands, artifact paths, measured numbers, remove the stale MNIST/ADR mentions, and a resume prompt for the next feature

**Checkpoint**: Both GenCV003 deliverables are complete and reproducible.

**Status (2026-10-04)**: Done. T062 ran 80 of 100 epochs on a Colab T4 (user decision); T063 reports raw weights because EMA had not converged (ADR 0001 D11, plan "Results-Phase Change Log"). Results: FID 39.69, IS 5.18 ± 0.14, benchmark ≈ 3.0–3.1 h. T067 refreshed with final numbers.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Contract drift checks, CLI tests, and final validation.

- [X] T068 [P] Write `tests/unit/test_cli.py` with `typer.testing.CliRunner`: `ddpm --help` lists exactly `verify`, `train`, `evaluate`, `sample`, `denoise-strip`, `benchmark`; `--version` prints `Hands-on DDPM version: 0.1.0`; `verify --config <tiny yaml> --skip-memory` exits 0; a config with an unknown key exits 1; a missing checkpoint for `sample` exits 1; each command's `--help` contains every option listed for it in `contracts/cli.md`
- [X] T069 Reconcile `specs/001-create-ddpm/contracts/cli.md`, `specs/001-create-ddpm/quickstart.md` and `README.md` with the implemented CLI (option names, defaults, artifact paths); change `contracts/cli.md` **Status** to `Completed`
- [X] T070 [P] Update `CONTEXT.md` with any term introduced during implementation (e.g., `NonFiniteLossError` behaviour belongs in docs, not the glossary — add only domain terms) and confirm no glossary term contradicts the code
- [X] T071 Run `pytest tests/ -v` (all pass; memory tests run on the T2000) and walk through every scenario in `specs/001-create-ddpm/quickstart.md`, fixing any discrepancy
- [X] T072 Mark every completed task `[X]` in `specs/001-create-ddpm/tasks.md`, commit, push `001-create-ddpm`, and open a pull request to `main` summarising results (then run graphify, per the user's plan)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; **blocks all user stories**.
- **US1 (Phase 3)**: Depends on Foundational. Builds the network and diffusion core that every later story uses.
- **US2 (Phase 4)**: Depends on US1 (needs `GaussianDiffusion.forward/compute_loss` and the CLI skeleton).
- **US3 (Phase 5)**: Depends on US1; needs a checkpoint from US2 only for its manual validation task (T051). T043–T047 can start right after Phase 3.
- **US4 (Phase 6)**: Depends on US3 (`sample` and `on_batch`); T052–T056 can start after Phase 3.
- **US5 (Phase 7)**: Depends on US2 (training) and US4 (benchmark).
- **Polish (Phase 8)**: Depends on all stories.

### User Story Dependencies

- **US1 (P1)**: Foundation only.
- **US2 (P1)**: US1.
- **US3 (P2)**: US1 (code), US2 (checkpoint for validation).
- **US4 (P2)**: US3.
- **US5 (P3)**: US2 + US4.

### Within Each User Story

- Tests are written first and fail before implementation (Constitution IV).
- Building blocks (embeddings, attention, resnet) before the U-Net; U-Net before diffusion; library code before CLI commands; CLI before the manual validation task.

### Parallel Opportunities

- Phase 1: T002, T003, T004, T006, T007 in parallel (T005 is the only longer task).
- Phase 2: all four test tasks (T008–T011) in parallel; T012 and T013 in parallel.
- Phase 3: tests T017–T020 in parallel; T021, T022, T023 in parallel.
- Phase 4: tests T030–T032 in parallel; T033, T034 and T040 in parallel.
- Phase 5/6: T043, T044, T052, T053, T054 can be written as soon as Phase 3 is done; T055 and T056 in parallel.
- Phase 7: T062 (hours of Colab training) runs while T064–T067 drafts are prepared; T066 and T067 in parallel.

---

## Parallel Example: User Story 1

```bash
# Tests first, together:
Task: "Write tests/unit/test_shapes.py (T017)"
Task: "Write tests/unit/test_forward_process.py (T018)"
Task: "Write tests/unit/test_loss.py (T019)"
Task: "Write tests/unit/test_memory_budget.py (T020)"

# Then the three building blocks, together:
Task: "Implement src/models/embeddings.py (T021)"
Task: "Implement src/models/attention.py (T022)"
Task: "Implement src/models/resnet.py (T023)"
```

## Parallel Example: User Story 2

```bash
Task: "Write tests/unit/test_ema.py (T030)"
Task: "Write tests/unit/test_checkpoint.py (T031)"
Task: "Write tests/unit/test_trainer.py (T032)"
Task: "Implement src/training/ema.py (T033)"
Task: "Implement src/training/checkpoint.py (T034)"
Task: "Create configs/cifar10_colab.yaml (T040)"
```

## Parallel Example: User Story 4

```bash
Task: "Port tests/unit/test_metrics.py (T052)"
Task: "Extend tests/unit/test_visualizer.py with nearest neighbors (T053)"
Task: "Write tests/unit/test_evaluator.py (T054)"
Task: "Port src/evaluation/metrics.py (T055)"
Task: "Implement src/evaluation/evaluator.py (T056)"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Phase 1: Setup.
2. Phase 2: Foundational (blocks everything).
3. Phase 3: US1 → **STOP and VALIDATE**: `ddpm verify` passes within 3 GB.
4. Phase 4: US2 → **STOP and VALIDATE**: smoke train + resume (quickstart §4). Start the long Colab training (T062) as early as here if desired; checkpoints remain usable by later phases because the checkpoint format is fixed in T034.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → verified model (gate passes).
3. US2 → trainable, resumable model; Colab path ready (MVP).
4. US3 → deterministic samples and denoising strips.
5. US4 → FID/IS, test loss, nearest-neighbor panel.
6. US5 → official results, README and report.
7. Polish → CLI tests, contract reconciliation, PR.

### Notes

- Commit after each task or logical group (e.g., `feat(models): add time-conditioned U-Net (T024)`).
- Each phase ends with a **Checkpoint**; do not continue past a failing checkpoint.
- Keep `contracts/cli.md` and `README.md` in step with any option change (avoids the documentation drift found in `Hands-on VAE`).
