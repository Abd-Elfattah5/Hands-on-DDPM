# Implementation Plan: From-Scratch Denoising Diffusion Probabilistic Model (DDPM)

**Branch**: `001-create-ddpm` | **Date**: 2026-10-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-create-ddpm/spec.md`

## Summary

Build an unconditional DDPM (Ho et al. 2020) from scratch on CIFAR-10 as Part 2 of GenCV003, mirroring
the module layout, registry pattern, CLI vocabulary, checkpoint convention and evaluation protocol of
the sibling `Hands-on VAE` repository. The pipeline has these parts:
- registered linear and cosine noise schedules (T = 1000);
- a 16.06 M-parameter time-conditioned U-Net [64, 128, 256] with attention at 16×16 and 8×8;
- L_simple training with EMA (0.9999) at an effective batch of 128 (32 × 4 locally);
- Algorithm 2 sampling, denoising strips and nearest-neighbor checks;
- FID/IS computed with the VAE's exact Inception protocol (5,000 vs 5,000, 10 IS splits);
- a benchmark JSON holding every DDPM value needed for the final, manually written VAE comparison.

Everything runs locally through the `ddpm` CLI within a measured 3 GB GPU budget, and the README
documents the local path. The official long training run uses a thin Colab notebook
(`notebooks/ddpm_colab_training.ipynb`) that only calls the same CLI, with checkpoints on Google
Drive and `--resume` across disconnects. Research decisions are in [research.md](./research.md).

## Technical Context

**Language/Version**: Python 3.12 (venv `DDPM/`)

**Primary Dependencies**: PyTorch 2.6.0+cu124, torchvision 0.21.0 (CIFAR-10, Inception-v3 weights for
metrics only), typer, PyYAML, numpy, scipy (`sqrtm` for FID), matplotlib, Pillow, tqdm, scikit-learn;
pytest (dev)

**Storage**: Local files: YAML configs; `.pt` checkpoints; JSON metrics; PNG artifacts under
`artifacts/` (gitignored); on Colab, the run directory is on Google Drive

**Testing**: pytest (`tests/unit/`, plus CUDA-only memory tests skipped without a GPU), and the
`ddpm verify` pre-training gate

**Target Platform**: Linux/WSL2 with an NVIDIA Quadro T2000 (4 GB, 3 GB budget) for development and
local training; Google Colab T4 (15 GB) for the official training run

**Project Type**: Single-project Python library + CLI (`ddpm`) + one driver notebook

**Performance Goals**:
- `verify` < 2 min (SC-001).
- Training ≈ 0.84 s per optimizer step on the T2000: ≈ 10 h for 100 epochs locally including
  validation (measured ~6 min/epoch, corrected from the 8.5 h planning estimate), ≈ 3.5–4 h
  estimated on a T4.
- 5,000-image benchmark < 3.5 h on the T2000 (SC-007).

**Constraints**:
- Total GPU memory per process ≤ 3072 MiB; measured 1,625 MiB at train 32 × 4, 2,061 MiB at sample
  batch 256, 2,269 MiB for Inception at batch 64.
- fp32 by default.
- No attention at 32×32.
- No pre-built diffusion libraries (Constitution I).
- Deterministic for a fixed seed on the same hardware.

**Scale/Scope**: CIFAR-10 (45k train / 5k val / 10k test); 1 model variant (unconditional baseline);
6 CLI commands; ~25 source modules; 2 configs; 1 notebook; 1 ADR; 1 technical report

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Verification Mechanism |
|---|---|---|---|
| I. From-Scratch Diffusion Modeling | Schedules, q(x_t\|x₀), posterior, ε-parametrization, L_simple, time embedding, U-Net and Algorithm 2 authored in-repo; no `diffusers`; pretrained weights only for Inception metrics | PASS | Code review; `pyproject.toml` has no diffusion libraries; unit tests check the formulas against hand-derived values |
| II. Deterministic Reproducibility | Single seed for all RNGs; versioned config, checkpoints with EMA, provenance; README reproduction commands | PASS | `seed_everything`; checkpoint `provenance`, `rng_state`, `config`; `resolved_config.yaml`; determinism test (same seed → identical samples) |
| III. Standardized Benchmarking & Cross-Model Comparison | CIFAR-10 with the same splits and preprocessing as the VAE; FID/IS with an identical protocol; qualitative grids/strips; side-by-side table with both VAEs | PASS | `metrics.py` ported verbatim; benchmark JSON stores all DDPM comparison values; VAE rows are added manually to the final report; metric unit tests ported from the VAE |
| IV. Test-Driven Tensor & Diffusion Integrity | Shape, schedule, forward-process, gradient-flow, stability and sampling-determinism tests before training | PASS | `tests/unit/*` (see structure); `ddpm verify` gate; Foundational phase tasks precede training tasks |
| V. CLI-Driven Pipeline & Artifact Observability | Every workflow through the CLI; machine-readable metrics; organized artifacts | PASS | `contracts/cli.md`; the notebook only calls the CLI; JSON metrics (the constitution's "JSON/CSV" is satisfied by JSON alone); `artifacts/{runs,samples,strips,eval}` |
| Tech Stack: Dependency isolation (`DDPM/`) | venv + `pyproject.toml` manifest | PASS | venv created; `pip install -e ".[dev]"` verified with CUDA |
| Tech Stack: Hardware budget | Defaults fit the T2000; no 32×32 attention; accumulation for a larger effective batch | PASS | GPU probe (`research/gpu_memory_results.md`); `verify` memory check; `test_memory_budget.py` |
| Tech Stack: Ecosystem consistency | Layout, CLI, config schema and reports mirror the VAE; deviations need an ADR | PASS | Every deviation is justified in `docs/adr/0001-deviations-from-hands-on-vae-conventions.md` (D1–D10), as the constitution requires; parity table in research §15 |
| Prohibited patterns | No training loop in an unversioned notebook | PASS | The Colab notebook is versioned (`!notebooks/*.ipynb`) and contains no model or training code, only CLI calls |
| Quality gates 1–5 | Pre-training gate, per-epoch tracking, evaluation gate, report (incl. computational complexity and distribution coverage), README | PASS | Plan phases P2/P3, P4, P6, P7 below; FR-029 and task T065 name both report items; `class_coverage` in the benchmark JSON supplies the coverage data |

**Post-design re-check (after Phase 1)**: PASS. The data model, contracts and quickstart introduce no
new dependencies or violations. The notebook is constrained to CLI calls, and the fused attention
kernel is a tensor primitive (research §5).

## Project Structure

### Documentation (this feature)

```text
specs/001-create-ddpm/
├── spec.md                      # Feature specification (with Clarifications)
├── plan.md                      # This file
├── research.md                  # Phase 0: decisions + VAE parity gap analysis
├── data-model.md                # Phase 1: entities, tensor contracts, config/checkpoint/report schemas
├── quickstart.md                # Phase 1: runnable validation scenarios
├── contracts/
│   ├── cli.md                   # `ddpm` command contract
│   └── component-interfaces.md  # Base classes, registries, trainer/eval signatures
├── research/
│   ├── gpu_memory_probe.py      # Empirical memory probe (evidence for the 3 GB budget)
│   └── gpu_memory_results.md
├── checklists/
│   └── requirements.md          # Spec quality checklist
└── tasks.md                     # Phase 2 output (/speckit.tasks; not created here)
```

### Source Code Layout

Mirrors `Hands-on VAE`. Entries marked **(new)** have no VAE counterpart; **(port)** files are adapted
from the VAE with minimal changes.

```text
configs/
├── cifar10_baseline.yaml        # Local default: T=1000 linear, UNet [64,128,256], batch 32 × accum 4
└── cifar10_colab.yaml           # (new) Same model/optimization; batch 128 × accum 1, num_workers 2

notebooks/
└── ddpm_colab_training.ipynb    # (new) Thin Colab driver: Drive mount → install → verify → train --resume → benchmark

src/
├── cli/
│   └── main.py                  # Typer app `ddpm`: verify, train, evaluate, sample, denoise-strip, benchmark
├── configs/
│   └── schema.py                # Dataclass sections, load_config, validate_config, ConfigError
├── data/
│   └── cifar10.py               # (port) [-1,1] transforms, random flip, seeded 45k/5k split, loaders, unnormalize
├── models/
│   ├── base.py                  # BaseNoiseSchedule, BaseDenoiser, BaseDiffusion
│   ├── types.py                 # DiffusionOutput, LossOutput, SamplingOutput
│   ├── registry.py              # SCHEDULE/DENOISER/DIFFUSION registries, build_diffusion_from_config
│   ├── schedules.py             # (new) linear + cosine schedules
│   ├── embeddings.py            # (new) Sinusoidal timestep embedding + MLP
│   ├── attention.py             # (new) Multi-head spatial self-attention (GroupNorm, SDPA)
│   ├── resnet.py                # (new) Time-conditioned residual block, Downsample, Upsample
│   ├── unet.py                  # (new) U-Net assembly (registered "unet")
│   └── gaussian_diffusion.py    # (new) Buffers, q_sample, L_simple, p_sample, sample (Algorithm 2), trajectory
├── training/
│   ├── ema.py                   # (new) EMA shadow weights
│   ├── checkpoint.py            # (port+) Adds EMA, scaler, history, rng_state, best_val_loss
│   └── trainer.py               # DDPMTrainer: accumulation, warmup, clipping, EMA, AMP opt-in, NaN/OOM guard
├── evaluation/
│   ├── metrics.py               # (port, verbatim maths) Inception extractor, FID, IS; batched DDPM sampling
│   ├── evaluator.py             # Fixed-seed test noise-prediction loss
│   └── visualizer.py            # Sample grid, denoise strip, nearest-neighbor panel, upscale_tensor
└── utils/
    ├── seeding.py               # (port) seed_everything + make_generator(seed, device)
    ├── logging.py               # (port) get_logger (ISO-8601)
    └── memory.py                # (new) GPU memory measurement helpers used by verify and tests

tests/
├── conftest.py                  # device, tiny config (channels [32,64], T=50), synthetic batches
└── unit/
    ├── test_foundations.py      # types, base classes, config validation (incl. no 32×32 attention)
    ├── test_registry.py         # registration, KeyError messages, build_diffusion_from_config
    ├── test_data.py             # split sizes, seeded split = VAE random_split, no flip on val/test, bounds
    ├── test_schedules.py        # linear/cosine values, monotonic ᾱ, ᾱ_T≈0, cosine clipping
    ├── test_forward_process.py  # q_sample closed form, statistics at t=T−1, posterior coefficients
    ├── test_shapes.py           # embedding, resblock, attention (SDPA vs einsum), U-Net I/O shapes
    ├── test_loss.py             # L_simple value, finite, non-zero gradients to all params
    ├── test_sampling.py         # Algorithm 2 shapes, clamp, z=0 at last step, seeded determinism, trajectory
    ├── test_ema.py              # EMA update maths, copy_to, state round trip
    ├── test_checkpoint.py       # save/restore incl. EMA, history, rng; resume continues epoch/step
    ├── test_trainer.py          # 2-step CPU smoke train, accumulation equivalence, NaN guard
    ├── test_metrics.py          # (port) FID=0 identical, known shift, IS uniform≈1, diverse, class_coverage
    ├── test_evaluator.py        # fixed-seed, repeatable test loss
    ├── test_visualizer.py       # grid, denoise strip layout, nearest-neighbor panel files
    ├── test_cli.py              # Typer CliRunner: --help/--version, verify on tiny config, exit codes
    └── test_memory_budget.py    # CUDA-only: train/sample/Inception ≤ 3072 MiB

docs/
├── adr/
│   └── 0001-deviations-from-hands-on-vae-conventions.md   # Required by the constitution for every VAE deviation
└── reports/
    └── 001-baseline-ddpm-report.md   # GenCV003 Deliverable (a); written after results exist

README.md                        # Overview, structure, local install + quick start + CLI guide, Colab section
HANDOFF.md                       # Updated per phase
CONTEXT.md                       # (parity, created) Domain glossary
ARCHITECTURE_DEEP_DIVE.md        # (parity, created) Design rationale + VAE→DDPM analysis
.gitignore                       # `*.ipynb` rule replaced so notebooks/ is tracked (done)
pyproject.toml                   # + explicit numpy dependency (done)
```

**Structure Decision**: Single Python project matching `Hands-on VAE`: `src/` subpackages without a
top-level `src/__init__.py`, console script `ddpm = "src.cli.main:app"`, unit tests in `tests/unit/`.
The only structural additions relative to the VAE are `notebooks/` (Colab driver) and
`src/utils/memory.py`. Deliberately omitted relative to the VAE: MNIST config/loader, VAE reference
metrics and comparison code (the comparison is written manually in the report), CSV outputs, and
`.gemini/`/`graphify-out/` (graphify runs after implementation). All of these deviations are
justified in ADR 0001.

## Implementation Phases

These phases map to the `tasks.md` phase headings generated by `/speckit.tasks`, and follow the VAE's
Setup → Foundational → User Stories → Polish order. Each phase ends with a verifiable checkpoint.

| Phase | Scope | Spec coverage | Checkpoint |
|---|---|---|---|
| P1 Setup | Ported utils (seeding, logging, memory), `schema.py`, 2 configs, `tests/conftest.py` | FR-013, FR-018 | `pytest` collects; configs validate |
| P2 Foundational | `types.py`, `base.py`, `registry.py`, `schedules.py`, CIFAR-10 data loader, checkpoint skeleton; tests for foundations, registry, schedules | FR-002, FR-003, FR-012–FR-014 | Schedule and registry tests pass |
| P3 US1 Verify (P1) | `embeddings.py`, `attention.py`, `resnet.py`, `unet.py`, `gaussian_diffusion.py` (q_sample, L_simple); `ddpm verify` + `--version`; tests for shapes, forward process, loss, memory budget | FR-001, FR-004, FR-005, FR-008–FR-011, SC-001, SC-002 | `ddpm verify` 7/7 ✓, ≤ 3072 MiB |
| P4 US2 Train (P1) 🎯 MVP | `ema.py`, full `checkpoint.py`, `DDPMTrainer` (accumulation, warmup, clip, EMA validation, AMP opt-in, NaN/OOM guard, history restore), `ddpm train` with overrides; EMA sample grids; tests for EMA, checkpoint, trainer | FR-015–FR-018b, FR-022, SC-003 | 2-epoch smoke train + resume per quickstart §4 |
| P5 US3 Sample (P2) | Algorithm 2 `p_sample`/`sample` with generator and trajectory, batched writing; `visualizer.py` grid + strip; `ddpm sample`, `ddpm denoise-strip`; sampling tests | FR-006, FR-007, FR-019–FR-021, SC-006 | Identical images for the same seed; strip rendered |
| P6 US4 Evaluate (P2) | `metrics.py` port, `evaluator.py`, nearest-neighbor panel, `ddpm evaluate`, `ddpm benchmark` (resumable samples); metric and visualizer tests | FR-023–FR-026, SC-007, SC-010 | Benchmark JSON on the smoke checkpoint |
| P7 Colab & Official Training | `cifar10_colab.yaml`, `notebooks/ddpm_colab_training.ipynb`; local full-run instructions; official T4 run; benchmark of the final EMA checkpoint | FR-018a, FR-031, SC-004, SC-005 | Trained checkpoint on Drive; `benchmark_metrics.json` |
| P8 US5 Deliver (P3) | README (verify quick start + CLI guide match the real commands; Colab section; results table), report `001-baseline-ddpm-report.md` with the manual VAE comparison, update `ARCHITECTURE_DEEP_DIVE.md`/`CONTEXT.md` with results, HANDOFF update | FR-027, FR-029–FR-031, SC-008, SC-009 | README walkthrough on a clean clone; report complete |
| P9 Polish | `test_cli.py`, contract/docs drift check (`contracts/cli.md` and README vs. `ddpm --help`), full `pytest`, quickstart run-through; graphify afterwards (user) | All | All quickstart scenarios pass |

**Dependencies**: P1 → P2 → P3 → P4 → (P5 ∥ P6 parts that do not need samples) → P7 → P8 → P9.
P5 and P6 can be developed in parallel on a smoke checkpoint. P7 training (hours) can run while P8
documentation is drafted.

## Complexity Tracking

*No constitutional violations identified.* Two items are recorded for transparency:

| Item | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Versioned Colab notebook (`notebooks/`, tracked in git) | User-chosen faster training on a T4 (Clarification Q4) | A self-contained notebook (as in the VAE) would duplicate the model and violate the Prohibited Patterns rule; the notebook here only calls the CLI |
| `torch.nn.functional.scaled_dot_product_attention` in attention | 170 MiB less memory, ~6% faster (measured) | Explicit einsum kept only as a test reference; `nn.MultiheadAttention` hides the projection layout |

## Analysis Remediation Log (2026-10-03)

`/speckit.analyze` reported 2 CRITICAL, 3 HIGH, 6 MEDIUM and 6 LOW findings. With the user's
approval, every recommended option was applied; no change was made without being recorded here.

| ID | Finding | Change applied | Files changed |
|---|---|---|---|
| K1 | Deviations from `Hands-on VAE` had no ADR (constitution MUST) | Created ADR 0001 listing deviations D1–D10; the constitution is unchanged | `docs/adr/0001-deviations-from-hands-on-vae-conventions.md`, plan Constitution Check and structure, `tasks.md` conventions, research §15 |
| K2 | Report lacked computational complexity and distribution coverage (constitution MUST) | Both added to FR-029 and T065; benchmark JSON gains a `class_coverage` block (distinct Inception top-1 classes, marginal entropy, top-20 classes) computed from existing probabilities | `spec.md` FR-029, `tasks.md` T052/T055/T060/T065, `data-model.md` §5, `contracts/cli.md` §2.6 |
| C1 | Index-based split could differ from the VAE's `random_split` partition | T016 pins train = `perm[:45000]`, val = `perm[45000:]` from `torch.randperm(50000, generator=seed)`; verified identical to `random_split` for seed 42; T011 tests the equality | `tasks.md` T011/T016, ADR 0001 D7 |
| C2 | "Same configuration unchanged" contradicted the separate Colab config | FR-018a and the Hardware assumption now say identical model, diffusion and optimization settings and effective batch; per-step batch, accumulation and workers may differ | `spec.md` FR-018a and Assumptions, ADR 0001 D2 |
| U1 | SC-008 required a "published checkpoint" that no task published | T064 publishes `best_checkpoint.pt` as a GitHub Release asset and documents the download in the README | `spec.md` SC-008, `tasks.md` T064, ADR 0001 D10 |
| G1 | SC-003 (≥ 50% loss drop) had no task | T063 checks and records it | `tasks.md` T063 |
| G2 | Same-seed training determinism (US2 scenario 5) untested | T032 (g) compares two seeded 1-epoch runs | `tasks.md` T032 |
| G3 | SC-007 (benchmark < 3.5 h) never measured | T063 records the benchmark time against 3.5 h | `tasks.md` T063 |
| I1 | Panel default (16 rows) vs SC-010 (≥ 64 samples) | `--nearest-rows` default changed to 64 | `tasks.md` T058, `contracts/cli.md` §2.4/§2.4.1, spec Clarifications |
| A1 | Ambiguous trajectory label rule at index 999 | Single rule: label T = initial `x_T`; label k < T = state after reverse step at index k | `tasks.md` T043/T046, `ARCHITECTURE_DEEP_DIVE.md` §6.4 |
| A2 | EMA/raw state-dict key prefixes unspecified | `model_state_dict` = full diffusion state dict; EMA shadow = denoiser-only, loaded via `model.denoiser.load_state_dict` | `tasks.md` T034, `data-model.md` §4 |
| A3 | SC-001 (< 2 min verify) not checked | `verify` prints elapsed seconds; T028 records it | `tasks.md` T026/T028 |
| I2 | Plan said "3 configs" | Now "2 configs; 1 ADR" | `plan.md` Scale/Scope |
| I3 | HANDOFF stale (64 × 2, MNIST) | Planning-update note added at the top of HANDOFF now; full rewrite stays in T067 | `HANDOFF.md`, `README.md` tree (ADR folder) |
| I4 | `test_data.py`, `test_evaluator.py` missing from plan tree | Added to the source layout | `plan.md` structure |
| D1 | JSON-only vs constitution "JSON/CSV" | Kept JSON only; justified in ADR 0001 D4 (constitution unchanged) | ADR 0001 |
| T1 | `mixed_precision: false` (config) vs `none` (CLI) | Config now uses `none` everywhere: `none` / `bf16` / `fp16` | `data-model.md` §3, `tasks.md` T005/T006/T035/T039, `research.md` §7 |

## Implementation Change Log

Changes made during `/speckit.implement` that refine (not alter) the design above.

| Date | Task | Change | Where documented |
|---|---|---|---|
| 2026-10-03 | T005 | Added config key `diffusion.name: gaussian` so the registry selects the diffusion process like `model.name` | data-model.md §3 |
| 2026-10-03 | T019/T026 | Gradient-flow checks run on a copy with zero-initialized parameters perturbed (`perturb_zero_init_`), because zero-init blocks upstream gradients at step 0 | tasks.md US1 notes, `src/models/gaussian_diffusion.py` docstring |
| 2026-10-03 | T026/T045 | `p_sample` implemented during T026 so `verify` uses the real Algorithm 2 step | tasks.md US1 notes |
| 2026-10-03 | Setup | `data/cifar-10-batches-py` symlinked to the existing `Hands-on VAE` download (gitignored, local only) to avoid a second 170 MB download | this log |
| 2026-10-03 | T035/T038/T047 | Generation half of US3 (T045–T047: `p_sample`, `sample`, `generate_sample_grid`, `render_denoise_strip`) implemented together with US2, so the trainer calls `generate_sample_grid` directly instead of the temporary guard described in T038 | tasks.md US2 notes |
| 2026-10-03 | T042 | Full local training estimate corrected from 8.5 h to ≈ 10 h (validation adds 30–60 s per epoch; measured 336–360 s/epoch) | research/gpu_memory_results.md, spec Assumptions, README, quickstart, deep dive §9 |
| 2026-10-03 | T060 | `benchmark` always writes sample batches but reuses them only when `--reuse-samples` is given, so a new checkpoint is never scored with stale images | contracts/cli.md §2.6 |
| 2026-10-03 | T068 | `test_cli.py` verifies the tiny config with `beta_end: 0.3`: with T=50 the default β range leaves ᾱ_T ≈ 0.60 and `verify` correctly fails | tests/unit/test_cli.py comment |

