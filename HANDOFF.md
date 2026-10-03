# HANDOFF: Hands-on DDPM (Denoising Diffusion Probabilistic Models)

**Repository**: `Hands-on DDPM` (`/home/amousa1/Projects/Hands-on DDPM`) · GitHub `Abd-Elfattah5/Hands-on-DDPM` (private)
**Assignment**: `GenCV003` Part 2 (DDPM), sibling of `Hands-on VAE`
**Feature branch**: `001-create-ddpm` · **Date**: 2026-10-03
**Status**: Implementation complete through User Story 4 (code, tests, CLI, Colab notebook). **Pending**: the official 100-epoch training run, final benchmark, report and README results (tasks T062–T066).

---

## 1. Where Everything Is

| What | Where |
|---|---|
| Specification, plan, research, data model, CLI contract, quickstart, tasks | `specs/001-create-ddpm/` |
| Measured GPU numbers (probe, verify, smoke training, sampling) | `specs/001-create-ddpm/research/gpu_memory_results.md` |
| Deviations from Hands-on VAE (required by the constitution) | `docs/adr/0001-deviations-from-hands-on-vae-conventions.md` |
| Design rationale / glossary | `ARCHITECTURE_DEEP_DIVE.md`, `CONTEXT.md` |
| Plan changes during analysis and implementation | `specs/001-create-ddpm/plan.md` → "Analysis Remediation Log", "Implementation Change Log" |
| Code | `src/` (cli, configs, data, models, training, evaluation, utils) |
| Tests | `tests/unit/` (90 tests; `test_memory_budget.py` needs CUDA) |
| Colab driver | `notebooks/ddpm_colab_training.ipynb` |
| Configs | `configs/cifar10_baseline.yaml` (local, 32 × 4), `configs/cifar10_colab.yaml` (Colab, 128 × 1) |

## 2. Implemented System (summary)

- **Model**: time-conditioned U-Net, channels [64, 128, 256], 2 residual blocks per level (3 on the up path), 4-head attention at 16×16, 8×8 and the bottleneck, GroupNorm(32), dropout 0.1, zero-initialized output layers; **16,056,451 parameters**.
- **Diffusion**: linear (default) or cosine schedule, T = 1000, buffers computed in float64; `q_sample`, L_simple, Algorithm 2 (`fixed_large` default, `fixed_small` option).
- **Training**: AdamW 2e-4, 5,000-step warmup, clip 1.0, EMA 0.9999, effective batch 128, fp32 default (`--mixed-precision bf16|fp16` opt-in), NaN/OOM guards, `latest.pt` / `best_checkpoint.pt` (EMA validation loss) / `epoch_NNN.pt` / `final_checkpoint.pt`, resume restores history and RNG.
- **Evaluation**: VAE-identical FID/IS protocol (5,000 vs 5,000, IS over 10 splits), `class_coverage`, test loss, nearest-neighbor panel.
- **CLI**: `ddpm verify | train | evaluate | sample | denoise-strip | benchmark` (exit codes 0/1/2).

## 3. Measured So Far (T2000, 3 GB budget)

| Item | Result |
|---|---|
| `ddpm verify` (linear and cosine) | 7/7 checks, 1,918 MiB, 5 s |
| Memory tests | train 32×4 1,772 MiB · sample 256 2,290 MiB · Inception 64 2,498 MiB |
| Training speed | ~336–360 s per epoch incl. validation → ≈ 10 h for 100 epochs locally |
| Smoke run (3 epochs + resume) | history intact (3 records), global step 1,053, peak 1,876 MiB |
| Sampling | ~2.2 s/image at batch 16–100; ≈ 2.8 h for 5,000 at batch 256 (probe) |
| Determinism | same seed → byte-identical PNGs |

## 4. Remaining Work (tasks.md Phase 7–8)

1. **T062** Run `notebooks/ddpm_colab_training.ipynb` on a Colab T4 (or locally: `ddpm train --config configs/cifar10_baseline.yaml`, ≈ 10 h, resumable). Copy `best_checkpoint.pt`, `metrics.json`, `loss_curve.png`, `samples/` from Drive `runs/cifar10_colab/` into `artifacts/runs/cifar10_colab/` (or use `artifacts/runs/cifar10_baseline/` for a local run).
2. **T063** `ddpm evaluate`, `ddpm benchmark --num-samples 5000`, `ddpm sample -n 64 --seed 42 --upscale 4 --out artifacts/samples/sample_grid_1024.png --nearest`, `ddpm denoise-strip --num-images 8 --seed 42`; check SC-003/004/005/007/010 and record them in `research/gpu_memory_results.md` ("Final results").
3. **T064** Publish `best_checkpoint.pt` as a GitHub Release, add results and the download command to `README.md`.
4. **T065** Write `docs/reports/001-baseline-ddpm-report.md`, including computational complexity, distribution coverage and the manual DDPM vs. VAE table (VAE: FID 169.02 / 181.00, IS 2.11 / 1.68).
5. **T066** Add a Results section to `ARCHITECTURE_DEEP_DIVE.md`; **T072** open the PR, then run graphify.

## 5. Mathematical Reference

The full derivations (forward process, closed form, posterior, ε-parametrization, L_simple, Algorithm 2) and the design rationale are in `ARCHITECTURE_DEEP_DIVE.md` §3–§6. The VAE vs. DDPM design comparison is in §10 there.

## 6. Resume Prompt for the Next Session

> *"We are continuing `Hands-on DDPM` (`/home/amousa1/Projects/Hands-on DDPM`, branch `001-create-ddpm`). Read `HANDOFF.md`, then `specs/001-create-ddpm/tasks.md`. Implementation through US4 is done and tested. Continue with T062 (official training run) onward."*
