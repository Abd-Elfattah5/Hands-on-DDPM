# HANDOFF: Hands-on DDPM (Denoising Diffusion Probabilistic Models)

**Repository**: `Hands-on DDPM` (`/home/amousa1/Projects/Hands-on DDPM`) · GitHub `Abd-Elfattah5/Hands-on-DDPM` (private)
**Assignment**: `GenCV003` Part 2 (DDPM), sibling of `Hands-on VAE`
**Feature branch**: `001-create-ddpm` · **Date**: 2026-10-03
**Status**: **Complete.** Trained (80 epochs, Colab T4), benchmarked (FID 39.69, IS 5.18 ± 0.14, raw weights), report, README and release published; PR to `main` opened (T072). Remaining for the user: confirm the VAE rows in the report, merge the PR, run graphify.

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

## 4. Final Results (T062–T066)

| Item | Result |
|---|---|
| Training | 80 of 100 epochs (user decision), 28,080 steps, Tesla T4, 236 s/epoch, 5.25 h |
| Weights reported | **raw** (EMA retains ~6% random init at 28k steps; its samples saturate). ADR 0001 D11 |
| FID / IS | **39.69 / 5.18 ± 0.14** (VAE: 169.02 / 2.11 baseline, 181.00 / 1.68 enhanced) |
| Test loss | 0.0309 (raw), 0.0390 (EMA) |
| Benchmark time | 2.99 h sampling (2.16 s/image), ≈ 3.0–3.1 h total on the T2000 (< 3.5 h) |
| Coverage / memorization | 322 Inception classes, max share 5.2% / nearest training L2 ≥ 4.75 |
| Checkpoint | release `v0.1.0-ddpm-baseline`, asset `ddpm_cifar10_baseline_ep080.pt` |
| Report | `docs/reports/001-baseline-ddpm-report.md` (figures in `docs/reports/figures/`) |
| Artifacts (local, gitignored) | `artifacts/runs/cifar10_baseline/`, `artifacts/eval/`, `artifacts/samples/`, `artifacts/strips/`, Drive zips in `artifacts/downloads/` |

Open for the user: verify the VAE numbers in the report's §7 table, merge the PR, then run graphify.

## 5. Mathematical Reference

The full derivations (forward process, closed form, posterior, ε-parametrization, L_simple, Algorithm 2) and the design rationale are in `ARCHITECTURE_DEEP_DIVE.md` §3–§6. The VAE vs. DDPM design comparison is in §10 there.

## 6. Resume Prompt for the Next Session

> *"`Hands-on DDPM` feature 001 is complete (see `HANDOFF.md` §4 and `docs/reports/001-baseline-ddpm-report.md`). Possible next feature `002-enhanced-ddpm`: longer training / EMA decay warm-up, larger U-Net, DDIM sampling."*
