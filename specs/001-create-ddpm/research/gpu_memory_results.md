# GPU Memory Probe Results (2026-10-02)

Hardware: NVIDIA Quadro T2000, 4096 MiB total, ~3935 MiB free at idle. PyTorch 2.6.0+cu124.
Probe: `gpu_memory_probe.py`. It builds the spec'd U-Net (channels [64,128,256], 2 res blocks/stage,
4-head attention at 16x16, 8x8 and bottleneck, none at 32x32, GroupNorm, dropout 0.1),
16.06 M parameters. Each scenario ran in a fresh process with real CUDA allocations.
Training scenarios include AdamW optimizer state and an EMA copy on the GPU (worst case).

"Process total" is the nvidia-smi figure, which counts the CUDA context and the allocator cache,
so it is the number to compare with the 3 GB (3072 MiB) limit.

| Scenario | Peak tensors (MiB) | Process total (MiB) | Under 3072? | Speed |
|---|---|---|---|---|
| Train batch 32 x accum 4 (fp32) | 1415 | 1625 | yes | 0.84 s/optimizer step |
| Train batch 48 x accum 3 (fp32) | 1959 | 2219 | yes | 0.91 s/optimizer step |
| Train batch 64 x accum 2 (fp32) | 2486 | 2925 | yes, 147 MiB margin | 0.87 s/optimizer step |
| Train batch 64 x accum 2 (fp32, fused attention) | 2488 | 2755 | yes, 317 MiB margin | 0.82 s/optimizer step |
| Train batch 64 x accum 2 (fp16 mixed precision) | 2213 | 2505 | yes | 3.43 s/step, loss became NaN/Inf |
| Train batch 96 x accum 1 (fp32, fused attention) | 3502 | 3915 | NO | 0.72 s/step |
| Train batch 128 x accum 1 (fp32) | 4580 | 3925 (spilled past VRAM) | NO | 5.13 s/step |
| Sample batch 64 (no grad) | 394 | 647 | yes | ~140 s per 1000-step batch, ~3.06 h for 5,000 images |
| Sample batch 128 (no grad) | 716 | 1115 | yes | ~257 s per batch, ~2.85 h for 5,000 images |
| Sample batch 256 (no grad) | 1359 | 2061 | yes | ~500 s per batch, ~2.78 h for 5,000 images |
| Inception-v3 feature extraction, batch 64 | 1833 | 2269 | yes | n/a |

Findings:
- The full architecture trains under 3 GB at batch 64, but with only a 147-317 MiB margin.
- Batch 48 x 3 (effective 144) or 32 x 4 (effective 128) leaves 850-1450 MiB of headroom.
- fp16 mixed precision on this GPU was slower and produced non-finite losses, so it is not usable as-is.
- Effective batch 128 without gradient accumulation does not fit.
- Estimated training time at batch 64 x 2: 352 steps/epoch x 0.87 s = ~5.1 min/epoch, so ~8.5 h for 100 epochs.

## Verify gate (implementation, 2026-10-03, T028)

`ddpm verify` on the implemented model (16,056,451 parameters, identical to the probe):

| Config | Checks | Measured total GPU memory (32 × 4 training step) | Elapsed |
|---|---|---|---|
| `configs/cifar10_baseline.yaml` (linear, ᾱ_T = 4.04e-05) | 7/7 | 1,918 MiB ≤ 3,072 | 5.0 s (SC-001 < 120 s) |
| baseline with `diffusion.schedule: cosine` (ᾱ_T = 2.43e-09) | 7/7 | 1,918 MiB ≤ 3,072 | 5.0 s |

`tests/unit/test_memory_budget.py` (fresh subprocess per scenario): train 32 × 4 = 1,772 MiB,
sample batch 256 = 2,290 MiB, Inception batch 64 = 2,498 MiB; all ≤ 3,072 MiB.
`measured_total_mib` reports max(driver-reported usage, peak reserved + 300 MiB context estimate);
on WSL2 the driver reports whole-GPU usage, so these figures are conservative.

## Smoke training (implementation, 2026-10-03, T042)

`ddpm train --epochs 2 --output-dir artifacts/runs/smoke`, then `--epochs 3 --resume latest.pt`
(real CIFAR-10, baseline config, T2000):

| Epoch | global_step | epoch_seconds (train + raw/EMA validation) | peak_memory_mib | train / val / val-EMA loss |
|---|---|---|---|---|
| 1 | 351 | 335.7 | 1,774 | 0.906 / 0.704 / 0.998 |
| 2 | 702 | 433.0 (GPU shared with the unit-test run) | 1,876 | 0.425 / 0.156 / 0.985 |
| 3 (resumed) | 1,053 | 360.5 | 1,860 | 0.077 / 0.043 / 0.957 |

- Resume continued at epoch 3 with `global_step` 702 → 1,053 and `metrics.json` holding all 3 records.
- The EMA validation loss stays near 1.0 this early because EMA 0.9999 averages over ~10,000 steps;
  it overtakes the raw loss after roughly 30 epochs (~10k optimizer steps). This is expected.
- **Corrected full-run estimate**: ~336–360 s/epoch (training ≈ 0.84 s/step × 351 steps ≈ 295 s, plus
  ≈ 30–60 s for raw + EMA validation and data loading) → **≈ 10 h for 100 epochs** on the T2000
  (the planning estimate of 8.5 h excluded validation).

## Sampling (implementation, 2026-10-03, T051)

- `ddpm sample -n 16 --seed 7` twice → byte-identical PNGs (`cmp` → IDENTICAL; SC-006).
- 16 images in one batch: 35.8 s (2.24 s/image); 100 images in one batch: 2.18 s/image.
- Batch-256 throughput was measured by the probe (~500 s per 256 images ≈ 1.95 s/image, ≈ 2.8 h for
  5,000); the per-image cost falls with larger batches.
- `ddpm denoise-strip --num-images 4` renders 8 labeled columns `t=1000 … t=0`.

## Evaluation pipeline (implementation, 2026-10-03, T061)

Smoke checkpoint (3 epochs, untrained-quality samples; integrity check only):
`ddpm evaluate` → test loss 0.957 on 10,000 images; `ddpm benchmark -n 100` → all data-model §5 keys
written, FID 527.6 / IS 1.23 (meaningless at 3 epochs and 100 samples, the pipeline is what was
validated); `ddpm sample --nearest` → panel and JSON written (min L2 27.7, no copies). Inception
extraction ran at batch 64 within budget (test_memory_budget: 2,498 MiB).

## Colab T4 (measured, 2026-10-03/04, T062)

- `notebooks/ddpm_colab_training.ipynb` on a Tesla T4 (14.6 GB, torch 2.11.0+cu130), `configs/cifar10_colab.yaml`
  (128 × 1), single uninterrupted session, **80 of 100 epochs** (stopped by the user; see "Weights used for reporting").
- Mean **236 s/epoch** (≈ 208 s epoch 1, ≈ 340 s on sample-grid epochs), **5.25 h** total, peak 5,794 MiB on the T4
  (the 3 GB budget applies to the local GPU only).
- Final record (epoch 80, global step 28,080): train 0.0311, val 0.0293, val-EMA 0.0372.

## Weights used for reporting (2026-10-04, T063)

The EMA decay of 0.9999 keeps 0.9999^28,080 ≈ **6%** of the untrained initialization in the EMA copy at epoch 80.
Measured on the final checkpoint:

| Weights | Test L_simple (10,000 images) | Final-sample saturation (|x| > 0.98) | Visual |
|---|---|---|---|
| EMA | 0.0390 | **56.9%** (washed-out, high-contrast blobs) | not recognisable |
| EMA + x₀ clipping (diagnostic only) | — | 1.7% | soft, low-contrast textures |
| **Raw** | **0.0309** | **1.7%** | recognisable CIFAR-10 objects (horses, birds, cars, ships, deer) |

The residual 6% of random initial weights still corrupts the 1,000-step reverse chain (errors compound), even though
the EMA single-step loss is close to the raw one. **All reported results therefore use `--weights raw`**, which the CLI
supports by design (FR-015: "allow raw weights to be selected"). With 100 epochs the residual would be 3%, and Ho et
al. train ~800k steps where it is negligible. The diagnosis compared EMA and raw weights with plain Algorithm 2 and an x₀-clipped variant on 16 seeded
samples (throwaway script, not kept in the repo; x₀ clipping is not used for any reported number, to stay
faithful to Algorithm 2).

## Final results (2026-10-04, T063)

| Check | Result |
|---|---|
| SC-003 training loss drop | 0.906 → 0.031 (−96.6%), no NaN/Inf ✓ |
| SC-010 memorization | 64 raw samples vs 50,000 training images: nearest L2 min 4.75, median 9.02 (pixels in [0,1], 3,072 dims); visual check of the closest pairs (e.g. sample 17 vs train #42014) shows same class/pose but different images: no copies ✓ |
| Sampling speed (T2000, batch 256, raw) | ≈ 521 s per 256 images (2.03 s/image) |
| SC-004 FID < 50 | **39.69** (5,000 raw-weight samples vs first 5,000 test images) ✓ |
| SC-005 IS above both VAEs (2.11, 1.68) | **5.18 ± 0.14** ✓ |
| SC-007 benchmark < 3.5 h on the T2000 | test loss + 5,000 samples + Inception scoring ≈ **3.0–3.1 h** wall clock (sampling 10,777 s = 2.99 h, 2.16 s/image; batches 2–3 were slowed by concurrent grid/strip generation) ✓ |
| Test L_simple | 0.0309 (raw), 0.0390 (EMA) |
| Class coverage | 322 distinct Inception top-1 classes, marginal entropy 5.85 nats |
