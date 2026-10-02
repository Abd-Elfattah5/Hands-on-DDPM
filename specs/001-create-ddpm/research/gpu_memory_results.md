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
