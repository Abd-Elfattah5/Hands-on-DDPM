# CLI Interface Contract: `ddpm`

**Feature**: `001-create-ddpm`
**Status**: Completed (implemented in `src/cli/main.py`; option parity checked by `tests/unit/test_cli.py`)

The command-line interface is the primary entry point for every operational workflow
(Constitution Principle V). It is built with `typer`, mirrors the option vocabulary of the `vae` CLI,
and is the only training path used by the Colab notebook.

---

## 1. Global Invariant & Options

```bash
ddpm [OPTIONS] COMMAND [ARGS]...
```

- `--help`: Shows the help message and exits.
- `--version`: Prints `Hands-on DDPM version: 0.1.0`.

**Exit codes (all commands)**:
- `0`: Success.
- `1`: Configuration or input error (invalid YAML, missing checkpoint, config mismatch on resume,
  failed verification check).
- `2`: Runtime error (CUDA out-of-memory, non-finite loss, interrupted download).

**Device**: The config value `experiment.device` (`auto` = CUDA if available) is used; CPU fallback
prints a warning.

**Console**: Human-readable progress (tqdm plus `├──`/`└──` summary trees, as in the VAE).
Machine-readable results are written as JSON only.

**Command set**: Exactly six commands: `verify`, `train`, `evaluate`, `sample`, `denoise-strip`,
`benchmark` (FR-028). Secondary outputs are options of the command that produces their input, not
extra commands. For example, the nearest-neighbor panel is `sample --nearest` (§2.4.1).

---

## 2. Commands

### 2.1 `verify`

Pre-training gate (Constitution: Pre-Training Gate; spec User Story 1).

```bash
ddpm verify --config <yaml> [--skip-memory]
```

- **Options**:
  - `-c, --config PATH`: Experiment YAML. **[Required]**
  - `--skip-memory`: Skip the GPU memory check (for CPU-only machines). *[Optional]*
- **Checks Performed**:
  1. The configuration validates against the schema.
  2. Schedule: β in (0, 1), ᾱ_t strictly decreasing, ᾱ_T < 1e-3 (linear and cosine).
  3. Forward process: x_t at t = T−1 has mean ≈ 0 and std ≈ 1 (tolerance 0.05 over a batch).
  4. U-Net output shape equals the input shape `[B, 3, 32, 32]`.
  5. L_simple is finite and every trainable parameter receives a finite, non-zero gradient
     (checked on a copy with zero-initialized parameters perturbed, since zero-init blocks
     upstream gradients at step 0).
  6. A short reverse pass (10 steps) returns finite values in [-1, 1] after clamping.
  7. Memory: one full training step at the configured batch reports total GPU memory ≤
     `verify.memory_budget_mib` (default 3072).
- **Output**: A ✓/✗ line per check, plus parameter count, measured MiB and elapsed seconds (SC-001
  target < 120 s, informational). Exit code `1` if any check fails. With `--skip-memory` or on CPU,
  6 checks run (`Checks passed: 6/6`).

### 2.2 `train`

```bash
ddpm train --config <yaml> [--resume <ckpt>] [--seed N] [--epochs N] [--batch-size N]
           [--grad-accum N] [--output-dir PATH] [--mixed-precision {none,bf16,fp16}]
```

- **Options**:
  - `-c, --config PATH`: Experiment YAML. **[Required]**
  - `-r, --resume PATH`: Checkpoint (`latest.pt` or `best_checkpoint.pt`) to resume from. Restores
    model, EMA, optimizer, scheduler, scaler, history and RNG state, then continues at `epoch + 1`. *[Optional]*
  - `-s, --seed INT`: Overrides `experiment.seed`. *[Optional]*
  - `--epochs INT`: Overrides `training.epochs` (may extend a resumed run). *[Optional]*
  - `--batch-size INT`: Overrides `data.batch_size` (per-step micro-batch). *[Optional]*
  - `--grad-accum INT`: Overrides `training.grad_accum_steps`. *[Optional]*
  - `--output-dir PATH`: Overrides `experiment.output_dir` (e.g., a Google Drive folder on Colab). *[Optional]*
  - `--mixed-precision TEXT`: Overrides `training.mixed_precision`. *[Default: from config, `none`]*
- **Artifacts Emitted** (`<output_dir>/`):
  - `latest.pt` (every epoch), `best_checkpoint.pt` and `best.pt` (EMA validation loss improved),
    `epoch_NNN.pt` (every `save_every`), `final_checkpoint.pt`.
  - `metrics.json` (list of epoch records), `loss_curve.png`, `train.log`.
  - `samples/epoch_NNN.png`: 8×8 EMA sample grid every `sample_every` epochs.
  - `resolved_config.yaml`: the effective configuration after overrides.
- **Failure behavior**: A non-finite loss or OOM stops training with exit code `2` and a message
  (suggesting a smaller `--batch-size` with a larger `--grad-accum`, or fp32); `best_checkpoint.pt`
  is never overwritten by a failed epoch.

### 2.3 `evaluate`

```bash
ddpm evaluate --checkpoint <ckpt> [--weights ema|raw] [--batch-size N] [--data-dir PATH] [--out PATH]
```

- **Options**:
  - `-k, --checkpoint PATH`: Checkpoint file. **[Required]**
  - `--weights TEXT`: `ema` or `raw`. *[Default: ema]*
  - `-b, --batch-size INT`: Evaluation batch size. *[Default: 128]*
  - `-d, --data-dir PATH`: Overrides the dataset directory. *[Optional]*
  - `-o, --out PATH`: *[Default: `<checkpoint_dir>/eval_metrics.json`]*
- **Output**: JSON containing `test_loss`, `total_samples`, `weights`, `checkpoint`, `seed`,
  `timestamp`, `device_name`. The test loss uses fixed-seed timesteps and noise so it is repeatable.

### 2.4 `sample`

```bash
ddpm sample --checkpoint <ckpt> [--num-samples 64] [--seed N] [--batch-size 256]
            [--weights ema|raw] [--out PATH] [--upscale 4] [--save-individual] [--nearest]
```

- **Options**:
  - `-k, --checkpoint PATH` **[Required]**
  - `-n, --num-samples INT`: *[Default: 64]* (square number recommended for the grid).
  - `-s, --seed INT`: *[Default: config seed]*; the same seed gives identical images on the same hardware.
  - `-b, --batch-size INT`: Images per reverse chain batch. *[Default: `sampling.batch_size`, 256]*
  - `--weights TEXT`: *[Default: ema]*
  - `-o, --out PATH`: Grid image path. *[Default: `artifacts/samples/sample_grid.png`]*
  - `--upscale INT`: Bicubic tile upscaling (4 → 1024×1024 canvas for 64 images). *[Default: 4]*
  - `--save-individual`: Also save each image as `<out_stem>/NNNNN.png`. *[Optional]*
  - `--nearest`: Also build the nearest-neighbor memorization panel (§2.4.1). *[Optional]*
  - `--nearest-k INT`: Neighbors shown per generated image. *[Default: 3]*
  - `--nearest-rows INT`: Generated images included in the panel. *[Default: 64]* (SC-010 inspects at least 64)
- **Output**: Grid PNG plus `<out_stem>_timing.json` (`num_samples`, `seed`, `sampling_seconds_total`,
  `sampling_seconds_per_image`, `device_name`). Finished batches are written as they complete.

#### 2.4.1 Nearest-neighbor panel (`--nearest`, FR-023, SC-010)

**Purpose**: Check that the model generates *new* images instead of memorizing (copying) CIFAR-10
training images. A low FID alone can't tell these apart, because a model that copied training
images would also score well.

**Method**:
1. Take the first `--nearest-rows` generated images (in [0, 1], 3×32×32).
2. For each one, compute the pixel-space Euclidean (L2) distance to all 50,000 CIFAR-10 training
   images, in GPU chunks (≈ 600 MB in float32, within the 3 GB budget).
3. Keep the `--nearest-k` training images with the smallest distance.

**Layout** (`<out_stem>_nearest.png`, plus `<out_stem>_nearest.json` with indices and distances):

```text
            generated   | 1st nearest   2nd nearest   3rd nearest   (training images)
row 1:     [ sample 1 ] | [ train #a ]  [ train #b ]  [ train #c ]
row 2:     [ sample 2 ] | [ train #d ]  [ train #e ]  [ train #f ]
...
row 64:    [ sample 64] | ...
```

**How to read it**: If a generated image is almost pixel-identical to its first neighbor, the model
memorized that image. If the neighbors share only color, layout or class (for example a similar
green background) but are clearly different pictures, the model is generalizing. SC-010 requires no
near-duplicates among the inspected samples.

**Why an option, not a seventh command**: The panel post-processes the images that `sample` has
just generated (same checkpoint, seed and weights). A separate command would have to regenerate
the samples or pass them through a file, and the spec fixes the command set at six.

### 2.5 `denoise-strip`

```bash
ddpm denoise-strip --checkpoint <ckpt> [--num-images 8] [--steps 1000,800,600,400,200,100,50,0]
                   [--seed N] [--weights ema|raw] [--out PATH] [--upscale 4]
```

- **Options**:
  - `-k, --checkpoint PATH` **[Required]**
  - `--num-images INT`: Number of independent chains (one row each). *[Default: 8]*
  - `--steps TEXT`: Comma-separated paper timesteps to capture. *[Default: `sampling.strip_steps`]*
  - `-s, --seed INT`, `--weights TEXT`, `--upscale INT`: as in `sample`.
  - `-o, --out PATH`: *[Default: `artifacts/strips/denoise_strip.png`]*
- **Output**: A PNG with rows = images and columns = timesteps, labeled `t=1000 … t=0`.

### 2.6 `benchmark`

```bash
ddpm benchmark --checkpoint <ckpt> [--num-samples 5000] [--batch-size 64] [--sample-batch-size 256]
               [--seed N] [--weights ema|raw] [--out PATH] [--reuse-samples DIR]
```

- **Options**:
  - `-k, --checkpoint PATH` **[Required]**
  - `-n, --num-samples INT`: Generated and real image count for FID/IS. *[Default: 5000]*
  - `-b, --batch-size INT`: Inception extraction batch. *[Default: 64]*
  - `--sample-batch-size INT`: Reverse chain batch. *[Default: 256]*
  - `-s, --seed INT`, `--weights TEXT`: as above.
  - `--reuse-samples DIR`: Reuse previously generated sample batches (resumes an interrupted
    benchmark). Without it, batches are still written to `artifacts/eval/samples_<seed>/` but
    always regenerated, so a new checkpoint is never scored with stale images. *[Optional]*
  - `-o, --out PATH`: *[Default: `artifacts/eval/benchmark_metrics.json`]*
- **Steps**: (1) test loss; (2) generate N EMA samples in batches, saving them to
  `artifacts/eval/samples_<seed>/`; (3) Inception features of the first N test images and the N
  samples; (4) FID, plus IS over 10 splits; (5) write JSON.
- **Output**: `benchmark_metrics.json` with every key in [data-model.md §5](../data-model.md),
  including the values needed for the manual DDPM vs. VAE table in the final report
  (`fid`, `inception_score_mean/std`, `sampling_seconds_per_image`, `num_parameters`,
  `training_seconds`) and the `class_coverage` block used for the report's distribution-coverage
  section. No VAE files are read. Example console (values illustrative):

```text
Benchmark (EMA, 5000 samples, seed 42)
├── Test noise-prediction loss: 0.0312
├── Fréchet Inception Distance: 23.41
├── Inception Score: 7.85 ± 0.09
├── Sampling time: 2.79 h (2.01 s/image)
└── Metrics written to artifacts/eval/benchmark_metrics.json
```
