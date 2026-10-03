# Quickstart & Validation Guide: From-Scratch DDPM

**Feature**: `001-create-ddpm`
**Date**: 2026-10-02

Runnable scenarios that prove the feature works end to end. Command details are in
[contracts/cli.md](./contracts/cli.md); data shapes and file schemas are in
[data-model.md](./data-model.md).

---

## 1. Prerequisites

- Python 3.12, NVIDIA GPU with CUDA 12.4 drivers (local: Quadro T2000, 4 GB), ~1 GB of disk for
  CIFAR-10, ~2 GB for checkpoints.
- Local setup (already done on the development machine):

```bash
python3 -m venv DDPM && source DDPM/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

## 2. Validation Scenario 1: Unit Test Suite (Pre-Training Gate)

```bash
pytest tests/ -v
```

**Expected**: All tests pass; `test_memory_budget.py` runs on CUDA (skipped on CPU) and asserts that
train (32 × 4), sample (batch 256) and Inception (batch 64) each stay ≤ 3072 MiB.

## 3. Validation Scenario 2: Architecture Verification (User Story 1)

```bash
ddpm verify --config configs/cifar10_baseline.yaml
```

**Expected**: 7/7 checks ✓, 16,056,451 parameters (16.06 M), measured GPU memory ≈ 1.9 GB (≤ 3072 MiB),
completes in < 2 minutes (SC-001), exit code 0. Repeat with a cosine-schedule override config and
expect the same result.

## 4. Validation Scenario 3: Smoke Training + Resume (User Story 2)

```bash
ddpm train --config configs/cifar10_baseline.yaml --epochs 2 --output-dir artifacts/runs/smoke
ddpm train --config configs/cifar10_baseline.yaml --epochs 3 --output-dir artifacts/runs/smoke \
           --resume artifacts/runs/smoke/latest.pt
```

**Expected**: `latest.pt`, `best_checkpoint.pt`, `final_checkpoint.pt`, `metrics.json` (3 records
after resume, not 1), `loss_curve.png` and `train.log`; the loss decreases from epoch 1
to epoch 3; the resumed run starts at epoch 3 with `global_step` continuing.

## 5. Validation Scenario 4: Sampling Determinism (User Story 3)

```bash
ddpm sample --checkpoint artifacts/runs/smoke/best_checkpoint.pt -n 16 --seed 7 --out /tmp/a.png
ddpm sample --checkpoint artifacts/runs/smoke/best_checkpoint.pt -n 16 --seed 7 --out /tmp/b.png
cmp /tmp/a.png /tmp/b.png && echo IDENTICAL
ddpm denoise-strip --checkpoint artifacts/runs/smoke/best_checkpoint.pt --num-images 4
```

**Expected**: `IDENTICAL` (SC-006); a strip with 8 labeled columns `t=1000 … t=0`; timing JSON
written. (Smoke-trained images look noisy; only integrity is checked here.)

## 6. Validation Scenario 5: Full Local Training (reproducibility path)

```bash
ddpm train --config configs/cifar10_baseline.yaml          # ~10 h on T2000 (~6 min/epoch measured), resumable
```

**Expected**: 100 epochs, `peak_memory_mib` in every `metrics.json` record ≤ 3072 MiB (SC-002); training loss
drops ≥ 50% between epoch 1 and epoch 100 (SC-003).

## 7. Validation Scenario 6: Colab Training (official run)

1. Open `notebooks/ddpm_colab_training.ipynb` in Colab and select Runtime → T4 GPU.
2. Add a `GH_TOKEN` Colab Secret (read access to the private repository), or upload a repository zip
   to Drive.
3. Run all cells. The notebook mounts Drive, installs the package, runs `ddpm verify`, then
   `ddpm train --config configs/cifar10_colab.yaml --output-dir /content/drive/MyDrive/hands-on-ddpm/runs/cifar10_baseline`.
4. Simulate a disconnect (Runtime → Disconnect), reconnect and run all cells again.

**Expected**: Training resumes from `latest.pt` on Drive and loses ≤ 1 epoch; `metrics.json` keeps
the full history.

## 8. Validation Scenario 7: Benchmark (User Story 4)

```bash
ddpm evaluate  --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt
ddpm benchmark --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt --num-samples 5000
ddpm sample    --checkpoint artifacts/runs/cifar10_baseline/best_checkpoint.pt -n 64 --seed 42 \
               --out artifacts/samples/sample_grid_1024.png --nearest
```

**Expected**: `benchmark_metrics.json` with `fid`, `inception_score_mean/std`, `benchmark_samples:
5000`, `sampling_seconds_per_image`, `num_parameters`, `training_seconds`; target FID < 50 (SC-004)
and IS above both VAEs (SC-005; VAE values 169.02 / 181.00 are compared manually in the report); benchmark time < 3.5 h (SC-007); nearest-neighbor
panel shows no near-duplicates (SC-010).
