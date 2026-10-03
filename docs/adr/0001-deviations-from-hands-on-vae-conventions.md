# 0001. Deviations from Hands-on VAE Conventions

**Status**: Accepted (2026-10-03) · **Feature**: `001-create-ddpm` · **Required by**: Constitution, *Technical Stack & Architectural Constraints → Ecosystem Consistency* ("MUST mirror `Hands-on VAE` unless an ADR in `docs/adr/` justifies a deviation").

The DDPM repository mirrors the `Hands-on VAE` module layout, registry pattern, CLI vocabulary, checkpoint convention and FID/IS protocol. This ADR is the single justification record for every place where it intentionally does not. Any future deviation MUST be added here (new row) or in a new ADR.

| # | VAE convention | DDPM decision | Justification |
|---|---|---|---|
| D1 | `configs/mnist_baseline.yaml` and `src/data/mnist.py` | Not included | User decision (plan review, 2026-10-02). MNIST was a VAE diagnostic, not a GenCV003 deliverable; CIFAR-10 is the only benchmarked dataset. |
| D2 | One config per experiment, run anywhere | Two configs: `cifar10_baseline.yaml` (32 × 4 accumulation) and `cifar10_colab.yaml` (128 × 1) | The local GPU has a measured 3 GB budget; Colab has 15 GB. Model, diffusion and optimization settings, including the effective batch of 128, are identical; only per-step batch, accumulation steps, data workers and the default `output_dir` (`artifacts/runs/cifar10_colab`, so the two runs never overwrite each other; code review R4) differ (spec FR-018a, research §13). |
| D3 | No notebooks tracked (`*.ipynb` ignored); the VAE diagnostic notebook is self-contained | `notebooks/ddpm_colab_training.ipynb` is tracked and contains no model code, only CLI calls | User decision (Clarification Q4, plan review). Colab gives faster training; restricting the notebook to CLI calls keeps the constitution's *Prohibited Patterns* rule (no training loops in notebooks). |
| D4 | Metrics as JSON | JSON only (no CSV) | User decision. The constitution's "JSON/CSV format" is satisfied by JSON; one format avoids duplicated outputs. |
| D5 | Benchmark results compared inside the VAE repo | No VAE metric files or comparison code; the DDPM benchmark JSON stores every DDPM value and the author writes the DDPM vs. VAE table manually in the report | User decision. Keeps the repositories independent; the table is produced once. |
| D6 | Training validation loss on raw weights; cosine LR decay; Adam | Best checkpoint selected by **EMA** validation loss; linear warmup (5,000 steps) then constant LR; AdamW | Ho et al. (2020) settings; EMA weights are the ones used for generation (research §6). |
| D7 | CIFAR-10 split via `random_split` on one dataset object | Two dataset objects (flip vs. no flip) split by indices from `torch.randperm(50000, generator=torch.Generator().manual_seed(seed))`: train = first 45,000, val = last 5,000 | Prevents the training flip augmentation from leaking into validation. The permutation is the one `random_split` draws for the same seed and lengths, so the partition is **identical** to the VAE's (verified by a unit test, task T011) and Constitution III is preserved. |
| D8 | Resume does not restore metric history; documented exit code 2 not implemented | Resume restores the full history and RNG state; exit codes 0/1/2 implemented | Fixes two VAE defects found during the parity review (research §11–12). |
| D9 | `.gemini/`, `GEMINI.md`, `graphify-out/` tracked | Not included now | This repository uses the Kilo Code integration; the user runs graphify after implementation. |
| D10 | VAE checkpoints and artifacts not published | `best_checkpoint.pt` of the reported run is published as a GitHub Release asset and linked from the README | Needed so a reviewer can reproduce the reported FID/IS from the published checkpoint (spec SC-008). |

**Consequences**: The repositories remain directly comparable on dataset, splits, preprocessing and FID/IS protocol (Constitution III), while the DDPM repository is simpler (no MNIST, CSV or comparison code) and supports Colab training without moving training logic into a notebook.
