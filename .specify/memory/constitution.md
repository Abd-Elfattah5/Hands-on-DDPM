<!--
Sync Impact Report:
- Version change: (none) → 1.0.0 (initial ratification)
- Source: modeled on Hands-on VAE constitution v1.1.0; requirements from assignment GenCV003.
- Principles defined:
  - I. From-Scratch Diffusion Modeling
  - II. Deterministic Reproducibility
  - III. Standardized Benchmarking & Cross-Model Comparison
  - IV. Test-Driven Tensor & Diffusion Mathematical Integrity
  - V. CLI-Driven Pipeline & Artifact Observability
- Added sections: Technical Stack & Architectural Constraints; Experimental Workflow & Quality Gates;
  Governance.
- Removed sections: none.
- Deferred TODOs: none.
-->

# Hands-on DDPM Constitution

## Core Principles

### I. From-Scratch Diffusion Modeling
The Denoising Diffusion Probabilistic Model MUST be implemented from foundational tensor
operations and neural network primitives, following Ho et al. (2020). Pre-existing full
implementations, high-level generative wrappers (e.g., HuggingFace `diffusers`), and pre-packaged
or pretrained diffusion pipelines MUST NOT be used for the generative backbone. Every mathematical
mechanism—variance schedules ($\beta_t$, $\alpha_t$, $\bar{\alpha}_t$), the closed-form forward
marginal $q(x_t \mid x_0)$, the true reverse posterior $q(x_{t-1} \mid x_t, x_0)$, the
$\epsilon$-prediction parametrization, the simplified objective $L_{\text{simple}}$, the
sinusoidal timestep embedding, the time-conditioned U-Net, and the reverse sampling procedure
(Algorithm 2)—MUST be explicitly authored, mathematically documented, and self-contained within
the project codebase. Pretrained networks are permitted only inside evaluation metrics (e.g., the
Inception network used for FID/IS).

### II. Deterministic Reproducibility
All experimental pipelines, data splits, training routines, and sampling runs MUST be deterministic
and reproducible. Global random seeds across all active runtimes (Python `random`, `numpy`, PyTorch
CPU/CUDA) MUST be strictly controlled and exposed via configuration. Environment specifications,
data preprocessing pipelines, diffusion hyperparameters ($T$, schedule type, $\beta$ range),
model hyperparameters, EMA decay, and model checkpoints (including EMA weights) MUST be versioned
and saved with unambiguous provenance so that any run, quantitative metric, or qualitative visual
can be recreated on demand. The README MUST document the exact commands required to reproduce every
reported result.

### III. Standardized Benchmarking & Cross-Model Comparison
Model evaluation MUST be rigorous and conducted under standardized conditions on the CIFAR-10
public benchmark dataset using the same training/validation/testing partitions, preprocessing,
and metric implementations as the sibling `Hands-on VAE` project, so results are directly
comparable. Evaluation MUST incorporate both quantitative rigor and qualitative inspection:
- Quantitative metrics MUST include Fréchet Inception Distance (FID) and Inception Score (IS),
  computed with identical sample counts and protocol as the VAE benchmarks, alongside training/test
  noise-prediction loss, sampling wall-clock time, and parameter count.
- Qualitative assessments MUST systematically capture realism, diversity, thematic (class)
  consistency, and mode-coverage diagnostics, including sample grids and step-by-step reverse
  denoising strips ($x_T \rightarrow x_0$).
- Results MUST be reported side by side with the VAE baseline and enhanced VAE, explicitly
  discussing the architectural differences and performance trade-offs between the two approaches.

### IV. Test-Driven Tensor & Diffusion Mathematical Integrity
All neural network modules, diffusion utilities, and loss functions MUST be verified through
automated unit tests before model training commences. Verification MUST explicitly check:
- Tensor dimension preservation and spatial transformation contracts through all U-Net layers,
  attention blocks, ResNet blocks, and timestep embeddings.
- Correctness of schedule-derived quantities and the closed-form forward process (e.g.,
  monotonic $\bar{\alpha}_t$, $\bar{\alpha}_T \approx 0$, $q(x_t \mid x_0)$ statistics).
- Non-zero gradient flow from $L_{\text{simple}}$ to all trainable parameters.
- Numerical stability of training and sampling (no NaN/Inf values, bounded posterior variances,
  outputs clipped to $[-1, 1]$).
- Shape and determinism of the reverse sampling loop under a fixed seed.

### V. CLI-Driven Pipeline & Artifact Observability
All operational workflows—dataset downloading/preprocessing, model training, checkpoint
evaluation, image sampling, denoising-strip visualization, and metric aggregation—MUST be exposed
through modular Command Line Interfaces. Pipelines MUST adhere to structured I/O principles: runtime
configurations supplied via CLI arguments or structured config files, machine-readable logs and
metrics emitted in JSON/CSV format alongside standard console output, and visual artifacts (loss
plots, sample grids, denoising strips) automatically organized in designated artifact directories.

## Technical Stack & Architectural Constraints

- **Language & Core Framework**: Python 3.12+ using PyTorch for tensor computation and automatic
  differentiation.
- **Dependency Isolation**: All third-party dependencies MUST be managed within an isolated virtual
  environment (`DDPM/`) and recorded in reproducible dependency manifests (`pyproject.toml`).
- **Hardware Budget**: Default configurations MUST fit the local NVIDIA Quadro T2000 (4 GB VRAM,
  ~3.5 GB safe working limit). Self-attention MUST NOT be placed at the 32×32 resolution;
  larger effective batch sizes MUST use gradient accumulation.
- **Architectural Separation**: Code MUST be modularly separated into distinct packages:
  - `models/`: Timestep embeddings, attention, ResNet blocks, U-Net backbone, Gaussian diffusion
    process, and component registries.
  - `data/`: Dataset loading, augmentation, and normalization to $[-1, 1]$.
  - `training/`: Training loops, optimizers, EMA tracking, and checkpoint handlers.
  - `evaluation/`: Metric computation (FID, Inception Score), sampling, and qualitative
    visualization.
  - `utils/`: Deterministic seeding, configuration loading, and logging utilities.
  - `cli/`: Command Line Interface entry points.
- **Ecosystem Consistency**: Module layout, CLI conventions, config schema, and reporting formats
  MUST mirror `Hands-on VAE` unless an ADR in `docs/adr/` justifies a deviation.
- **Prohibited Patterns**: No monolithic unmaintainable scripts; training loops and model
  definitions MUST NOT be entangled in unversioned interactive notebooks without corresponding
  tested library modules.

## Experimental Workflow & Quality Gates

1. **Pre-Training Gate**: Unit tests for U-Net forward passes, diffusion formulas, loss
   computation, backward gradient passes, and the sampling loop MUST pass prior to launching
   training experiments.
2. **Experiment Tracking**: Training loss, validation loss, learning rate, and periodic EMA sample
   grids MUST be tracked and saved per epoch.
3. **Evaluation Gate**: Final EMA checkpoints MUST undergo automated quantitative evaluation (FID,
   IS) on the test split protocol, producing timestamped metric summaries.
4. **Deliverable Reporting**: Analyses MUST culminate in a comprehensive technical report
   documenting implementation steps, theoretical formulation, experimental results, sample
   generation speed, computational complexity, sample quality, distribution coverage, and
   conclusions that clarify the main differences between VAE and DDPM approaches.
5. **Repository Deliverable**: The GitHub repository MUST include a README describing how to set
   up the environment and reproduce all reported results.

## Governance

This Constitution serves as the authoritative standard for architectural, empirical, and coding
decisions in the project. Any deviation from these principles requires an explicit amendment.

- **Amendment Process**: Proposed amendments MUST be documented with clear technical rationale and
  recorded with an updated Sync Impact Report.
- **Versioning Policy**: Semantic Versioning (`MAJOR.MINOR.PATCH`): MAJOR for principle removals
  or redefinitions, MINOR for new principles/sections or materially expanded guidance, PATCH for
  clarifications and wording fixes.
- **Compliance**: All pull requests, code reviews, and project milestones MUST verify compliance
  with this constitution before merging or release.

**Version**: 1.0.0 | **Ratified**: 2026-10-02 | **Last Amended**: 2026-10-02
