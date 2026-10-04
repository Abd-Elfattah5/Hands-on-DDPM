# Graph Report - Hands-on DDPM  (2026-10-05)

## Corpus Check
- Large corpus: 140 files · ~569,763 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 864 nodes · 1676 edges · 58 communities (36 shown, 22 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 122 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Checkpoint Serialization
- FID IS Metrics
- Attention And Building Blocks
- Specs Domain Contracts
- CIFAR-10 Data Pipeline
- DDPM Trainer Loop
- CLI Contract Spec
- Config Schema Validation
- Spec Kit Workflow
- Noise Schedules
- Visualizer And Nearest Neighbors
- Component Registry
- Spec Kit Scripts
- Model Module Docs
- Architecture Diagram
- Gaussian Diffusion Core
- Evaluate Benchmark Commands
- Final Sample Evaluation
- Test Fixtures
- CLI Entry And Verify
- Loss And Gradient Checks
- Diffusion Base Interfaces
- Enhancement Taxonomy
- Reverse Process Concepts
- CLI Contract Tests
- GPU Memory Helpers
- EMA Epoch Progression
- Project Documents Hub
- U-Net Design Concepts
- Local And Colab Configs
- Benchmark Protocol Results
- CLI Error Handling
- Forward Process Tests
- VAE Comparison Context
- Sampling Command
- Forward Process Concepts
- Memorization Check
- VAE DDPM Difference
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- EMA Epoch Grid (isolated)
- Package Metadata
- Evaluate Command Node

## God Nodes (most connected - your core abstractions)
1. `build_diffusion_from_config()` - 36 edges
2. `BaseDiffusion` - 29 edges
3. `make_generator()` - 21 edges
4. `device()` - 21 edges
5. `EMA` - 20 edges
6. `DDPMTrainer` - 20 edges
7. `ConfigError` - 19 edges
8. `compute_fid_and_is()` - 19 edges
9. `validate_config()` - 18 edges
10. `Hands-on DDPM Constitution v1.0.0` - 18 edges

## Surprising Connections (you probably didn't know these)
- `DDIM Sampler (Song et al. 2021)` --semantically_similar_to--> `Ancestral Sampling (Algorithm 2)`  [INFERRED] [semantically similar]
  docs/reports/001-baseline-ddpm-report.md → CONTEXT.md
- `extract()` --indirect_call--> `device()`  [INFERRED]
  src/evaluation/metrics.py → tests/conftest.py
- `Resolved Config (official Colab run)` --shares_data_with--> `cifar10_colab.yaml (Colab 128x1)`  [INFERRED]
  artifacts/runs/cifar10_baseline/resolved_config.yaml → configs/cifar10_colab.yaml
- `Velocity Prediction (v)` --semantically_similar_to--> `Noise-Prediction Parametrization (epsilon-prediction)`  [INFERRED] [semantically similar]
  architectural_enhancements_taxonomy.html → CONTEXT.md
- `Report Figure: DDPM Training Loss (Colab T4, 80 epochs)` --semantically_similar_to--> `CIFAR-10 Baseline Loss Curve (Colab T4, 80 epochs)`  [INFERRED] [semantically similar]
  docs/reports/figures/loss_curve.png → artifacts/runs/cifar10_baseline/loss_curve.png

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Spec Kit SDD command pipeline (constitution → specify → clarify → plan → tasks → analyze → implement → converge)** — _kilocode_workflows_speckit_constitution_speckit_constitution, _kilocode_workflows_speckit_specify_speckit_specify, _kilocode_workflows_speckit_clarify_speckit_clarify, _kilocode_workflows_speckit_plan_speckit_plan, _kilocode_workflows_speckit_tasks_speckit_tasks, _kilocode_workflows_speckit_analyze_speckit_analyze, _kilocode_workflows_speckit_implement_speckit_implement, _kilocode_workflows_speckit_converge_speckit_converge [EXTRACTED 1.00]
- **Hands-on DDPM constitution core principles I-V** — _specify_memory_constitution_from_scratch_diffusion_modeling, _specify_memory_constitution_deterministic_reproducibility, _specify_memory_constitution_standardized_benchmarking, _specify_memory_constitution_test_driven_mathematical_integrity, _specify_memory_constitution_cli_driven_pipeline [EXTRACTED 1.00]
- **Templates kept in sync with constitution** — _specify_templates_plan_template_plan_template, _specify_templates_spec_template_spec_template, _specify_templates_tasks_template_tasks_template, _specify_templates_checklist_template_checklist_template, _specify_templates_constitution_template_constitution_template [INFERRED 0.85]
- **DDPM core: forward marginal, eps-prediction L_simple, Algorithm 2 sampling** — context_closed_form_forward_marginal, context_noise_prediction_parametrization, context_simplified_objective, context_denoising_network, context_ancestral_sampling, context_reverse_variance [EXTRACTED 1.00]
- **Evaluation suite: FID, IS, class coverage, nearest-neighbor memorization check** — context_frechet_inception_distance, context_inception_score, context_class_coverage, context_nearest_neighbor_panel, context_shared_benchmark_protocol [EXTRACTED 1.00]
- **Future speed-ups for DDPM sampling and training** — docs_reports_001_baseline_ddpm_report_ddim_sampler, docs_reports_001_baseline_ddpm_report_progressive_distillation, architectural_enhancements_taxonomy_learned_interpolated_variance, docs_reports_001_baseline_ddpm_report_ema_warmup, docs_reports_001_baseline_ddpm_report_latent_diffusion [EXTRACTED 1.00]
- **Six-command ddpm CLI (FR-028)** — specs_001_create_ddpm_contracts_cli_verify_command, specs_001_create_ddpm_contracts_cli_train_command, specs_001_create_ddpm_contracts_cli_evaluate_command, specs_001_create_ddpm_contracts_cli_sample_command, specs_001_create_ddpm_contracts_cli_denoise_strip_command, specs_001_create_ddpm_contracts_cli_benchmark_command [EXTRACTED 1.00]
- **Registry-built diffusion components** — specs_001_create_ddpm_contracts_component_interfaces_component_registry, specs_001_create_ddpm_research_noise_schedules, specs_001_create_ddpm_research_unet, specs_001_create_ddpm_tasks_gaussiandiffusion, specs_001_create_ddpm_contracts_component_interfaces_build_diffusion_from_config [EXTRACTED 1.00]
- **EMA non-convergence leads to raw-weight reporting** — specs_001_create_ddpm_contracts_component_interfaces_ema, specs_001_create_ddpm_research_gpu_memory_results_ema_convergence_diagnosis, specs_001_create_ddpm_plan_raw_weights_reporting, specs_001_create_ddpm_research_gpu_memory_results_final_results [INFERRED 0.85]
- **Every-5-epoch EMA sample grids (epochs 5-80) of cifar10_baseline** — artifacts_runs_cifar10_baseline_samples_epoch_005, artifacts_runs_cifar10_baseline_samples_epoch_010, artifacts_runs_cifar10_baseline_samples_epoch_015, artifacts_runs_cifar10_baseline_samples_epoch_020, artifacts_runs_cifar10_baseline_samples_epoch_025, artifacts_runs_cifar10_baseline_samples_epoch_030, artifacts_runs_cifar10_baseline_samples_epoch_035, artifacts_runs_cifar10_baseline_samples_epoch_040, artifacts_runs_cifar10_baseline_samples_epoch_045, artifacts_runs_cifar10_baseline_samples_epoch_050, artifacts_runs_cifar10_baseline_samples_epoch_055, artifacts_runs_cifar10_baseline_samples_epoch_060, artifacts_runs_cifar10_baseline_samples_epoch_065, artifacts_runs_cifar10_baseline_samples_epoch_070, artifacts_runs_cifar10_baseline_samples_epoch_075, artifacts_runs_cifar10_baseline_samples_epoch_080 [INFERRED 0.95]
- **Report sample-quality and memorization figures** — docs_reports_figures_sample_grid, docs_reports_figures_sample_grid_ema, docs_reports_figures_nearest_neighbors_top12, docs_reports_figures_denoise_strip [INFERRED 0.85]
- **DDPM pipeline: schedule -> backbone -> prediction -> sampling -> loss** — architecture_visualization_forward_process, architecture_visualization_ho_unet, architecture_visualization_noise_prediction, architecture_visualization_reverse_process, architecture_visualization_l_simple [EXTRACTED 1.00]
- **3-Epoch Smoke-Test Artifacts** — artifacts_samples_smoke_grid_smoke_sample_grid, artifacts_samples_smoke_grid_nearest_smoke_nearest_panel, artifacts_strips_smoke_strip_smoke_denoising_strip, artifacts_samples_smoke_grid_smoke_test [EXTRACTED 1.00]
- **80-Epoch Sample Quality & Memorization Evaluation** — artifacts_samples_sample_grid_1024_raw_sample_grid, artifacts_samples_sample_grid_1024_ema_ema_sample_grid, artifacts_samples_sample_grid_1024_nearest_nearest_neighbor_panel, artifacts_strips_denoise_strip_denoising_strip [INFERRED 0.85]

## Communities (58 total, 22 thin omitted)

### Community 0 - "Checkpoint Serialization"
Cohesion: 0.05
Nodes (48): datetime, Logger, numpy, Optimizer, os, pathlib, random, capture_rng_state() (+40 more)

### Community 1 - "FID IS Metrics"
Cohesion: 0.05
Nodes (52): ndarray, scipy, src_evaluation, batch_generator(), calculate_class_coverage(), calculate_frechet_distance(), calculate_inception_score(), _check_or_write_manifest() (+44 more)

### Community 2 - "Attention And Building Blocks"
Cohesion: 0.06
Nodes (32): copy, register_denoiser, Tensor, Explicit attention: softmax(Q Kᵀ / √d) V for ``[..., N, d]`` tensors., GroupNorm → 1×1 conv to Q/K/V → multi-head attention → 1×1 output conv →…, reference_attention(), SpatialSelfAttention, Tensor (+24 more)

### Community 3 - "Specs Domain Contracts"
Cohesion: 0.07
Nodes (36): ddpm verify (pre-training gate, 7 checks), BaseDenoiser, BaseDiffusion, BaseNoiseSchedule, Component Registry (SCHEDULE/DENOISER/DIFFUSION), DiffusionOutput, Experiment Configuration Schema, LossOutput (+28 more)

### Community 4 - "CIFAR-10 Data Pipeline"
Cohesion: 0.07
Nodes (38): Compose, src_data, _cifar(), ddpm_collate_fn(), get_cifar10_dataloaders(), get_cifar10_datasets(), get_cifar10_train_transforms(), get_cifar10_transforms() (+30 more)

### Community 5 - "DDPM Trainer Loop"
Cohesion: 0.07
Nodes (30): RuntimeError, DDPMTrainer, NonFiniteLossError, Any, DataLoader, Module, no_grad, Path (+22 more)

### Community 6 - "CLI Contract Spec"
Cohesion: 0.06
Nodes (44): Specification Quality Checklist (requirements.md), SC-004 Risk (FID < 50 literature target), CLI Interface Contract: ddpm, ddpm benchmark (resumable FID/IS), ddpm denoise-strip, Exit Code Policy (0 success / 1 config-input / 2 runtime), Nearest-Neighbor Panel (sample --nearest), ddpm sample (+36 more)

### Community 7 - "Config Schema Validation"
Cohesion: 0.08
Nodes (39): dataclasses, apply_overrides(), _build_section(), ConfigError, DataSectionConfig, DiffusionSectionConfig, EvaluationSectionConfig, ExperimentConfig (+31 more)

### Community 8 - "Spec Kit Workflow"
Cohesion: 0.11
Nodes (33): Constitution Authority (conflicts are CRITICAL), Spec Kit Extension Hooks (.specify/extensions.yml), /speckit.analyze (cross-artifact consistency analysis), /speckit.checklist (requirements quality checklist), /speckit.clarify (spec ambiguity clarification), /speckit.constitution (constitution authoring/sync), /speckit.converge (append unbuilt work as tasks), /speckit.implement (execute tasks.md) (+25 more)

### Community 9 - "Noise Schedules"
Cohesion: 0.09
Nodes (24): pytest, register_schedule, CosineSchedule, LinearSchedule, Tensor, Ho et al. (2020) linear schedule: β_t evenly spaced from ``beta_start`` to…, Nichol & Dhariwal (2021) cosine schedule with offset ``s`` and β_t clipped at…, make_generator() (+16 more)

### Community 10 - "Visualizer And Nearest Neighbors"
Cohesion: 0.12
Nodes (27): json, pil, find_nearest_neighbors(), generate_sample_grid(), _grid_nrow(), load_train_images(), no_grad, Path (+19 more)

### Community 11 - "Component Registry"
Cohesion: 0.12
Nodes (22): importlib, src_models, build_diffusion_from_config(), _ensure_default_components(), get_denoiser(), get_diffusion(), get_schedule(), _lookup() (+14 more)

### Community 12 - "Spec Kit Scripts"
Cohesion: 0.16
Nodes (22): check-prerequisites.sh script, check_dir(), check_file(), find_specify_root(), format_speckit_command(), get_current_branch(), get_feature_paths(), get_invoke_separator() (+14 more)

### Community 13 - "Model Module Docs"
Cohesion: 0.13
Nodes (17): collections_abc, math, Test-split noise-prediction loss with fixed-seed timesteps and noise…, Multi-head spatial self-attention built from primitives (research §5). The…, Abstract base classes for registered diffusion components (contracts/component-…, Sinusoidal timestep embedding and its projection MLP (research §4, deep dive…, Gaussian diffusion (Ho et al. 2020): schedule buffers, q(x_t|x_0), L_simple and…, Time-conditioned residual block and resolution changes (research §4, deep dive… (+9 more)

### Community 14 - "Architecture Diagram"
Cohesion: 0.09
Nodes (24): Full DDPM End-to-End Architecture Diagram, ADM U-Net (High Fidelity Backbone), Class/Text Embedding via Cross-Attention, Cosine Noise Schedule (Nichol & Dhariwal), Optional Non-Markovian DDIM Fast Sampler, Fixed Reverse Variance sigma_t^2 I (Ho et al.), Forward Diffusion Process q(x_t|x_{t-1}), q(x_t|x_0), Standard Ho et al. U-Net (Default Backbone) (+16 more)

### Community 15 - "Gaussian Diffusion Core"
Cohesion: 0.14
Nodes (16): ABC, register_diffusion, Size, BaseDenoiser, BaseNoiseSchedule, A variance schedule β_1..β_T., A time-conditioned network predicting the noise added to ``x_t``., GaussianDiffusion (+8 more)

### Community 16 - "Evaluate Benchmark Commands"
Cohesion: 0.17
Nodes (21): benchmark(), _device_name(), evaluate(), _load_for_inference(), Any, Path, `auto` → CUDA if available else CPU (with a warning); explicit `cuda`/`cpu`…, Average noise-prediction loss on the CIFAR-10 test split (fixed-seed,… (+13 more)

### Community 17 - "Final Sample Evaluation"
Cohesion: 0.13
Nodes (19): EMA-Weight Sample Grid (same seed, unconverged EMA), EMA Weights (decay 0.9999), Unconverged EMA Artifacts (oversaturated, blotchy high-contrast samples), Memorization, Nearest-Neighbor Panel (80-epoch samples vs 3 nearest CIFAR-10 train images), No Memorization Evidence (neighbors share color/layout, not exact copies), Pixel L2 Distance Retrieval, Raw-Weight Sample Grid (64 samples, 80 epochs) (+11 more)

### Community 18 - "Test Fixtures"
Cohesion: 0.20
Nodes (16): baseline_config(), fake_cifar_loaders(), fake_loader_factory(), _loader(), make_fake_loaders(), make_tiny_config(), DataLoader, fixture (+8 more)

### Community 19 - "CLI Entry And Verify"
Cohesion: 0.15
Nodes (14): callback, functools, _check(), main(), Typer CLI `ddpm`: verify, train, evaluate, sample, denoise-strip, benchmark…, Hands-on DDPM command-line interface., Pre-training gate: schedule, forward process, shapes, gradients, reverse pass…, verify() (+6 more)

### Community 20 - "Loss And Gradient Checks"
Cohesion: 0.15
Nodes (13): Training objective for a forward-pass output., perturb_zero_init_(), Module, Fill every all-zero parameter with small Gaussian noise (in place). Zero-…, L_simple = ‖ε − ε_θ(x_t, t)‖² averaged over batch and pixels (FR-005)., DiffusionOutput, LossOutput, Output of a training forward pass. Attributes: x_t: Noised input ``[B, C, H,… (+5 more)

### Community 21 - "Diffusion Base Interfaces"
Cohesion: 0.16
Nodes (9): BaseDiffusion, Tensor, Return ``[T]`` float64 per-step variances, each in (0, 1)., Map ``[B, C, H, W]`` noisy images and ``[B]`` int64 timesteps to ``[B, C, H,…, A diffusion process wrapping a denoiser: forward noising, loss and reverse…, Closed-form forward marginal q(x_t | x_0)., Training forward pass on clean images., One reverse step from zero-based index ``t`` to ``t-1``. (+1 more)

### Community 22 - "Enhancement Taxonomy"
Cohesion: 0.19
Nodes (14): Architectural Enhancements Taxonomy, Classifier-Free Guidance (CFG), Continuous / SDE Schedule, Diffusion Transformer (DiT) Backbone, Hybrid Loss (L_simple + lambda L_vlb), Learned Interpolated Variance, Velocity Prediction (v), Cosine Noise Schedule (Nichol & Dhariwal) (+6 more)

### Community 23 - "Reverse Process Concepts"
Cohesion: 0.18
Nodes (12): Coarse-to-Fine Generation, Ancestral Sampling (Algorithm 2), Checkpoint Convention (latest/best/final), Denoising Trajectory / Strip, EMA Weights (decay 0.9999), Reverse Denoising Process p_theta(x_{t-1}|x_t), Reverse Variance sigma_t^2 (fixed_large / fixed_small), True Reverse Posterior q(x_{t-1}|x_t,x_0) (+4 more)

### Community 24 - "CLI Contract Tests"
Cohesion: 0.20
Nodes (7): re, _plain(), CLI contract: command set, version, verify on a tiny config, exit codes,…, test_help_lists_exactly_six_commands(), test_help_options_match_contract(), typer_testing, yaml

### Community 25 - "GPU Memory Helpers"
Cohesion: 0.27
Nodes (9): shutil, measured_total_mib(), peak_reserved_mib(), process_total_mib(), _query(), GPU memory measurement helpers used by `ddpm verify` and the memory-budget…, Peak memory reserved by the PyTorch caching allocator, in MiB (0 without CUDA)., Total GPU memory used by this process as reported by `nvidia-smi`. Uses the… (+1 more)

### Community 26 - "EMA Epoch Progression"
Cohesion: 0.33
Nodes (9): Ancestral Sampling (Algorithm 2), Unconverged EMA Sample Progression (noise to saturated blobs over 80 epochs), EMA Sample Grid Epoch 5 (pure Gaussian-like color noise), EMA Sample Grid Epoch 20, EMA Sample Grid Epoch 40 (saturated speckle noise, emerging blotches), EMA Sample Grid Epoch 60, EMA Sample Grid Epoch 80 (oversaturated blob-like shapes, no recognizable objects), cifar10_baseline Training Run (+1 more)

### Community 27 - "Project Documents Hub"
Cohesion: 0.44
Nodes (6): Ho et al. (2020) Denoising Diffusion Probabilistic Models, ADR 0001: Deviations from Hands-on VAE Conventions, Project Constitution (Ecosystem Consistency), Baseline DDPM Technical Report, GenCV003 Assignment, Release v0.1.0-ddpm-baseline (published checkpoint)

### Community 28 - "U-Net Design Concepts"
Cohesion: 0.28
Nodes (9): Measured 3 GB GPU Budget (Quadro T2000), Nearest-Neighbor + Conv Upsampling, Spatial Self-Attention at 16x16 and 8x8, Time-Conditioned Residual Block, Denoising Network epsilon_theta (time-conditioned U-Net), Gradient Accumulation, Group Normalization (32 groups), Sinusoidal Timestep Embedding (+1 more)

### Community 29 - "Local And Colab Configs"
Cohesion: 0.28
Nodes (9): One Code Path, Two Machines (CLI + Colab notebook), Resolved Config (smoke run), cifar10_baseline.yaml (local 32x4), cifar10_colab.yaml (Colab 128x1), Effective Batch (128), D2: Two Configs (local vs Colab), Colab Training Notebook (ddpm_colab_training.ipynb), ddpm CLI (verify/train/evaluate/sample/denoise-strip/benchmark) (+1 more)

### Community 30 - "Benchmark Protocol Results"
Cohesion: 0.25
Nodes (8): CIFAR-10 Dataset (45k/5k/10k split), Resolved Config (official Colab run), Class Coverage, Frechet Inception Distance (FID), Inception Score (IS), Shared Benchmark Protocol (5k vs 5k), D7: Index-Based Split Identical to VAE, Benchmark Results (FID 39.69, IS 5.18)

### Community 31 - "CLI Error Handling"
Cohesion: 0.29
Nodes (7): command, denoise_strip(), _handle_errors(), Train the DDPM; writes checkpoints, metrics.json, loss_curve.png and sample…, Render the reverse process: rows = images, columns = timesteps t=1000 … t=0., Exit-code policy (contracts/cli.md §1): 0 success, 1 config/input error, 2…, train()

### Community 32 - "Forward Process Tests"
Cohesion: 0.25
Nodes (4): model(), fixture, Forward diffusion: buffer identities, closed-form q(x_t|x_0), statistics at…, test_final_step_is_standard_normal()

### Community 33 - "VAE Comparison Context"
Cohesion: 0.33
Nodes (6): Hands-on VAE (sibling repository), L2 Conditional-Mean Trap (VAE blur), VAE vs DDPM Paradigm Comparison, CONTEXT.md (Domain Glossary), Component Registry (decorator factory), Latent Diffusion

### Community 34 - "Sampling Command"
Cohesion: 0.40
Nodes (6): Generate images with Algorithm 2 and save a grid (+ timing JSON, optional…, sample(), on_batch(), Inverse transform converting [-1, 1] normalized tensors back to [0, 1] image…, unnormalize(), test_unnormalize()

### Community 35 - "Forward Process Concepts"
Cohesion: 0.40
Nodes (5): Linear Noise Schedule (1e-4 to 0.02), Closed-Form Forward Marginal q(x_t|x_0), Cumulative Signal Level (alpha_bar_t), Forward Diffusion Process q(x_t|x_{t-1}), Noise Schedule (beta_t)

## Knowledge Gaps
- **72 isolated node(s):** `common.sh script`, `hands-on-ddpm`, `ExperimentSectionConfig`, `DataSectionConfig`, `ModelSectionConfig` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 313 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GPU Memory Probe Results` connect `CLI Contract Spec` to `Specs Domain Contracts`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `BaseDiffusion` connect `Diffusion Base Interfaces` to `Checkpoint Serialization`, `FID IS Metrics`, `DDPM Trainer Loop`, `Config Schema Validation`, `Visualizer And Nearest Neighbors`, `Component Registry`, `Model Module Docs`, `Gaussian Diffusion Core`, `Evaluate Benchmark Commands`, `Loss And Gradient Checks`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `BaseDiffusion` (e.g. with `evaluate_model()` and `compute_fid_and_is()`) actually correct?**
  _`BaseDiffusion` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `common.sh script`, `hands-on-ddpm`, `ExperimentSectionConfig` to the rest of the system?**
  _72 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Checkpoint Serialization` be split into smaller, more focused modules?**
  _Cohesion score 0.05478750640040963 - nodes in this community are weakly interconnected._
- **Should `FID IS Metrics` be split into smaller, more focused modules?**
  _Cohesion score 0.05026300409117475 - nodes in this community are weakly interconnected._
- **Should `Attention And Building Blocks` be split into smaller, more focused modules?**
  _Cohesion score 0.061224489795918366 - nodes in this community are weakly interconnected._