"""Typer CLI `ddpm`: verify, train, evaluate, sample, denoise-strip, benchmark (contracts/cli.md)."""

import copy
import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
import typer

# Import every component module so the registry decorators run.
import src.models.gaussian_diffusion  # noqa: F401
import src.models.schedules  # noqa: F401
import src.models.unet  # noqa: F401
from src.configs.schema import ConfigError, load_config, validate_config
from src.models.gaussian_diffusion import perturb_zero_init_
from src.models.registry import build_diffusion_from_config
from src.utils.logging import get_logger
from src.utils.memory import measured_total_mib, reset_peak

__version__ = "0.1.0"

app = typer.Typer(
    name="ddpm",
    help="Modular From-Scratch Denoising Diffusion Probabilistic Model CLI for CIFAR-10.",
    add_completion=False,
)
logger = get_logger("ddpm.cli")


# --------------------------------------------------------------------------------------------- helpers

def _resolve_device(config: dict[str, Any]) -> torch.device:
    """`auto` → CUDA if available else CPU (with a warning); explicit `cuda`/`cpu` honoured."""
    requested = config.get("experiment", {}).get("device", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("experiment.device is 'cuda' but no CUDA device is available")
    typer.secho("Warning: CUDA not available, falling back to CPU (much slower).", fg=typer.colors.YELLOW)
    return torch.device("cpu")


def _handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Exit-code policy (contracts/cli.md §1): 0 success, 1 config/input error, 2 runtime error."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except (ConfigError, FileNotFoundError, KeyError, ValueError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except (torch.cuda.OutOfMemoryError, RuntimeError, ArithmeticError) as exc:
            typer.secho(f"Runtime error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    return wrapper


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Hands-on DDPM version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Hands-on DDPM command-line interface."""


def _check(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark, color = ("✓", typer.colors.GREEN) if ok else ("✗", typer.colors.RED)
    typer.secho(f"  {mark} {name}" + (f" ({detail})" if detail else ""), fg=color)


# ---------------------------------------------------------------------------------------------- verify

@app.command()
@_handle_errors
def verify(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True, help="Path to experiment YAML configuration."),
    skip_memory: bool = typer.Option(False, "--skip-memory", help="Skip the GPU memory check (CPU-only machines)."),
) -> None:
    """Pre-training gate: schedule, forward process, shapes, gradients, reverse pass and memory."""
    start = time.time()
    raw = load_config(config)
    results: list[tuple[str, bool, str]] = []
    typer.secho(f"Verifying {config}", bold=True)

    # 1. schema
    cfg = validate_config(raw)
    _check(results, "Configuration validates against the schema", True)
    device = _resolve_device(raw)
    torch.manual_seed(cfg.experiment.seed)

    model = build_diffusion_from_config(raw).to(device)
    params = sum(p.numel() for p in model.parameters())
    T = model.timesteps

    # 2. schedule
    abar = model.alphas_cumprod
    betas_ok = bool(((model.betas > 0) & (model.betas < 1)).all())
    mono = bool((abar[1:] < abar[:-1]).all())
    _check(results, "Schedule: β ∈ (0,1), ᾱ_t strictly decreasing, ᾱ_T < 1e-3",
           betas_ok and mono and abar[-1].item() < 1e-3, f"ᾱ_T = {abar[-1].item():.2e}")

    # 3. forward statistics at t = T-1
    g = torch.Generator(device=device).manual_seed(0)
    x0 = torch.full((256, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size), 0.5, device=device)
    xt = model.q_sample(x0, torch.full((256,), T - 1, device=device),
                        torch.randn(x0.shape, device=device, generator=g))
    mean, std = xt.mean().item(), xt.std().item()
    _check(results, "Forward process: x_{T} ≈ N(0, I)", abs(mean) < 0.05 and abs(std - 1) < 0.05,
           f"mean {mean:+.3f}, std {std:.3f}")

    # 4. output shape
    b = 2
    x = torch.rand(b, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size, device=device) * 2 - 1
    with torch.no_grad():
        eps = model.denoiser(x, torch.tensor([0, T - 1], device=device))
    _check(results, "U-Net output shape equals input shape", eps.shape == x.shape, str(tuple(eps.shape)))

    # 5. loss and gradient flow (on a perturbed copy: zero-initialized layers block gradients at init)
    probe = copy.deepcopy(model)
    perturb_zero_init_(probe)
    probe.train()
    loss = probe.compute_loss(probe(x, generator=torch.Generator(device=device).manual_seed(1))).loss
    loss.backward()
    dead = [n for n, p in probe.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.norm() == 0)]
    _check(results, "L_simple finite; every parameter gets a finite non-zero gradient",
           bool(torch.isfinite(loss)) and not dead, f"loss {loss.item():.4f}" + (f", dead: {dead[:3]}" if dead else ""))
    del probe

    # 6. short reverse pass
    model.eval()
    with torch.no_grad():
        xr = torch.randn(b, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size,
                         device=device, generator=torch.Generator(device=device).manual_seed(2))
        for t in range(T - 1, max(T - 11, -1), -1):
            xr = model.p_sample(xr, t, generator=None)
        xr = xr.clamp(-1, 1)
    _check(results, "10-step reverse pass finite and within [-1, 1]",
           bool(torch.isfinite(xr).all()) and xr.abs().max().item() <= 1.0)

    # 7. memory budget
    budget = cfg.verify.memory_budget_mib
    if skip_memory or device.type != "cuda":
        typer.secho(f"  - Memory check skipped ({'--skip-memory' if skip_memory else 'no CUDA'})", fg=typer.colors.YELLOW)
        measured = None
    else:
        del model
        model = build_diffusion_from_config(raw).to(device).train()
        ema = copy.deepcopy(model.denoiser).requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
        reset_peak(device)
        accum, bs = cfg.training.grad_accum_steps, cfg.data.batch_size
        for _ in range(accum):
            xb = torch.rand(bs, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size, device=device) * 2 - 1
            (model.compute_loss(model(xb)).loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.denoiser.parameters()):
                e.lerp_(p, 1 - cfg.training.ema_decay)
        measured = measured_total_mib(device)
        _check(results, f"One training step ({bs} × {accum}) within the GPU memory budget",
               measured <= budget, f"{measured} MiB ≤ {budget} MiB" if measured <= budget else f"{measured} MiB > {budget} MiB")

    elapsed = time.time() - start
    passed = sum(ok for _, ok, _ in results)
    typer.echo("")
    typer.secho("Verification summary", bold=True)
    typer.echo(f"├── Checks passed: {passed}/{len(results)}")
    typer.echo(f"├── Parameters: {params:,} ({params / 1e6:.2f} M)")
    typer.echo(f"├── Measured GPU memory: {measured if measured is not None else 'n/a'} MiB (budget {budget})")
    note = "within" if elapsed < 120 else "exceeds"
    typer.echo(f"└── Elapsed: {elapsed:.1f} s ({note} the SC-001 target of 120 s)")
    if passed != len(results):
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------------------------------- train

@app.command()
@_handle_errors
def train(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True, help="Path to experiment YAML configuration."),
    resume: Optional[Path] = typer.Option(None, "-r", "--resume", help="Checkpoint (latest.pt or best_checkpoint.pt) to resume from."),
    seed: Optional[int] = typer.Option(None, "-s", "--seed", help="Override experiment random seed."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Override number of training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override per-step (micro-)batch size."),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum", help="Override gradient-accumulation steps."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override experiment.output_dir (e.g. a Google Drive folder)."),
    mixed_precision: Optional[str] = typer.Option(None, "--mixed-precision", help="Override training.mixed_precision: none, bf16 or fp16."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Start a fresh run even if output_dir already holds checkpoints (replaces them)."),
) -> None:
    """Train the DDPM; writes checkpoints, metrics.json, loss_curve.png and sample grids."""
    from src.configs.schema import apply_overrides
    from src.data.cifar10 import get_dataloaders_from_config
    from src.training.trainer import DDPMTrainer, NonFiniteLossError

    raw = apply_overrides(load_config(config), {
        "experiment.seed": seed,
        "training.epochs": epochs,
        "data.batch_size": batch_size,
        "training.grad_accum_steps": grad_accum,
        "experiment.output_dir": str(output_dir) if output_dir else None,
        "training.mixed_precision": mixed_precision,
    })
    validate_config(raw)
    device = _resolve_device(raw)
    train_loader, val_loader, _ = get_dataloaders_from_config(raw)
    trainer = DDPMTrainer(raw, train_loader=train_loader, val_loader=val_loader, device=device)
    start_epoch = 1
    if resume is not None:
        typer.echo(f"Resuming training from checkpoint: {resume}")
        start_epoch = trainer.resume_from_checkpoint(resume)
        if start_epoch > trainer.config["training"]["epochs"]:
            typer.secho(f"Checkpoint already reached epoch {start_epoch - 1}; raise --epochs to continue.",
                        fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
    try:
        result = trainer.train(start_epoch=start_epoch, overwrite=overwrite)
    except torch.cuda.OutOfMemoryError as exc:
        typer.secho("CUDA out of memory. Retry with a smaller per-step batch and more accumulation, "
                    "e.g. --batch-size 16 --grad-accum 8 (same effective batch).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except NonFiniteLossError as exc:
        typer.secho(f"{exc} best_checkpoint.pt was not modified.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    out = trainer.output_dir
    typer.secho("Training complete", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"├── Best EMA validation loss: {result['best_val_loss']:.4f}")
    typer.echo(f"├── Final epoch: {result['final_epoch']}")
    typer.echo(f"├── Best checkpoint: {out / 'best_checkpoint.pt'}")
    typer.echo(f"└── Metrics: {out / 'metrics.json'}")


# ------------------------------------------------------------------------------------ sampling helpers

def _load_for_inference(checkpoint: Path, weights: str) -> tuple[Any, dict[str, Any], torch.device]:
    from src.training.checkpoint import restore_model_from_checkpoint

    if weights not in ("ema", "raw"):
        raise ValueError("--weights must be 'ema' or 'raw'")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = _resolve_device({"experiment": {"device": "auto"}})
    model, ckpt = restore_model_from_checkpoint(checkpoint, map_location=device, weights=weights)
    return model, ckpt, device


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- sample

@app.command()
@_handle_errors
def sample(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", help="Path to .pt checkpoint file."),
    num_samples: int = typer.Option(64, "-n", "--num-samples", help="Number of images (square number recommended)."),
    seed: Optional[int] = typer.Option(None, "-s", "--seed", help="Sampling seed (default: config seed)."),
    batch_size: Optional[int] = typer.Option(None, "-b", "--batch-size", help="Images per reverse-chain batch (default: sampling.batch_size)."),
    weights: str = typer.Option("ema", "--weights", help="Weights to sample with: ema or raw."),
    out: Path = typer.Option(Path("artifacts/samples/sample_grid.png"), "-o", "--out", help="Output grid PNG path."),
    upscale: int = typer.Option(4, "--upscale", help="Tile upscaling factor (4 → 1024×1024 canvas for 64 images)."),
    save_individual: bool = typer.Option(False, "--save-individual", help="Also save each image as <out_stem>/NNNNN.png."),
    nearest: bool = typer.Option(False, "--nearest", help="Also build the nearest-neighbor memorization panel."),
    nearest_k: int = typer.Option(3, "--nearest-k", help="Training neighbors shown per generated image."),
    nearest_rows: int = typer.Option(64, "--nearest-rows", help="Generated images included in the panel (SC-010: ≥ 64)."),
) -> None:
    """Generate images with Algorithm 2 and save a grid (+ timing JSON, optional nearest-neighbor panel)."""
    import torchvision.utils as vutils

    from src.data.cifar10 import unnormalize
    from src.evaluation.visualizer import _grid_nrow, save_grid
    from src.utils.seeding import make_generator

    model, ckpt, device = _load_for_inference(checkpoint, weights)
    cfg = ckpt["config"]
    seed = cfg["experiment"]["seed"] if seed is None else seed
    batch_size = batch_size or cfg["sampling"]["batch_size"]
    stem_dir = out.with_suffix("")

    def on_batch(index: int, chunk: torch.Tensor) -> None:
        if save_individual:
            stem_dir.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(unnormalize(chunk)):
                vutils.save_image(img, str(stem_dir / f"{index * batch_size + j:05d}.png"))

    result = model.sample(num_samples, device, generator=make_generator(seed, device), batch_size=batch_size,
                          on_batch=on_batch)
    save_grid(unnormalize(result.samples), out, _grid_nrow(num_samples), upscale)
    timing = {
        "num_samples": num_samples, "seed": seed, "weights": weights, "checkpoint": str(checkpoint),
        "sampling_seconds_total": round(result.seconds, 3),
        "sampling_seconds_per_image": round(result.seconds / num_samples, 4),
        "device_name": _device_name(device), "timestamp": _utc_now(),
    }
    _write_json(out.with_name(out.stem + "_timing.json"), timing)

    typer.secho(f"Generated {num_samples} samples ({weights} weights, seed {seed})", fg=typer.colors.GREEN)
    typer.echo(f"├── Grid: {out}")
    typer.echo(f"├── Time: {result.seconds:.1f} s ({timing['sampling_seconds_per_image']:.3f} s/image)")
    if nearest:
        from src.evaluation.visualizer import find_nearest_neighbors, load_train_images, render_nearest_neighbors

        rows = min(nearest_rows, num_samples)
        samples01 = unnormalize(result.samples[:rows]).to(device)
        train01 = load_train_images(cfg["data"]["data_dir"])
        idx, dist = find_nearest_neighbors(samples01, train01, k=nearest_k)
        panel = render_nearest_neighbors(samples01, train01, idx, dist, out.with_name(out.stem + "_nearest.png"),
                                         upscale=min(upscale, 2))
        typer.echo(f"├── Nearest-neighbor panel: {panel} (min L2 distance {dist[:, 0].min().item():.3f})")
    typer.echo(f"└── Timing: {out.with_name(out.stem + '_timing.json')}")


# --------------------------------------------------------------------------------------- denoise-strip

@app.command("denoise-strip")
@_handle_errors
def denoise_strip(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", help="Path to .pt checkpoint file."),
    num_images: int = typer.Option(8, "--num-images", help="Number of independent chains (one row each)."),
    steps: Optional[str] = typer.Option(None, "--steps", help="Comma-separated paper timesteps (default: sampling.strip_steps)."),
    seed: Optional[int] = typer.Option(None, "-s", "--seed", help="Sampling seed (default: config seed)."),
    weights: str = typer.Option("ema", "--weights", help="Weights to sample with: ema or raw."),
    out: Path = typer.Option(Path("artifacts/strips/denoise_strip.png"), "-o", "--out", help="Output PNG path."),
    upscale: int = typer.Option(4, "--upscale", help="Tile upscaling factor."),
) -> None:
    """Render the reverse process: rows = images, columns = timesteps t=1000 … t=0."""
    from src.evaluation.visualizer import render_denoise_strip
    from src.utils.seeding import make_generator

    model, ckpt, device = _load_for_inference(checkpoint, weights)
    cfg = ckpt["config"]
    T = model.timesteps
    step_list = [int(s) for s in steps.split(",")] if steps else list(cfg["sampling"]["strip_steps"])
    if not all(0 <= s <= T for s in step_list) or any(a <= b for a, b in zip(step_list, step_list[1:])):
        raise ValueError(f"--steps must be strictly descending values within [0, {T}], got {step_list}")
    seed = cfg["experiment"]["seed"] if seed is None else seed
    result = model.sample(num_images, device, generator=make_generator(seed, device), trajectory_steps=step_list)
    path = render_denoise_strip(result.trajectory, step_list, out, upscale=upscale)
    typer.secho(f"Denoising strip saved: {path}", fg=typer.colors.GREEN)
    typer.echo(f"└── {num_images} chains × {len(step_list)} timesteps ({result.seconds:.1f} s)")


# -------------------------------------------------------------------------------------------- evaluate

def _test_loader(cfg: dict[str, Any], batch_size: int, data_dir: Optional[Path] = None) -> Any:
    from src.data.cifar10 import get_cifar10_dataloaders

    data = cfg["data"]
    return get_cifar10_dataloaders(
        data_dir=data_dir or data["data_dir"], batch_size=batch_size, val_split=data["val_split"],
        num_workers=data["num_workers"], seed=cfg["experiment"]["seed"], random_flip=False,
    )[2]


@app.command()
@_handle_errors
def evaluate(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", help="Path to .pt checkpoint file."),
    weights: str = typer.Option("ema", "--weights", help="Weights to evaluate: ema or raw."),
    batch_size: int = typer.Option(128, "-b", "--batch-size", help="Evaluation batch size."),
    data_dir: Optional[Path] = typer.Option(None, "-d", "--data-dir", help="Override dataset directory."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output JSON (default: <checkpoint_dir>/eval_metrics.json)."),
) -> None:
    """Average noise-prediction loss on the CIFAR-10 test split (fixed-seed, repeatable)."""
    from src.evaluation.evaluator import evaluate_model

    model, ckpt, device = _load_for_inference(checkpoint, weights)
    cfg = ckpt["config"]
    seed = cfg["experiment"]["seed"]
    metrics = evaluate_model(model, _test_loader(cfg, batch_size, data_dir), device, seed=seed)
    payload = {"timestamp": _utc_now(), "checkpoint": str(checkpoint), "weights": weights, "seed": seed,
               **metrics, "device_name": _device_name(device)}
    out = out or checkpoint.parent / "eval_metrics.json"
    _write_json(out, payload)
    typer.secho("Evaluation complete", fg=typer.colors.GREEN)
    typer.echo(f"├── Test noise-prediction loss: {metrics['test_loss']:.4f} ({metrics['total_samples']} images)")
    typer.echo(f"└── Metrics written to {out}")


# ------------------------------------------------------------------------------------------- benchmark

@app.command()
@_handle_errors
def benchmark(
    checkpoint: Path = typer.Option(..., "-k", "--checkpoint", help="Path to .pt checkpoint file."),
    num_samples: int = typer.Option(5000, "-n", "--num-samples", help="Generated and real image count for FID/IS."),
    batch_size: int = typer.Option(64, "-b", "--batch-size", help="Batch size for Inception feature extraction."),
    sample_batch_size: int = typer.Option(256, "--sample-batch-size", help="Images per reverse-chain batch."),
    seed: Optional[int] = typer.Option(None, "-s", "--seed", help="Sampling seed (default: config seed)."),
    weights: str = typer.Option("ema", "--weights", help="Weights to benchmark: ema or raw."),
    out: Path = typer.Option(Path("artifacts/eval/benchmark_metrics.json"), "-o", "--out", help="Output JSON path."),
    reuse_samples: Optional[Path] = typer.Option(None, "--reuse-samples", help="Directory of previously generated sample batches to reuse (resume)."),
) -> None:
    """Test loss + FID/IS with the shared Hands-on VAE protocol; writes benchmark_metrics.json."""
    import json

    from src.evaluation.evaluator import evaluate_model
    from src.evaluation.metrics import compute_fid_and_is

    model, ckpt, device = _load_for_inference(checkpoint, weights)
    cfg = ckpt["config"]
    seed = cfg["experiment"]["seed"] if seed is None else seed
    is_splits = cfg["evaluation"]["is_splits"]
    if num_samples % is_splits != 0:
        raise ValueError(f"--num-samples must be divisible by evaluation.is_splits={is_splits}")

    typer.echo("Step 1/2: test noise-prediction loss ...")
    test_loader = _test_loader(cfg, 128)
    eval_metrics = evaluate_model(model, test_loader, device, seed=cfg["experiment"]["seed"])

    typer.echo(f"Step 2/2: FID and Inception Score over {num_samples} samples ...")
    # Samples are always written to disk; they are only *reused* when --reuse-samples is given, so a
    # different checkpoint can never be scored with stale images.
    sample_dir = reuse_samples or out.parent / f"samples_{seed}"
    manifest = {"checkpoint": str(checkpoint.resolve()), "epoch": ckpt.get("epoch"),
                "global_step": ckpt.get("global_step"), "weights": weights, "seed": seed,
                "sample_batch_size": sample_batch_size}
    gen = compute_fid_and_is(model, test_loader, num_samples=num_samples, batch_size=batch_size, device=device,
                             sample_batch_size=sample_batch_size, seed=seed, sample_dir=sample_dir,
                             reuse_samples=reuse_samples is not None, is_splits=is_splits, manifest=manifest)

    metrics_file = checkpoint.parent / "metrics.json"
    training_seconds = None
    if metrics_file.exists():
        records = json.loads(metrics_file.read_text())
        training_seconds = round(sum(r.get("epoch_seconds", 0) for r in records), 1)

    payload = {
        "timestamp": _utc_now(), "checkpoint": str(checkpoint), "weights": weights, "seed": seed,
        **eval_metrics, **gen,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds, "device_name": _device_name(device),
    }
    _write_json(out, payload)
    hours = (gen["sampling_seconds_total"] or 0.0) / 3600
    typer.secho(f"Benchmark ({weights.upper()}, {gen['benchmark_samples']} samples, seed {seed})", bold=True)
    typer.echo(f"├── Test noise-prediction loss: {eval_metrics['test_loss']:.4f}")
    typer.echo(f"├── Fréchet Inception Distance: {gen['fid']:.2f}")
    typer.echo(f"├── Inception Score: {gen['inception_score_mean']:.2f} ± {gen['inception_score_std']:.2f}")
    if gen["generated_samples"]:
        typer.echo(f"├── Sampling time: {hours:.2f} h ({gen['sampling_seconds_per_image']:.2f} s/image, "
                   f"{gen['generated_samples']} generated, {gen['reused_batches']} batches reused)")
    else:
        typer.echo(f"├── Sampling time: n/a (all {gen['reused_batches']} batches reused from disk)")
    typer.echo(f"├── Class coverage: {gen['class_coverage']['distinct_top1_classes']} distinct Inception classes")
    typer.echo(f"└── Metrics written to {out}")


if __name__ == "__main__":
    app()
