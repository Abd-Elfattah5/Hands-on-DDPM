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


if __name__ == "__main__":
    app()
