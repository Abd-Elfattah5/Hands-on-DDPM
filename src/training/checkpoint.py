"""Checkpoint serialization, restoration and provenance tracking (data-model.md §4).

Ported from `Hands-on VAE/src/training/checkpoint.py` and extended with EMA weights, AMP scaler,
best validation loss, full metric history and RNG state, so `train --resume` continues exactly.

Layout: ``model_state_dict`` is the full ``GaussianDiffusion.state_dict()`` (raw weights, keys prefixed
``denoiser.``; schedule buffers are non-persistent and rebuilt from ``config``); ``ema_state_dict`` is
``{decay, num_updates, shadow}`` where ``shadow`` is the denoiser-only EMA state dict.
"""

import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.base import BaseDiffusion
from src.models.registry import build_diffusion_from_config
from src.training.ema import EMA

FORMAT_VERSION = "1.0"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
                                       timeout=5).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: BaseDiffusion,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
    seed: int,
) -> Path:
    """Write a checkpoint atomically (temporary file + ``os.replace``) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ckpt = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "best_val_loss": float(best_val_loss),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "config": config or getattr(model, "config", {}),
        "metrics": metrics or {},
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": device_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "git_commit": _git_commit(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict; raise ``FileNotFoundError`` / ``ValueError`` for missing or invalid files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a valid DDPM checkpoint (missing 'model_state_dict')")
    return ckpt


def restore_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    weights: str = "ema",
) -> tuple[BaseDiffusion, dict[str, Any]]:
    """Rebuild the model from the checkpoint config and load ``"ema"`` (default) or ``"raw"`` weights."""
    if weights not in ("ema", "raw"):
        raise ValueError(f"weights must be 'ema' or 'raw', got {weights!r}")
    ckpt = load_checkpoint(path, map_location)
    model = build_diffusion_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    if weights == "ema":
        if not ckpt.get("ema_state_dict"):
            raise ValueError(f"{path} has no EMA weights; use --weights raw")
        model.denoiser.load_state_dict(ckpt["ema_state_dict"]["shadow"])
    model.to(map_location)
    model.eval()
    return model, ckpt
