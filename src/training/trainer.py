"""DDPM training loop: gradient accumulation, warmup, clipping, EMA, opt-in AMP, NaN/OOM guards.

Follows the `Hands-on VAE` checkpoint convention (latest / best / epoch_NNN / final) and restores the
full metric history on resume (research §6, §12; contracts/component-interfaces.md §3).
"""

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.configs.schema import ConfigError, resolve_config
from src.models.base import BaseDiffusion
from src.models.registry import build_diffusion_from_config
from src.training.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from src.training.ema import EMA
from src.utils.logging import get_logger
from src.utils.memory import measured_total_mib, reset_peak
from src.utils.seeding import make_generator, seed_everything


class NonFiniteLossError(RuntimeError):
    """Raised at the first NaN/Inf training loss (FR-018b)."""

    def __init__(self, epoch: int, step: int, mixed_precision: str) -> None:
        hint = " Mixed precision is enabled; retry with --mixed-precision none (fp32)." if mixed_precision != "none" else ""
        super().__init__(f"Non-finite training loss at epoch {epoch}, optimizer step {step}.{hint}")
        self.epoch, self.step = epoch, step


class DDPMTrainer:
    """Trains a ``GaussianDiffusion`` model and writes all artifacts to ``experiment.output_dir``."""

    def __init__(
        self,
        config: dict[str, Any],
        model: BaseDiffusion | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = resolve_config(config)
        exp, tr = self.config["experiment"], self.config["training"]
        self.seed = int(exp["seed"])
        seed_everything(self.seed)

        if device is None:
            want = exp["device"]
            device = torch.device("cuda" if want != "cpu" and torch.cuda.is_available() else "cpu")
        self.device = device

        self.output_dir = Path(exp["output_dir"])
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"ddpm.trainer.{self.output_dir}", log_file=self.output_dir / "train.log")
        with (self.output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)

        self.model = (model or build_diffusion_from_config(self.config)).to(self.device)
        self.ema = EMA(self.model.denoiser, decay=tr["ema_decay"]).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
        warmup = int(tr["warmup_steps"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
        )
        self.mixed_precision = tr["mixed_precision"]
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.mixed_precision == "fp16")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.mixed_precision)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = math.inf
        self.global_step = 0

    # ------------------------------------------------------------------------------------------ steps

    def _autocast(self) -> Any:
        return torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """One pass over the training loader with gradient accumulation."""
        from tqdm import tqdm

        tr = self.config["training"]
        accum = int(tr["grad_accum_steps"])
        self.model.train()
        if self.device.type == "cuda":
            reset_peak(self.device)
        n_batches = len(self.train_loader)
        usable = (n_batches // accum) * accum  # drop a trailing incomplete accumulation group
        total, count = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        bar = tqdm(self.train_loader, total=usable, leave=False,
                   desc=f"Epoch {epoch:03d}/{tr['epochs']:03d} [Train]")
        for i, (x0, _) in enumerate(bar):
            if i >= usable:
                break
            x0 = x0.to(self.device, non_blocking=True)
            with self._autocast():
                loss = self.model.compute_loss(self.model(x0)).loss
            if not torch.isfinite(loss):
                raise NonFiniteLossError(epoch, self.global_step, self.mixed_precision)
            self.scaler.scale(loss / accum).backward()
            total += loss.item()
            count += 1
            if (i + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tr["gradient_clip_val"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model.denoiser)
                self.global_step += 1
                bar.set_postfix(loss=f"{loss.item():.4f}")
        return {"train_loss": total / max(count, 1), "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def _loss_over(self, loader: DataLoader, denoiser: torch.nn.Module) -> float:
        original = self.model.denoiser
        self.model.denoiser = denoiser
        try:
            g = make_generator(self.seed, self.device)
            total, n = 0.0, 0
            for x0, _ in loader:
                x0 = x0.to(self.device, non_blocking=True)
                loss = self.model.compute_loss(self.model(x0, generator=g)).loss
                total += loss.item() * x0.shape[0]
                n += x0.shape[0]
            return total / max(n, 1)
        finally:
            self.model.denoiser = original

    def validate(self, epoch: int, use_ema: bool = True) -> dict[str, float]:
        """Fixed-seed validation L_simple for raw weights and (by default) EMA weights."""
        self.model.eval()
        out = {"val_loss": self._loss_over(self.val_loader, self.model.denoiser)}
        if use_ema:
            out["val_loss_ema"] = self._loss_over(self.val_loader, self.ema.module)
        return out

    # ----------------------------------------------------------------------------------- checkpoints

    def _save(self, name: str, epoch: int, record: dict[str, Any]) -> Path:
        return save_checkpoint(
            self.output_dir / name, self.model, self.ema, self.optimizer, self.scheduler, self.scaler,
            epoch, self.global_step, self.best_val_loss, self.history, self.config, record, self.seed,
        )

    def resume_from_checkpoint(self, path: str | Path) -> int:
        """Restore full training state from ``path``; return the next epoch to run."""
        ckpt = load_checkpoint(path, map_location=self.device)
        diffs = []
        for section in ("model", "diffusion"):
            saved, current = ckpt["config"].get(section, {}), self.config[section]
            diffs += [f"{section}.{k}: checkpoint={saved.get(k)!r} current={current.get(k)!r}"
                      for k in sorted(set(saved) | set(current)) if saved.get(k) != current.get(k)]
        if diffs:
            raise ConfigError("Config does not match the checkpoint being resumed: " + "; ".join(diffs))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = int(ckpt["global_step"])
        self.best_val_loss = float(ckpt.get("best_val_loss", math.inf))
        self.history = list(ckpt.get("history", []))
        restore_rng_state(ckpt.get("rng_state"))
        self.logger.info(f"Resumed from {path} at epoch {ckpt['epoch']} (global step {self.global_step})")
        return int(ckpt["epoch"]) + 1

    # ---------------------------------------------------------------------------------------- driver

    def _write_metrics(self) -> None:
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    def _save_sample_grid(self, epoch: int) -> None:
        from src.evaluation.visualizer import generate_sample_grid

        original = self.model.denoiser
        self.model.denoiser = self.ema.module
        try:
            generate_sample_grid(self.model, num_samples=64, device=self.device, seed=self.seed, upscale=1,
                                 out_path=self.output_dir / "samples" / f"epoch_{epoch:03d}.png",
                                 batch_size=self.config["sampling"]["batch_size"])
        finally:
            self.model.denoiser = original

    def train(self, start_epoch: int = 1) -> dict[str, Any]:
        """Run epochs ``start_epoch..training.epochs``, writing checkpoints, metrics and plots."""
        tr = self.config["training"]
        epochs = int(tr["epochs"])
        self.logger.info(
            f"Training on {self.device} | epochs {start_epoch}-{epochs} | batch {self.config['data']['batch_size']} "
            f"× accum {tr['grad_accum_steps']} | mixed precision {self.mixed_precision}"
        )
        epoch = start_epoch - 1
        try:
            for epoch in range(start_epoch, epochs + 1):
                t0 = time.time()
                train_stats = self.train_epoch(epoch)
                val_stats = self.validate(epoch)
                peak = measured_total_mib(self.device) if self.device.type == "cuda" else 0
                record = {
                    "epoch": epoch, "global_step": self.global_step, **train_stats, **val_stats,
                    "epoch_seconds": round(time.time() - t0, 2), "peak_memory_mib": peak,
                }
                self.history.append(record)
                improved = record["val_loss_ema"] < self.best_val_loss
                if improved:
                    self.best_val_loss = record["val_loss_ema"]
                self._save("latest.pt", epoch, record)
                if improved:
                    self._save("best_checkpoint.pt", epoch, record)
                    self._save("best.pt", epoch, record)
                if epoch % tr["save_every"] == 0:
                    self._save(f"epoch_{epoch:03d}.pt", epoch, record)
                if epoch % tr["sample_every"] == 0:
                    self._save_sample_grid(epoch)
                self._write_metrics()
                self.logger.info(
                    f"Epoch {epoch:03d} | train {record['train_loss']:.4f} | val {record['val_loss']:.4f} | "
                    f"val EMA {record['val_loss_ema']:.4f} | lr {record['lr']:.2e} | "
                    f"{record['epoch_seconds']:.0f}s | {peak} MiB" + (" | best" if improved else "")
                )
        except (NonFiniteLossError, torch.cuda.OutOfMemoryError) as exc:
            self.logger.error(f"Training stopped: {exc}. latest.pt holds the last finite epoch; "
                              f"best_checkpoint.pt is unchanged.")
            raise

        final = self.history[-1] if self.history else {}
        self._save("final_checkpoint.pt", epoch, final)
        self._plot_loss_curves()
        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": epoch,
            "history": self.history,
            "best_checkpoint": str(self.output_dir / "best_checkpoint.pt"),
            "latest_checkpoint": str(self.output_dir / "latest.pt"),
        }

    def _plot_loss_curves(self) -> None:
        """Train / validation / EMA-validation L_simple per epoch → loss_curve.png."""
        if not self.history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in self.history]
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in (("train_loss", "Train"), ("val_loss", "Validation (raw)"), ("val_loss_ema", "Validation (EMA)")):
            ax.plot(epochs, [r[key] for r in self.history], marker="o", ms=3, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L_simple (noise MSE)")
        ax.set_title("DDPM training loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
