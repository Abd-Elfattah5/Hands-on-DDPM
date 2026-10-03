"""DDPMTrainer on CPU with a tiny config: artifacts, step counting, accumulation equivalence,
resume with history, NaN guard, config-mismatch guard and seeded determinism."""

import copy
import json
from pathlib import Path

import pytest
import torch

from src.configs.schema import ConfigError
from src.models.registry import build_diffusion_from_config
from src.training.trainer import DDPMTrainer, NonFiniteLossError

RECORD_KEYS = {"epoch", "global_step", "lr", "train_loss", "val_loss", "val_loss_ema", "epoch_seconds", "peak_memory_mib"}


def _trainer(cfg, loaders, **kw):
    train, val, _ = loaders
    return DDPMTrainer(cfg, train_loader=train, val_loader=val, device=torch.device("cpu"), **kw)


def test_two_epochs_artifacts(tiny_config, fake_cifar_loaders):
    out = Path(tiny_config["experiment"]["output_dir"])
    tiny_config["training"]["sample_every"] = 2
    tiny_config["diffusion"]["timesteps"] = 10  # keep the 64-image epoch grid fast on CPU
    tiny_config["sampling"]["strip_steps"] = [10, 0]
    result = _trainer(tiny_config, fake_cifar_loaders).train()
    for name in ["latest.pt", "best_checkpoint.pt", "best.pt", "final_checkpoint.pt", "metrics.json",
                 "loss_curve.png", "train.log", "resolved_config.yaml", "epoch_001.pt", "samples/epoch_002.png"]:
        assert (out / name).exists(), name
    records = json.loads((out / "metrics.json").read_text())
    assert len(records) == 2 and RECORD_KEYS <= set(records[0])
    assert result["final_epoch"] == 2


def test_global_step_counts_optimizer_steps(tiny_config, fake_cifar_loaders):
    trainer = _trainer(tiny_config, fake_cifar_loaders)
    trainer.train()
    per_epoch = len(fake_cifar_loaders[0]) // tiny_config["training"]["grad_accum_steps"]
    assert trainer.global_step == 2 * per_epoch


def test_accumulation_equivalence(tiny_config):
    """batch 4 × accum 2 must give the same update as batch 8 × accum 1 (dropout 0)."""
    tiny_config["model"]["dropout"] = 0.0
    torch.manual_seed(0)
    base = build_diffusion_from_config(tiny_config)
    for p in base.parameters():  # avoid zero-init making the comparison trivial
        torch.nn.init.normal_(p, std=0.02)
    x = torch.rand(8, 3, 32, 32) * 2 - 1

    def step(chunks):
        model = copy.deepcopy(base)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(5)
        t = torch.randint(0, model.timesteps, (8,), generator=g)
        noise = torch.randn(x.shape, generator=g)
        for idx in torch.arange(8).chunk(chunks):
            pred = model.denoiser(model.q_sample(x[idx], t[idx], noise[idx]), t[idx])
            (torch.nn.functional.mse_loss(pred, noise[idx]) / chunks).backward()
        opt.step()
        return [p.detach().clone() for p in model.parameters()]

    for a, b in zip(step(2), step(1)):
        assert torch.allclose(a, b, atol=1e-5)


def test_resume_keeps_history(tiny_config, fake_cifar_loaders, tmp_path):
    full_cfg = copy.deepcopy(tiny_config)
    full_cfg["experiment"]["output_dir"] = str(tmp_path / "full")
    full = _trainer(full_cfg, fake_cifar_loaders)
    full.train()

    part_cfg = copy.deepcopy(tiny_config)
    part_cfg["experiment"]["output_dir"] = str(tmp_path / "part")
    part_cfg["training"]["epochs"] = 1
    _trainer(part_cfg, fake_cifar_loaders).train()
    part_cfg["training"]["epochs"] = 2
    resumed = _trainer(part_cfg, fake_cifar_loaders)
    start = resumed.resume_from_checkpoint(tmp_path / "part" / "latest.pt")
    assert start == 2
    resumed.train(start_epoch=start)
    records = json.loads((tmp_path / "part" / "metrics.json").read_text())
    assert [r["epoch"] for r in records] == [1, 2]
    assert resumed.global_step == full.global_step


def test_nan_guard_keeps_best(tiny_config, fake_cifar_loaders, monkeypatch):
    tiny_config["training"]["epochs"] = 1
    trainer = _trainer(tiny_config, fake_cifar_loaders)
    trainer.train()
    best = Path(tiny_config["experiment"]["output_dir"]) / "best_checkpoint.pt"
    before = best.read_bytes()

    tiny_config["training"]["epochs"] = 2
    resumed = _trainer(tiny_config, fake_cifar_loaders)
    start = resumed.resume_from_checkpoint(best.parent / "latest.pt")
    from src.models.types import LossOutput
    monkeypatch.setattr(resumed.model, "compute_loss", lambda out: LossOutput(loss=out.predicted_noise.sum() * float("nan")))
    with pytest.raises(NonFiniteLossError):
        resumed.train(start_epoch=start)
    assert best.read_bytes() == before


def test_resume_config_mismatch(tiny_config, fake_cifar_loaders):
    tiny_config["training"]["epochs"] = 1
    _trainer(tiny_config, fake_cifar_loaders).train()
    other = copy.deepcopy(tiny_config)
    other["model"]["channels"] = [32, 32]
    trainer = _trainer(other, fake_cifar_loaders)
    with pytest.raises(ConfigError, match="channels"):
        trainer.resume_from_checkpoint(Path(tiny_config["experiment"]["output_dir"]) / "latest.pt")


def test_seeded_runs_are_identical(tiny_config, fake_loader_factory, tmp_path):
    """Spec US2 scenario 5: same seed and config → identical recorded losses."""
    losses = []
    for name in ("a", "b"):
        cfg = copy.deepcopy(tiny_config)
        cfg["experiment"]["output_dir"] = str(tmp_path / name)
        cfg["training"]["epochs"] = 1
        _trainer(cfg, fake_loader_factory()).train()
        rec = json.loads((tmp_path / name / "metrics.json").read_text())[0]
        losses.append((rec["train_loss"], rec["val_loss"], rec["val_loss_ema"]))
    assert losses[0] == losses[1]


def test_fresh_run_refuses_existing_checkpoints(tiny_config, fake_cifar_loaders):
    """Review finding 3: a run without --resume must not overwrite an existing run's checkpoints."""
    from src.training.trainer import RunDirectoryNotEmptyError

    tiny_config["training"]["epochs"] = 1
    _trainer(tiny_config, fake_cifar_loaders).train()
    best = Path(tiny_config["experiment"]["output_dir"]) / "best_checkpoint.pt"
    before = best.read_bytes()
    with pytest.raises(RunDirectoryNotEmptyError, match="--resume"):
        _trainer(tiny_config, fake_cifar_loaders).train()
    assert best.read_bytes() == before
    _trainer(tiny_config, fake_cifar_loaders).train(overwrite=True)
