"""Checkpoint schema (data-model.md §4), EMA vs raw restore and invalid files."""

import torch

from src.models.registry import build_diffusion_from_config
from src.training.checkpoint import load_checkpoint, restore_model_from_checkpoint, save_checkpoint
from src.training.ema import EMA

KEYS = {"format_version", "epoch", "global_step", "model_state_dict", "ema_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "scaler_state_dict", "best_val_loss", "history", "rng_state", "config",
        "metrics", "provenance"}
PROVENANCE = {"torch_version", "cuda_available", "device_name", "timestamp", "seed", "git_commit"}


def _setup(cfg):
    model = build_diffusion_from_config(cfg)
    ema = EMA(model.denoiser, decay=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.compute_loss(model(torch.rand(2, 3, 32, 32) * 2 - 1)).loss
    loss.backward()
    opt.step()
    ema.update(model.denoiser)
    return model, ema, opt


def test_schema_and_restore(tiny_config, tmp_path):
    model, ema, opt = _setup(tiny_config)
    path = save_checkpoint(tmp_path / "c.pt", model, ema, opt, None, None, epoch=1, global_step=3,
                           best_val_loss=0.5, history=[{"epoch": 1}], config=model.config,
                           metrics={"epoch": 1}, seed=42)
    ckpt = load_checkpoint(path)
    assert KEYS <= set(ckpt)
    assert PROVENANCE <= set(ckpt["provenance"])
    assert any(k.startswith("denoiser.") for k in ckpt["model_state_dict"])

    raw, _ = restore_model_from_checkpoint(path, weights="raw")
    emaw, _ = restore_model_from_checkpoint(path, weights="ema")
    assert not raw.training and not emaw.training
    for (n, a), b in zip(raw.denoiser.state_dict().items(), model.denoiser.state_dict().values()):
        assert torch.equal(a, b), n
    for a, b in zip(emaw.denoiser.state_dict().values(), ema.module.state_dict().values()):
        assert torch.equal(a, b)
    differs = any(not torch.equal(a, b) for a, b in zip(raw.denoiser.parameters(), emaw.denoiser.parameters()))
    assert differs


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"hello": 1}, bad)
    try:
        load_checkpoint(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
