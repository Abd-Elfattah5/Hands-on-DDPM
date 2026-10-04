"""EMA weights: update maths, buffer copying, copy_to and state round trip."""

import torch
from torch import nn

from src.training.ema import EMA


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))


def test_update_formula_and_buffers():
    model = _net()
    ema = EMA(model, decay=0.9)
    old = [p.clone() for p in ema.module.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        model[1].running_mean.fill_(3.0)
    ema.update(model)
    for o, e, m in zip(old, ema.module.parameters(), model.parameters()):
        assert torch.allclose(e, 0.9 * o + 0.1 * m)
    assert torch.equal(ema.module[1].running_mean, model[1].running_mean)
    assert ema.num_updates == 1


def test_no_grad_and_copy_to():
    model = _net()
    ema = EMA(model, decay=0.5)
    assert all(not p.requires_grad for p in ema.module.parameters())
    target = _net()
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
    ema.copy_to(target)
    for a, b in zip(target.parameters(), ema.module.parameters()):
        assert torch.equal(a, b)


def test_state_round_trip():
    model = _net()
    ema = EMA(model, decay=0.99)
    ema.update(model)
    state = ema.state_dict()
    other = EMA(_net(), decay=0.5)
    other.load_state_dict(state)
    assert other.decay == 0.99 and other.num_updates == 1
    for a, b in zip(other.module.state_dict().values(), ema.module.state_dict().values()):
        assert torch.equal(a, b)
