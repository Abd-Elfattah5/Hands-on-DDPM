"""Exponential moving average (EMA) of network weights (research §6, deep dive §4.4).

After every optimizer step: θ_EMA ← decay · θ_EMA + (1 − decay) · θ. Buffers are copied, not averaged.
"""

import copy
from typing import Any

import torch
from torch import nn


class EMA:
    """Keeps a frozen shadow copy of a module whose weights follow the live weights slowly."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)

    @property
    def module(self) -> nn.Module:
        return self.shadow

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            if p.dtype.is_floating_point:
                e.lerp_(p.detach(), 1.0 - self.decay)
            else:
                e.copy_(p)
        for e, b in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(b)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow.state_dict())

    def to(self, device: torch.device | str) -> "EMA":
        self.shadow.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow.load_state_dict(state["shadow"])
