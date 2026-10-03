"""Strongly-typed containers exchanged between diffusion components (data-model.md §2)."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class DiffusionOutput:
    """Output of a training forward pass.

    Attributes:
        x_t: Noised input ``[B, C, H, W]`` at the sampled timesteps.
        t: Zero-based timestep indices ``[B]`` (int64, values in ``[0, T-1]``).
        noise: Target noise ``ε ~ N(0, I)`` ``[B, C, H, W]``.
        predicted_noise: Network prediction ``ε_θ(x_t, t)`` ``[B, C, H, W]``.
        extra: Optional diagnostics.
    """

    x_t: Tensor
    t: Tensor
    noise: Tensor
    predicted_noise: Tensor
    extra: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class LossOutput:
    """Scalar training loss (L_simple) plus detached logging metrics."""

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SamplingOutput:
    """Result of reverse sampling (Algorithm 2).

    Attributes:
        samples: Final ``x_0`` ``[N, C, H, W]`` clamped to ``[-1, 1]``.
        trajectory: Captured frames ``[F, N, C, H, W]`` or ``None``.
        trajectory_steps: Paper timestep labels of the frames (e.g. ``[1000, ..., 0]``).
        seconds: Wall-clock sampling time.
    """

    samples: Tensor
    trajectory: Tensor | None
    trajectory_steps: list[int]
    seconds: float
