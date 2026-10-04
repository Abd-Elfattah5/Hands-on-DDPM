"""Decorator-based component registries and the config-driven diffusion factory.

Mirrors the `Hands-on VAE` registry pattern (contracts/component-interfaces.md §2).
"""

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from src.configs.schema import resolve_config
from src.models.base import BaseDenoiser, BaseDiffusion, BaseNoiseSchedule

SCHEDULE_REGISTRY: dict[str, type[BaseNoiseSchedule]] = {}
DENOISER_REGISTRY: dict[str, type[BaseDenoiser]] = {}
DIFFUSION_REGISTRY: dict[str, type[BaseDiffusion]] = {}

S = TypeVar("S", bound=type[BaseNoiseSchedule])
D = TypeVar("D", bound=type[BaseDenoiser])
P = TypeVar("P", bound=type[BaseDiffusion])

_DEFAULT_MODULES = ("src.models.schedules", "src.models.unet", "src.models.gaussian_diffusion")


def _make_register(registry: dict, base: type, kind: str) -> Callable[[str], Callable[[Any], Any]]:
    def register(name: str) -> Callable[[Any], Any]:
        def decorator(cls: Any) -> Any:
            if not isinstance(cls, type) or not issubclass(cls, base):
                raise TypeError(f"{kind} '{name}' must subclass {base.__name__}, got {cls!r}")
            registry[name] = cls
            return cls
        return decorator
    return register


register_schedule: Callable[[str], Callable[[S], S]] = _make_register(SCHEDULE_REGISTRY, BaseNoiseSchedule, "Schedule")
register_denoiser: Callable[[str], Callable[[D], D]] = _make_register(DENOISER_REGISTRY, BaseDenoiser, "Denoiser")
register_diffusion: Callable[[str], Callable[[P], P]] = _make_register(DIFFUSION_REGISTRY, BaseDiffusion, "Diffusion")


def _ensure_default_components() -> None:
    """Lazily import built-in component modules so their decorators register."""
    for module in _DEFAULT_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module:  # a real missing dependency inside the module
                raise


def _lookup(registry: dict, name: str, kind: str) -> Any:
    if name not in registry:
        _ensure_default_components()
    if name not in registry:
        raise KeyError(f"{kind.capitalize()} '{name}' not found. Available {kind}s: {sorted(registry)}")
    return registry[name]


def get_schedule(name: str) -> type[BaseNoiseSchedule]:
    return _lookup(SCHEDULE_REGISTRY, name, "schedule")


def get_denoiser(name: str) -> type[BaseDenoiser]:
    return _lookup(DENOISER_REGISTRY, name, "denoiser")


def get_diffusion(name: str) -> type[BaseDiffusion]:
    return _lookup(DIFFUSION_REGISTRY, name, "diffusion")


def build_diffusion_from_config(config: dict[str, Any]) -> BaseDiffusion:
    """Validate ``config`` and build schedule → denoiser → diffusion.

    The resolved config (all defaults filled in) is stored on the returned model as ``model.config``
    so checkpoints can rebuild it exactly.
    """
    resolved = resolve_config(config)
    diff_cfg = dict(resolved["diffusion"])
    model_cfg = dict(resolved["model"])

    schedule = get_schedule(diff_cfg["schedule"])(**diff_cfg)
    denoiser = get_denoiser(model_cfg.pop("name"))(**model_cfg)
    diffusion = get_diffusion(diff_cfg["name"])(
        schedule=schedule, denoiser=denoiser, variance_type=diff_cfg["variance_type"]
    )
    diffusion.config = resolved
    return diffusion
