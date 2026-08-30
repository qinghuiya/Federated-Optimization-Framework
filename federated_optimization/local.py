from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "asgd": torch.optim.ASGD,
    "radam": torch.optim.RAdam,
    "nadam": torch.optim.NAdam,
}


def available_optimizers() -> tuple[str, ...]:
    """Return stable names accepted by :func:`create_optimizer`."""
    return tuple(sorted((*OPTIMIZERS, "momentum", "nesterov")))


def create_optimizer(
    parameters: Iterable[nn.Parameter], config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Create a fresh client optimizer from a small, explicit registry.

    ``momentum`` and ``nesterov`` are convenient SGD presets. Every other
    keyword is forwarded to the corresponding PyTorch optimizer, so advanced
    options remain available without framework changes.
    """
    options = dict(config)
    name = str(options.pop("name", "sgd")).lower().replace("-", "")
    if "lr" not in options:
        options["lr"] = 0.01
    if name == "momentum":
        name = "sgd"
        options.setdefault("momentum", 0.9)
    elif name == "nesterov":
        name = "sgd"
        options.setdefault("momentum", 0.9)
        options["nesterov"] = True
    if name not in OPTIMIZERS:
        choices = ", ".join(available_optimizers())
        raise ValueError(f"Unknown local optimizer '{name}'. Available: {choices}")
    return OPTIMIZERS[name](parameters, **options)


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any] | None
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create an optional local learning-rate schedule from configuration."""
    if not config:
        return None
    options = dict(config)
    name = str(options.pop("name", "none")).lower().replace("-", "_")
    if name in {"", "none", "constant"}:
        return None
    # Keep the registry explicit: typos fail early instead of being interpreted as an
    # arbitrary import path or silently falling back to a constant learning rate.
    schedulers: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "linear": torch.optim.lr_scheduler.LinearLR,
    }
    if name not in schedulers:
        raise ValueError(f"Unknown local scheduler '{name}'. Available: {', '.join(schedulers)}")
    return schedulers[name](optimizer, **options)
