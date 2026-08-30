from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

State = dict[str, torch.Tensor]


def clone_state(
    state: Mapping[str, torch.Tensor], *, device: str | torch.device | None = None
) -> State:
    return {
        name: (
            tensor.detach().clone().to(device=device)
            if device is not None
            else tensor.detach().clone()
        )
        for name, tensor in state.items()
    }


def weighted_average(
    states: Iterable[Mapping[str, torch.Tensor]], weights: Iterable[float]
) -> State:
    states = list(states)
    weights = [float(weight) for weight in weights]
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    reference = states[0]
    output: State = {}
    for name, value in reference.items():
        if torch.is_floating_point(value) or torch.is_complex(value):
            accumulator = torch.zeros_like(value)
            for state, weight in zip(states, weights, strict=True):
                accumulator.add_(state[name].to(accumulator.device), alpha=weight / total)
            output[name] = accumulator
        else:
            best = max(range(len(weights)), key=weights.__getitem__)
            output[name] = states[best][name].detach().clone()
    return output


def state_delta(new: Mapping[str, torch.Tensor], old: Mapping[str, torch.Tensor]) -> State:
    return {
        name: new[name] - value
        for name, value in old.items()
        if torch.is_floating_point(value) or torch.is_complex(value)
    }


def add_delta(
    base: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    alpha: float = 1.0,
) -> State:
    output = clone_state(base)
    for name, value in delta.items():
        output[name].add_(value.to(output[name].device), alpha=float(alpha))
    return output


def zeros_like(state: Mapping[str, torch.Tensor]) -> State:
    return {
        name: torch.zeros_like(value)
        for name, value in state.items()
        if torch.is_floating_point(value) or torch.is_complex(value)
    }
