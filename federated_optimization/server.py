from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from federated_optimization.client import ClientResult
from federated_optimization.state import (
    State,
    add_delta,
    clone_state,
    state_delta,
    weighted_average,
    zeros_like,
)


class ServerOptimizer:
    """Base class for round-level server optimizers.

    ``parameter_names`` lets stateful optimizers distinguish trainable parameters from
    floating model buffers. The runner fills it from ``model.named_parameters()``.
    """

    parameter_names: set[str] | None = None

    def _parameter_delta(
        self,
        new: Mapping[str, torch.Tensor],
        old: Mapping[str, torch.Tensor],
    ) -> State:
        """Compute a model displacement restricted to trainable parameters."""
        delta = state_delta(new, old)
        if self.parameter_names is None:
            return delta
        return {name: value for name, value in delta.items() if name in self.parameter_names}

    def step(
        self,
        global_state: Mapping[str, torch.Tensor],
        results: Sequence[ClientResult],
        *,
        total_clients: int | None = None,
    ) -> State:
        raise NotImplementedError

    @staticmethod
    def _validate(results: Sequence[ClientResult]) -> None:
        if not results:
            raise ValueError("Cannot aggregate an empty client result list")


class FedAvg(ServerOptimizer):
    """Replace the global model with the sample-weighted mean client endpoint."""

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        return weighted_average(
            [result.state for result in results], [result.num_examples for result in results]
        )


class FedAvgM(ServerOptimizer):
    """FedAvg with persistent momentum on the average client displacement."""

    def __init__(self, *, server_lr: float = 1.0, momentum: float = 0.9) -> None:
        self.server_lr = float(server_lr)
        self.momentum = float(momentum)
        self.velocity: State | None = None

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        averaged = weighted_average(
            [result.state for result in results], [result.num_examples for result in results]
        )
        delta = self._parameter_delta(averaged, global_state)
        if self.velocity is None:
            # Server state is initialized lazily so this class also works with arbitrary
            # user models and parameter names.
            self.velocity = {name: torch.zeros_like(value) for name, value in delta.items()}
        for name, value in delta.items():
            self.velocity[name].mul_(self.momentum).add_(value)
        output = add_delta(global_state, self.velocity, self.server_lr)
        for name, value in averaged.items():
            if name not in self.velocity:
                output[name] = value
        return output


class FedOpt(ServerOptimizer):
    """FedAdagrad, FedAdam, and FedYogi using the FedOpt pseudo-gradient.

    Clients return model endpoints rather than gradients. FedOpt interprets the negative
    average displacement, ``-(w_clients - w_global)``, as a server pseudo-gradient and
    feeds it into an adaptive optimizer.
    """

    def __init__(
        self,
        variant: str,
        *,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        self.variant = variant.lower()
        if self.variant not in {"fedadagrad", "fedadam", "fedyogi"}:
            raise ValueError(f"Unsupported FedOpt variant: {variant}")
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.m: State | None = None
        self.v: State | None = None

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        averaged = weighted_average(
            [result.state for result in results], [result.num_examples for result in results]
        )
        client_delta = self._parameter_delta(averaged, global_state)
        if self.m is None:
            self.m = zeros_like(global_state)
            self.v = zeros_like(global_state)
        assert self.v is not None
        output = clone_state(global_state)
        for name, delta in client_delta.items():
            # Negating the client displacement makes the following equations use the
            # familiar gradient-descent sign convention.
            gradient = -delta
            self.m[name].mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            squared = gradient.square()
            if self.variant == "fedadagrad":
                # Adagrad accumulates all historical squared pseudo-gradients.
                self.v[name].add_(squared)
            elif self.variant == "fedadam":
                # Adam uses an exponential moving average of squared gradients.
                self.v[name].mul_(self.beta2).add_(squared, alpha=1.0 - self.beta2)
            else:
                # Yogi changes the second moment only in the direction indicated by the
                # current mismatch, limiting overly rapid growth of the denominator.
                self.v[name].addcmul_(
                    torch.sign(self.v[name] - squared), squared, value=-(1.0 - self.beta2)
                )
            output[name].addcdiv_(
                self.m[name].to(output[name].device),
                self.v[name].sqrt().add(self.tau).to(output[name].device),
                value=-self.server_lr,
            )
        for name, value in averaged.items():
            if name not in client_delta:
                output[name] = value
        return output


class SCAFFOLD(FedAvg):
    """FedAvg model aggregation plus persistent server control variates."""

    def __init__(self) -> None:
        self.global_control: State | None = None

    def step(self, global_state, results, *, total_clients=None) -> State:
        output = super().step(global_state, results, total_clients=total_clients)
        if total_clients is None or total_clients <= 0:
            raise ValueError("SCAFFOLD requires total_clients")
        deltas = [result.control_delta for result in results]
        if any(delta is None for delta in deltas):
            raise ValueError("SCAFFOLD client result is missing control_delta")
        if self.global_control is None:
            self.global_control = zeros_like(deltas[0] or {})
        # The original SCAFFOLD server update averages control changes over the full
        # population, so partial participation contributes a factor of 1/N (not 1/S).
        for delta in deltas:
            assert delta is not None
            for name, value in delta.items():
                self.global_control[name].add_(value, alpha=1.0 / total_clients)
        return output


class FedNova(ServerOptimizer):
    """Normalized averaging for the common plain-SGD local solver case.

    Dividing each displacement by its local step count removes objective inconsistency;
    multiplying by the weighted mean step count restores the round's update scale.
    """

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        weights = torch.tensor([result.num_examples for result in results], dtype=torch.float64)
        weights /= weights.sum()
        pairs = list(zip(weights, results, strict=True))
        average_steps = sum(float(weight) * result.steps for weight, result in pairs)
        parameter_template = {
            name: value
            for name, value in global_state.items()
            if self.parameter_names is None or name in self.parameter_names
        }
        normalized = zeros_like(parameter_template)
        for weight, result in pairs:
            if result.steps <= 0:
                raise ValueError("FedNova requires positive local step counts")
            delta = self._parameter_delta(result.state, global_state)
            for name, value in delta.items():
                normalized[name].add_(value, alpha=float(weight) / result.steps)
        output = add_delta(global_state, normalized, average_steps)
        averaged = weighted_average(
            [result.state for result in results], [result.num_examples for result in results]
        )
        for name, value in averaged.items():
            if name not in normalized:
                output[name] = value
        return output


class CoordinateMedian(ServerOptimizer):
    """Take an unweighted client median independently at every coordinate."""

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        output = clone_state(global_state)
        for name, value in global_state.items():
            if torch.is_floating_point(value):
                stacked = torch.stack([result.state[name].to(value.device) for result in results])
                output[name] = stacked.median(dim=0).values
            else:
                output[name] = results[0].state[name].detach().clone()
        return output


class TrimmedMean(ServerOptimizer):
    """Drop coordinate-wise low/high extremes before taking the mean."""

    def __init__(self, *, trim_ratio: float = 0.1) -> None:
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = float(trim_ratio)

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        trim = int(len(results) * self.trim_ratio)
        if 2 * trim >= len(results):
            raise ValueError("trim_ratio removes every client")
        output = clone_state(global_state)
        for name, value in global_state.items():
            if torch.is_floating_point(value):
                stacked = torch.stack([result.state[name].to(value.device) for result in results])
                # Sorting happens independently along the client dimension for every
                # parameter coordinate; it is not a ranking of whole client updates.
                sorted_values = stacked.sort(dim=0).values
                output[name] = sorted_values[trim : len(results) - trim].mean(dim=0)
            else:
                output[name] = results[0].state[name].detach().clone()
        return output


class Krum(ServerOptimizer):
    """Select the client update most consistent with its nearest peers."""

    def __init__(self, *, byzantine_clients: int = 1) -> None:
        self.byzantine_clients = int(byzantine_clients)

    def step(self, global_state, results, *, total_clients=None) -> State:
        self._validate(results)
        n = len(results)
        f = self.byzantine_clients
        if f < 0 or n < 2 * f + 3:
            raise ValueError("Krum requires n >= 2f + 3 participating clients")
        vectors = []
        for result in results:
            # Krum compares whole client updates, so flatten trainable tensors into one
            # vector while deliberately excluding buffers such as BatchNorm statistics.
            pieces = [
                (result.state[name] - value.cpu()).reshape(-1).float()
                for name, value in global_state.items()
                if torch.is_floating_point(value)
                and (self.parameter_names is None or name in self.parameter_names)
            ]
            vectors.append(torch.cat(pieces))
        scores = []
        neighbors = n - f - 2
        for i, vector in enumerate(vectors):
            distances = [
                torch.sum((vector - other) ** 2).item()
                for j, other in enumerate(vectors)
                if i != j
            ]
            # A Byzantine outlier is far from honest clients and therefore receives a
            # larger nearest-neighbor distance score.
            scores.append(sum(sorted(distances)[:neighbors]))
        return clone_state(results[min(range(n), key=scores.__getitem__)].state)


def create_server_optimizer(name: str, config: Mapping[str, Any] | None = None) -> ServerOptimizer:
    """Create a federated server rule from its configuration-friendly name."""
    options = dict(config or {})
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"fedavg", "fedprox", "fedsgd"}:
        return FedAvg()
    if key == "fedavgm":
        return FedAvgM(**options)
    if key in {"fedadagrad", "fedadam", "fedyogi"}:
        return FedOpt(key, **options)
    if key == "scaffold":
        return SCAFFOLD()
    if key == "fednova":
        return FedNova()
    if key in {"median", "fedmedian"}:
        return CoordinateMedian()
    if key in {"trimmedmean", "fedtrimmedmean"}:
        return TrimmedMean(**options)
    if key == "krum":
        return Krum(**options)
    choices = (
        "fedavg, fedsgd, fedavgm, fedprox, scaffold, fednova, fedadagrad, "
        "fedadam, fedyogi, fedmedian, fedtrimmedmean, krum"
    )
    raise ValueError(f"Unknown federated optimizer '{name}'. Available: {choices}")
