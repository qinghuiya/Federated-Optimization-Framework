from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from federated_optimization.local import create_optimizer, create_scheduler
from federated_optimization.state import State, clone_state, zeros_like


@dataclass
class ClientResult:
    client_id: int
    state: State
    num_examples: int
    steps: int
    mean_loss: float
    control_delta: State | None = None


class ClientTrainer:
    """Reusable sequential client trainer.

    The trainer supports standard local optimization, FedProx's proximal term,
    and SCAFFOLD's control-variate gradient correction. A fresh optimizer is
    created for each client participation, matching the common stateless-client
    simulation protocol.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        *,
        client_id: int,
        loader,
        global_state: Mapping[str, torch.Tensor],
        optimizer_config: Mapping[str, Any],
        local_epochs: int | None = 1,
        local_steps: int | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        proximal_mu: float = 0.0,
        global_control: Mapping[str, torch.Tensor] | None = None,
        client_control: Mapping[str, torch.Tensor] | None = None,
    ) -> ClientResult:
        if (local_epochs is None) == (local_steps is None):
            raise ValueError("Specify exactly one of local_epochs and local_steps")
        if local_epochs is not None and local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if local_steps is not None and local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if proximal_mu < 0:
            raise ValueError("proximal_mu cannot be negative")

        device_state = {name: value.to(self.device) for name, value in global_state.items()}
        self.model.load_state_dict(device_state, strict=True)
        self.model.train()
        optimizer = create_optimizer(self.model.parameters(), optimizer_config)
        scheduler = create_scheduler(optimizer, scheduler_config)
        initial_parameters = {
            name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
        }
        use_scaffold = global_control is not None or client_control is not None
        if use_scaffold and (global_control is None or client_control is None):
            raise ValueError("SCAFFOLD requires both global_control and client_control")

        total_examples = 0
        weighted_loss = 0.0
        completed_steps = 0
        iterator = iter(loader)

        def update(batch) -> None:
            nonlocal total_examples, weighted_loss, completed_steps
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            if proximal_mu:
                penalty = torch.zeros((), device=self.device)
                for name, parameter in self.model.named_parameters():
                    penalty.add_(torch.sum((parameter - initial_parameters[name]) ** 2))
                loss = loss + 0.5 * proximal_mu * penalty
            loss.backward()
            if use_scaffold:
                assert global_control is not None and client_control is not None
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        correction = global_control[name].to(self.device) - client_control[name].to(
                            self.device
                        )
                        parameter.grad.add_(correction)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            count = int(targets.shape[0])
            total_examples += count
            weighted_loss += float(loss.detach().item()) * count
            completed_steps += 1

        if local_steps is not None:
            for _ in range(local_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(f"Client {client_id} has no training batches") from exc
                update(batch)
        else:
            assert local_epochs is not None
            for _ in range(local_epochs):
                seen_batch = False
                for batch in loader:
                    seen_batch = True
                    update(batch)
                if not seen_batch:
                    raise RuntimeError(f"Client {client_id} has no training batches")

        control_delta = None
        if use_scaffold:
            assert global_control is not None and client_control is not None
            learning_rate = float(optimizer_config.get("lr", 0.01))
            if learning_rate <= 0:
                raise ValueError("SCAFFOLD requires a positive client learning rate")
            control_delta = {}
            for name, parameter in self.model.named_parameters():
                old_control = client_control[name].to(self.device)
                new_control = (
                    old_control
                    - global_control[name].to(self.device)
                    + (initial_parameters[name] - parameter.detach())
                    / (completed_steps * learning_rate)
                )
                control_delta[name] = (new_control - old_control).cpu()

        return ClientResult(
            client_id=client_id,
            state=clone_state(self.model.state_dict(), device="cpu"),
            num_examples=max(int(len(loader.dataset)), 1),
            steps=completed_steps,
            mean_loss=weighted_loss / max(total_examples, 1),
            control_delta=control_delta,
        )


def empty_control(model: nn.Module) -> State:
    return zeros_like(dict(model.named_parameters()))
