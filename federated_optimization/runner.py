"""End-to-end, single-process federated-learning experiment loop.

This module intentionally keeps orchestration explicit: select clients, train each
client from the same global checkpoint, aggregate their results, and evaluate the new
global model. Beginners can therefore follow one communication round without jumping
through callbacks or a distributed runtime.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from federated_optimization.client import ClientTrainer, empty_control
from federated_optimization.data import build_data
from federated_optimization.models import create_model
from federated_optimization.server import SCAFFOLD, create_server_optimizer
from federated_optimization.state import clone_state
from federated_optimization.utils import resolve_device, save_config, seed_everything


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Compute sample-mean cross-entropy and accuracy without building gradients."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    examples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(loss_fn(logits, targets).item())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        examples += int(targets.shape[0])
    return {"loss": loss_sum / max(examples, 1), "accuracy": correct / max(examples, 1)}


def clients_per_round(federated: dict[str, Any], total_clients: int) -> int:
    """Resolve an absolute client count from a count or participation fraction."""
    if "clients_per_round" in federated:
        count = int(federated["clients_per_round"])
    else:
        participation = float(federated.get("participation", 0.2))
        if not 0 < participation <= 1:
            raise ValueError("federated.participation must be in (0, 1]")
        count = max(1, round(total_clients * participation))
    if not 1 <= count <= total_clients:
        raise ValueError("clients_per_round must be between 1 and the total number of clients")
    return count


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run a configured FL simulation and persist reproducible artifacts.

    All clients are simulated sequentially on one device. This is easier to inspect than
    a distributed implementation while preserving the algorithmic communication-round
    semantics used by the included methods.
    """
    experiment = dict(config.get("experiment", {}))
    seed = int(experiment.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(experiment.get("device", "auto")))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "resolved_config.yaml")

    # Build the partition once. Reusing it across rounds is essential for a meaningful
    # FL simulation: each client represents one stable local data distribution.
    data = build_data(dict(config.get("data", {})), seed)
    model_config = dict(config.get("model", {"name": "mlp"}))
    model_name = str(model_config.pop("name", "mlp"))
    model = create_model(model_name, data.input_shape, data.num_classes, **model_config)
    # ClientTrainer owns one reusable worker model. Client states are still independent
    # because the worker is reset from global_state before every participation.
    trainer = ClientTrainer(model, device)
    global_state = clone_state(model.state_dict(), device="cpu")

    federated = dict(config.get("federated", {}))
    algorithm = str(federated.get("algorithm", "fedavg"))
    rounds = int(federated.get("rounds", 20))
    if rounds <= 0:
        raise ValueError("federated.rounds must be positive")
    selected_count = clients_per_round(federated, len(data.client_loaders))
    server = create_server_optimizer(algorithm, dict(federated.get("server", {})))
    # Adaptive server optimizers must update trainable parameters only. Floating buffers
    # such as BatchNorm running statistics are aggregated directly instead.
    server.parameter_names = {name for name, _ in model.named_parameters()}

    local = dict(config.get("local", {}))
    optimizer_config = dict(local.get("optimizer", {"name": "sgd", "lr": 0.05}))
    scheduler_config = local.get("scheduler")
    local_epochs = local.get("epochs", 1)
    local_steps = local.get("steps")
    if local_steps is not None:
        local_steps = int(local_steps)
        local_epochs = None
    elif local_epochs is not None:
        local_epochs = int(local_epochs)
    algorithm_key = algorithm.lower().replace("-", "").replace("_", "")
    if algorithm_key == "fedsgd":
        # In this simulator one FedSGD participation means exactly one local batch.
        local_steps, local_epochs = 1, None
    if algorithm_key in {"scaffold", "fednova"}:
        # The implemented closed-form corrections assume a constant plain-SGD step.
        # Rejecting unsupported combinations is safer than silently changing the math.
        optimizer_name = str(optimizer_config.get("name", "sgd")).lower()
        momentum = float(optimizer_config.get("momentum", 0.0))
        if optimizer_name != "sgd" or momentum != 0.0 or scheduler_config:
            raise ValueError(
                f"This {algorithm} implementation requires plain SGD without "
                "momentum or a scheduler"
            )
    proximal_mu = (
        float(federated.get("proximal_mu", 0.0)) if algorithm_key == "fedprox" else 0.0
    )

    # SCAFFOLD keeps one control variate per client, including clients that sit out a
    # round. Other algorithms leave these zero tensors unused.
    client_controls = [empty_control(model) for _ in data.client_loaders]
    if isinstance(server, SCAFFOLD):
        server.global_control = empty_control(model)

    rng = random.Random(seed)
    metrics_path = output / "metrics.jsonl"
    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for round_index in range(1, rounds + 1):
            # Sampling without replacement is deterministic for a fixed experiment seed.
            selected = sorted(rng.sample(range(len(data.client_loaders)), selected_count))
            results = []
            for client_id in selected:
                global_control = server.global_control if isinstance(server, SCAFFOLD) else None
                result = trainer.train(
                    client_id=client_id,
                    loader=data.client_loaders[client_id],
                    global_state=global_state,
                    optimizer_config=optimizer_config,
                    local_epochs=local_epochs,
                    local_steps=local_steps,
                    scheduler_config=scheduler_config,
                    proximal_mu=proximal_mu,
                    global_control=global_control,
                    client_control=(
                        client_controls[client_id] if global_control is not None else None
                    ),
                )
                results.append(result)

            # The server consumes only compact ClientResult objects. Different server
            # optimizers can therefore share the same client-training implementation.
            global_state = server.step(
                global_state, results, total_clients=len(data.client_loaders)
            )
            if isinstance(server, SCAFFOLD):
                # Persist each participating client's new control variate for its next
                # appearance. Non-participating clients deliberately keep old controls.
                for result in results:
                    assert result.control_delta is not None
                    for name, value in result.control_delta.items():
                        client_controls[result.client_id][name].add_(value)

            row: dict[str, Any] = {
                "round": round_index,
                "selected_clients": selected,
                "mean_client_loss": sum(result.mean_loss for result in results) / len(results),
            }
            # Evaluation can be less frequent than communication to reduce expensive
            # full-test-set passes on larger datasets.
            evaluate_every = int(experiment.get("evaluate_every", 1))
            if round_index % evaluate_every == 0 or round_index == rounds:
                model.load_state_dict(global_state)
                model.to(device)
                row.update(evaluate(model, data.test_loader, device))
            history.append(row)
            metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_file.flush()
            if bool(experiment.get("verbose", True)):
                accuracy = row.get("accuracy")
                suffix = f", accuracy={accuracy:.4f}" if accuracy is not None else ""
                print(f"round={round_index}/{rounds}, loss={row['mean_client_loss']:.4f}{suffix}")

    elapsed = time.perf_counter() - start
    final_metrics = next(row for row in reversed(history) if "accuracy" in row)
    summary = {
        "algorithm": algorithm,
        "rounds": rounds,
        "num_clients": len(data.client_loaders),
        "clients_per_round": selected_count,
        "device": str(device),
        "final_accuracy": final_metrics["accuracy"],
        "final_loss": final_metrics["loss"],
        "elapsed_seconds": elapsed,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    # A state_dict is more reusable than serializing a Python model object. Recreate the
    # configured model first, then load this checkpoint for inference or fine-tuning.
    torch.save(global_state, output / "final_model.pt")
    return summary
