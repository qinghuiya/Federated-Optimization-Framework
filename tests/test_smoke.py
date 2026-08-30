from pathlib import Path

import pytest

from federated_optimization.runner import run_experiment


@pytest.mark.parametrize(
    "algorithm",
    [
        "fedsgd",
        "fedavg",
        "fedavgm",
        "fedprox",
        "scaffold",
        "fednova",
        "fedadagrad",
        "fedadam",
        "fedyogi",
        "fedmedian",
        "fedtrimmedmean",
        "krum",
    ],
)
def test_algorithm_end_to_end_smoke(tmp_path: Path, algorithm: str):
    server = {}
    if algorithm == "krum":
        server["byzantine_clients"] = 1
    if algorithm == "fedtrimmedmean":
        server["trim_ratio"] = 0.2
    config = {
        "experiment": {"seed": 0, "device": "cpu", "verbose": False},
        "data": {
            "name": "synthetic",
            "samples": 120,
            "features": 5,
            "classes": 2,
            "num_clients": 5,
            "batch_size": 8,
            "partition": {"name": "iid"},
        },
        "model": {"name": "mlp", "hidden": 8},
        "local": {"steps": 2, "optimizer": {"name": "sgd", "lr": 0.05}},
        "federated": {
            "algorithm": algorithm,
            "rounds": 1,
            "participation": 1.0,
            "proximal_mu": 0.1,
            "server": server,
        },
    }
    summary = run_experiment(config, tmp_path / algorithm)
    assert 0 <= summary["final_accuracy"] <= 1
    assert (tmp_path / algorithm / "metrics.jsonl").is_file()

