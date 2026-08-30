from __future__ import annotations

import argparse
import json
from pathlib import Path

from federated_optimization.runner import run_experiment
from federated_optimization.utils import load_config


def build_parser() -> argparse.ArgumentParser:
    """Define a small set of convenient overrides for the YAML configuration."""
    parser = argparse.ArgumentParser(description="Federated Optimization Framework trainer")
    parser.add_argument("--config", default="configs/synthetic_fedavg.yaml")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--algorithm", help="Override federated.algorithm")
    parser.add_argument("--optimizer", help="Override local.optimizer.name")
    parser.add_argument("--rounds", type=int, help="Override federated.rounds")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> int:
    """Load configuration, apply explicit CLI overrides, and start one run."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    # CLI values intentionally override YAML only when explicitly provided. This keeps
    # the resolved configuration truthful and makes shell-based comparisons convenient.
    if args.algorithm:
        config.setdefault("federated", {})["algorithm"] = args.algorithm
    if args.optimizer:
        config.setdefault("local", {}).setdefault("optimizer", {})["name"] = args.optimizer
    if args.rounds is not None:
        config.setdefault("federated", {})["rounds"] = args.rounds
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
    if args.device:
        config.setdefault("experiment", {})["device"] = args.device
    summary = run_experiment(config, Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
