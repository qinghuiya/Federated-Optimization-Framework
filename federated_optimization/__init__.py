"""Federated Optimization Framework public API."""

from federated_optimization.client import ClientResult, ClientTrainer
from federated_optimization.server import create_server_optimizer

__all__ = ["ClientResult", "ClientTrainer", "create_server_optimizer"]
__version__ = "1.0.0"

