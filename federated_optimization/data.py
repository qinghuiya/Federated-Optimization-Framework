from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


@dataclass
class DataBundle:
    """Datasets after partitioning, ready for the experiment runner."""

    client_loaders: list[DataLoader]
    test_loader: DataLoader
    num_classes: int
    input_shape: tuple[int, ...]
    client_indices: list[list[int]]


def labels_of(dataset: Dataset) -> np.ndarray:
    """Read class labels from common datasets, with a generic fallback."""
    labels = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if labels is None:
        labels = [dataset[index][1] for index in range(len(dataset))]
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels, dtype=np.int64)


def iid_partition(size: int, num_clients: int, seed: int) -> list[list[int]]:
    """Shuffle once and split examples as evenly as possible across clients."""
    if num_clients <= 0 or num_clients > size:
        raise ValueError("num_clients must be in [1, dataset size]")
    rng = np.random.default_rng(seed)
    return [chunk.tolist() for chunk in np.array_split(rng.permutation(size), num_clients)]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
) -> list[list[int]]:
    """Create label-skew using one Dirichlet draw per class.

    A smaller ``alpha`` concentrates each class on fewer clients and therefore produces
    stronger statistical heterogeneity.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    if num_clients <= 0 or num_clients * min_samples > len(labels):
        raise ValueError("The requested minimum client size is impossible")
    rng = np.random.default_rng(seed)
    clients: list[list[int]] = [[] for _ in range(num_clients)]
    # Allocate every class separately. This preserves every example exactly once while
    # allowing class proportions to differ substantially between clients.
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        counts = rng.multinomial(
            len(indices), rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        )
        offset = 0
        for client_id, count in enumerate(counts):
            clients[client_id].extend(indices[offset : offset + count].tolist())
            offset += count
    # Very small alpha values can create empty clients. Repair them deterministically by
    # moving examples from the current largest client.
    for client_id in range(num_clients):
        while len(clients[client_id]) < min_samples:
            donor = max(range(num_clients), key=lambda index: len(clients[index]))
            if len(clients[donor]) <= min_samples:
                raise RuntimeError("Could not repair an undersized Dirichlet partition")
            clients[client_id].append(clients[donor].pop())
    for indices in clients:
        rng.shuffle(indices)
    return clients


def shard_partition(
    labels: np.ndarray, num_clients: int, shards_per_client: int, seed: int
) -> list[list[int]]:
    """Create the classic pathological split from label-sorted shards."""
    if shards_per_client <= 0:
        raise ValueError("shards_per_client must be positive")
    shard_count = num_clients * shards_per_client
    if shard_count > len(labels):
        raise ValueError("There are more shards than examples")
    # Sorting by label makes most shards contain only one or a few classes.
    sorted_indices = np.argsort(labels, kind="stable")
    shards = [chunk.tolist() for chunk in np.array_split(sorted_indices, shard_count)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(shard_count)
    clients = [[] for _ in range(num_clients)]
    for position, shard_id in enumerate(order):
        clients[position // shards_per_client].extend(shards[shard_id])
    return clients


def make_synthetic_dataset(config: dict[str, Any], seed: int):
    """Generate a download-free classification problem from Gaussian clusters."""
    samples = int(config.get("samples", 2000))
    features = int(config.get("features", 20))
    classes = int(config.get("classes", 4))
    test_fraction = float(config.get("test_fraction", 0.2))
    if samples < 10 or features <= 0 or classes < 2 or not 0 < test_fraction < 1:
        raise ValueError("Invalid synthetic dataset configuration")
    generator = torch.Generator().manual_seed(seed)
    # Each label owns a random feature-space center; unit Gaussian noise adds overlap.
    centers = torch.randn(classes, features, generator=generator) * 2.5
    labels = torch.randint(classes, (samples,), generator=generator)
    inputs = centers[labels] + torch.randn(samples, features, generator=generator)
    order = torch.randperm(samples, generator=generator)
    test_size = int(samples * test_fraction)
    test_ids, train_ids = order[:test_size], order[test_size:]
    return (
        TensorDataset(inputs[train_ids], labels[train_ids]),
        TensorDataset(inputs[test_ids], labels[test_ids]),
        classes,
        (features,),
    )


def load_dataset(config: dict[str, Any], seed: int):
    """Load a supported dataset and return its classification metadata."""
    name = str(config.get("name", "synthetic")).lower().replace("-", "")
    if name == "synthetic":
        return make_synthetic_dataset(config, seed)

    from torchvision import datasets, transforms

    root = str(Path(config.get("root", "data")).expanduser())
    if name in {"mnist", "fashionmnist"}:
        # Normalization constants follow the standard MNIST scale. FashionMNIST uses the
        # same channel shape, which keeps this beginner example intentionally compact.
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset_class = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        return (
            dataset_class(root, train=True, download=True, transform=transform),
            dataset_class(root, train=False, download=True, transform=transform),
            10,
            (1, 28, 28),
        )
    if name in {"cifar10", "cifar100"}:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset_class = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        classes = 10 if name == "cifar10" else 100
        return (
            dataset_class(root, train=True, download=True, transform=transform),
            dataset_class(root, train=False, download=True, transform=transform),
            classes,
            (3, 32, 32),
        )
    raise ValueError("Supported datasets: synthetic, MNIST, FashionMNIST, CIFAR-10, CIFAR-100")


def build_data(config: dict[str, Any], seed: int) -> DataBundle:
    """Load, partition, and wrap one dataset in deterministic DataLoaders."""
    train, test, num_classes, input_shape = load_dataset(config, seed)
    num_clients = int(config.get("num_clients", 20))
    partition = dict(config.get("partition", {"name": "dirichlet", "alpha": 0.5}))
    partition_name = str(partition.get("name", "dirichlet")).lower()
    labels = labels_of(train)
    if partition_name == "iid":
        indices = iid_partition(len(train), num_clients, seed)
    elif partition_name == "dirichlet":
        indices = dirichlet_partition(
            labels,
            num_clients,
            float(partition.get("alpha", 0.5)),
            seed,
            int(partition.get("min_samples", 1)),
        )
    elif partition_name in {"shard", "pathological"}:
        indices = shard_partition(
            labels, num_clients, int(partition.get("shards_per_client", 2)), seed
        )
    else:
        raise ValueError("partition.name must be iid, dirichlet, or shard")

    batch_size = int(config.get("batch_size", 32))
    client_loaders = []
    for client_id, client_indices in enumerate(indices):
        # A client-specific generator makes shuffling reproducible without coupling one
        # client's batch order to another client's participation.
        generator = torch.Generator().manual_seed(seed + client_id)
        client_loaders.append(
            DataLoader(
                Subset(train, client_indices),
                batch_size=min(batch_size, max(len(client_indices), 1)),
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
        )
    test_loader = DataLoader(
        test,
        batch_size=int(config.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=0,
    )
    return DataBundle(client_loaders, test_loader, num_classes, input_shape, indices)
