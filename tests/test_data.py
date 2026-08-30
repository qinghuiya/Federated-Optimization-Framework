import numpy as np

from federated_optimization.data import dirichlet_partition, iid_partition, shard_partition


def assert_partition(partition, size):
    flattened = [index for client in partition for index in client]
    assert sorted(flattened) == list(range(size))
    assert len(flattened) == len(set(flattened))


def test_iid_partition_is_complete_and_deterministic():
    first = iid_partition(100, 7, 9)
    second = iid_partition(100, 7, 9)
    assert first == second
    assert_partition(first, 100)


def test_dirichlet_partition_respects_minimum():
    labels = np.repeat(np.arange(5), 40)
    partition = dirichlet_partition(labels, 10, alpha=0.1, seed=3, min_samples=5)
    assert min(map(len, partition)) >= 5
    assert_partition(partition, len(labels))


def test_shard_partition_is_complete():
    labels = np.repeat(np.arange(10), 20)
    partition = shard_partition(labels, 10, shards_per_client=2, seed=1)
    assert_partition(partition, len(labels))

