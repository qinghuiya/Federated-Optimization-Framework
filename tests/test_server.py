import torch

from federated_optimization.client import ClientResult
from federated_optimization.server import create_server_optimizer


def state(value):
    return {"weight": torch.tensor([float(value)]), "counter": torch.tensor(0)}


def result(value, examples=1, steps=1, client_id=0):
    return ClientResult(client_id, state(value), examples, steps, 0.0)


def test_fedavg_is_sample_weighted():
    server = create_server_optimizer("fedavg")
    output = server.step(state(0), [result(1, 1), result(3, 3)])
    assert torch.allclose(output["weight"], torch.tensor([2.5]))


def test_fedavgm_tracks_server_velocity():
    server = create_server_optimizer("fedavgm", {"server_lr": 1.0, "momentum": 0.5})
    first = server.step(state(0), [result(2)])
    second = server.step(first, [result(3)])
    assert torch.allclose(first["weight"], torch.tensor([2.0]))
    assert torch.allclose(second["weight"], torch.tensor([4.0]))


def test_fednova_matches_fedavg_when_local_steps_are_equal():
    results = [result(1, examples=1, steps=4), result(3, examples=3, steps=4)]
    avg = create_server_optimizer("fedavg").step(state(0), results)
    nova = create_server_optimizer("fednova").step(state(0), results)
    assert torch.allclose(avg["weight"], nova["weight"])


def test_fedopt_variants_move_in_client_direction():
    for name in ("fedadagrad", "fedadam", "fedyogi"):
        server = create_server_optimizer(
            name, {"server_lr": 0.1, "beta1": 0.0, "beta2": 0.9, "tau": 1e-3}
        )
        output = server.step(state(0), [result(1)])
        assert output["weight"].item() > 0


def test_server_adaptivity_does_not_apply_to_model_buffers():
    global_state = {"weight": torch.tensor([0.0]), "running_mean": torch.tensor([10.0])}
    client = ClientResult(
        0,
        {"weight": torch.tensor([1.0]), "running_mean": torch.tensor([2.0])},
        1,
        1,
        0.0,
    )
    server = create_server_optimizer(
        "fedadam", {"server_lr": 0.1, "beta1": 0.0, "beta2": 0.9, "tau": 1e-3}
    )
    server.parameter_names = {"weight"}
    output = server.step(global_state, [client])
    assert output["weight"].item() > 0
    assert torch.equal(output["running_mean"], torch.tensor([2.0]))


def test_robust_aggregators_reject_outlier():
    results = [result(1), result(1.1), result(0.9), result(1.05), result(100)]
    median = create_server_optimizer("fedmedian").step(state(0), results)
    trimmed = create_server_optimizer("fedtrimmedmean", {"trim_ratio": 0.2}).step(
        state(0), results
    )
    krum = create_server_optimizer("krum", {"byzantine_clients": 1}).step(state(0), results)
    assert torch.allclose(median["weight"], torch.tensor([1.05]))
    assert trimmed["weight"].item() < 2
    assert krum["weight"].item() < 2
