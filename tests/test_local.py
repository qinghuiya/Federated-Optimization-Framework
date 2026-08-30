import pytest
import torch

from federated_optimization.local import available_optimizers, create_optimizer, create_scheduler


@pytest.mark.parametrize("name", available_optimizers())
def test_every_documented_local_optimizer_takes_a_step(name):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = create_optimizer([parameter], {"name": name, "lr": 0.01})
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("step", {"step_size": 1, "gamma": 0.5}),
        ("multistep", {"milestones": [1], "gamma": 0.5}),
        ("exponential", {"gamma": 0.9}),
        ("cosine", {"T_max": 2}),
        ("linear", {"total_iters": 2}),
    ],
)
def test_local_schedulers(name, options):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = create_scheduler(optimizer, {"name": name, **options})
    assert scheduler is not None
    optimizer.step()
    scheduler.step()


def test_unknown_optimizer_has_helpful_error():
    with pytest.raises(ValueError, match="Available"):
        create_optimizer([], {"name": "does-not-exist"})

