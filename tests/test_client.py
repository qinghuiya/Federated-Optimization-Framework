import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_optimization.client import ClientTrainer, empty_control


def make_problem():
    torch.manual_seed(4)
    inputs = torch.randn(24, 3)
    targets = (inputs[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=6, shuffle=False)
    model = nn.Linear(3, 2)
    return model, loader


def distance(left, right):
    return sum(
        torch.sum((left[name] - right[name]) ** 2).item()
        for name in left
        if torch.is_floating_point(left[name])
    )


def test_fedprox_keeps_local_model_closer_to_global_model():
    model, loader = make_problem()
    initial = copy.deepcopy(model.state_dict())
    trainer = ClientTrainer(model)
    plain = trainer.train(
        client_id=0,
        loader=loader,
        global_state=initial,
        optimizer_config={"name": "sgd", "lr": 0.1},
        local_epochs=3,
    )
    proximal = trainer.train(
        client_id=0,
        loader=loader,
        global_state=initial,
        optimizer_config={"name": "sgd", "lr": 0.1},
        local_epochs=3,
        proximal_mu=2.0,
    )
    assert distance(proximal.state, initial) < distance(plain.state, initial)


def test_scaffold_returns_parameter_control_delta():
    model, loader = make_problem()
    controls = empty_control(model)
    result = ClientTrainer(model).train(
        client_id=0,
        loader=loader,
        global_state=model.state_dict(),
        optimizer_config={"name": "sgd", "lr": 0.1},
        local_steps=2,
        local_epochs=None,
        global_control=controls,
        client_control=controls,
    )
    assert result.control_delta is not None
    assert set(result.control_delta) == {name for name, _ in model.named_parameters()}
    assert any(torch.count_nonzero(value) for value in result.control_delta.values())

