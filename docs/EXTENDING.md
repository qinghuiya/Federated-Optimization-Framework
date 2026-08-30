# Extending the framework

## Add a local optimizer

Add a stable lowercase name and optimizer class to `OPTIMIZERS` in
`federated_optimization/local.py`. If the optimizer needs a closure, sparse parameters,
or unusual state lifetime, add an explicit client-training path instead of pretending it
fits the generic mini-batch step. Add a one-step test in `tests/test_local.py`.

## Add a server optimizer

1. Subclass `ServerOptimizer` in `federated_optimization/server.py`.
2. Implement `step(global_state, results, total_clients=...)`.
3. Keep persistent server state on the instance.
4. Register the public name in `create_server_optimizer`.
5. Add a small formula-level test and an end-to-end smoke case.
6. Document the source paper, assumptions, and configuration fields.

`ClientResult` contains client id, endpoint state, local dataset size, local step count,
mean loss, and an optional control-variate delta. Add a clearly named optional field if a
new algorithm needs more metadata.

## Add a client strategy

Client-specific loss or gradient logic belongs in `ClientTrainer.train`. Keep the plain
path readable and gate special behavior behind explicit arguments. If an algorithm needs
persistent personalized state, create a dedicated strategy class instead of adding an
ambiguous flag.

## Correctness checklist

- Compare a one-dimensional update against a hand calculation.
- Check unequal client sample counts.
- Check partial participation and persistent state across two rounds.
- Test invalid theoretical configurations and fail with a helpful error.
- Run the synthetic end-to-end smoke test.
- Cite the original paper; distinguish an exact implementation from a restricted case.

