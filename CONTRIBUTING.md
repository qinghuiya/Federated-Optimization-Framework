# Contributing

Thank you for helping make federated optimization easier to learn.

## Development setup

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
ruff check .
```

## Pull requests

- Keep the public training path small and readable.
- Include tests for behavior, not only import coverage.
- Add the original source and state implementation limitations for new algorithms.
- Do not add unpublished/private methods, private datasets, generated outputs, model
  checkpoints, credentials, or personally identifying experiment paths.
- Update both README files when a user-facing capability changes.

Small focused changes are easier to review. By contributing, you agree that your work is
licensed under this repository's MIT License.

