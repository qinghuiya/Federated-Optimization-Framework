# Getting started

## 1. Create an isolated environment

Python 3.10 or newer is required. A virtual environment prevents this project's PyTorch
version from changing another project.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on
Linux and macOS. Then install the package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

For tests and style checks, use `pip install -e ".[dev]"`.

## 2. Run the smallest example

```bash
fedopt-train --config configs/synthetic_fedavg.yaml --output outputs/tutorial
```

The synthetic example generates deterministic class clusters locally. It is useful for
checking an installation and learning the workflow; it is not a research benchmark.

## 3. Read the outputs

- `resolved_config.yaml` is the exact configuration used for the run.
- `metrics.jsonl` contains one JSON object per communication round.
- `summary.json` contains final accuracy, loss, runtime, and run metadata.
- `final_model.pt` is the final PyTorch state dictionary.

JSONL is convenient because a long run can be inspected before it finishes and loaded
with pandas using `pandas.read_json(path, lines=True)`.

## 4. Change one variable at a time

Use command-line overrides for a first comparison:

```bash
fedopt-train --config configs/synthetic_fedavg.yaml --algorithm fedprox --rounds 10 \
  --output outputs/tutorial-fedprox
```

For a real experiment, copy a YAML file and commit it with your results. Fix the seed,
data partition, selected-client rate, model, and evaluation schedule before comparing
optimizers.

## 5. Move to image data

`configs/mnist_scaffold.yaml` and `configs/cifar10_fedadam.yaml` download their datasets
on the first run. Start with MNIST on CPU. CIFAR-10 with ResNet-18 is much faster on a
CUDA GPU.

```bash
fedopt-train --config configs/mnist_scaffold.yaml --output outputs/mnist-scaffold
```

## Common problems

- **CUDA requested but unavailable:** use `--device cpu`, or install a CUDA-enabled
  PyTorch build that matches your driver.
- **An empty or tiny client:** increase `partition.min_samples` for a Dirichlet split.
- **Krum rejects the configuration:** a round needs at least `2f + 3` clients, where
  `f` is `server.byzantine_clients`.
- **SCAFFOLD or FedNova rejects the optimizer:** use plain SGD without momentum or a
  local scheduler; this framework does not silently use an approximate correction.

