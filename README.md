![CI](https://github.com/njkrichardson/numpygrad/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-cc6600)
![Coverage](https://codecov.io/gh/njkrichardson/numpygrad/branch/main/graph/badge.svg)
![mypy](https://github.com/njkrichardson/numpygrad/actions/workflows/mypy.yml/badge.svg)

# NumpyGrad

A small **autograd** and neural network library with a **PyTorch-like API**, built on **NumPy** only. No GPU, no C++ extensions—just Python and NumPy. Useful for learning how backprop and frameworks like PyTorch work, or for lightweight experiments where a full framework is overkill.

## Features

### Autograd
- **NumPy-only** — Single dependency: NumPy. 
- **Define-by-run autograd** — Builds a computation graph as ops are invoked (i.e., Torch eager).
- **Familiar array API** — `array` with `shape`, `ndim`, `dtype` etc.
- **Familiar array creation** - `ones`, `zeros`, `arange`, `randn`, etc.
- **Familiar torch autograd API** - `.backward()`, `requires_grad`, `.grad`, `no_grad()`, etc.
- **Broadcasting & batched ops** — Linear algebra, reductions, transforms, and elementwise ops support batched and broadcasted shapes.
- **Familiar special methods** - `x @ y`, `mask = x > 0`, etc.

### Neural Nets
- **Modules** - `Linear`, `Conv2d`, `MultiHeadAttention`, etc.
- **Optimizers** - `AdamW`, `SGD`.
- **Activations** - `ReLU`, `Tanh`, `Sigmoid`. 
- **Losses** - `CrossEntropy`, `mse`. 

## Installation

From the project root:

```bash
pip install -e .
```

Requires Python ≥3.12 and NumPy ≥2.4.2.

Optional dependencies (e.g. for tests and examples):

```bash
pip install -e ".[tests]"      # pytest, hypothesis, torch (for gradient checks)
pip install -e ".[examples]"  # matplotlib for plotting
```

## Quick start

```python
import numpygrad as np # live on the edge! 
import numpygrad.nn as nn

# Arrays and gradients
x = np.randn((3, 4), requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # gradients of sum(x²) w.r.t. x

# Small MLP
net = nn.MLP(input_dim=1, hidden_sizes=[8, 8], output_dim=1)
optimizer = np.optim.SGD(net.parameters(), step_size=1e-1)

x = np.randn(32, 1)
targets = np.randn(32, 1)
out = net(x)
loss = ((out - targets) ** 2).mean()
loss.backward()
optimizer.step()
```

## Examples

The `examples/` directory includes demos that run with only the core install (NumPy +
numpygrad). You can run them without installing the `[examples]` extra:

```bash
python -m examples.regression_1d.main   # use --help for CLI options
python -m examples.classification_2d.main
```

If **matplotlib** is not installed, training and evaluation run as usual but no figures
are generated. Install the examples extra to enable plotting:

```bash
pip install -e ".[examples]"   # adds matplotlib
```

### Scalar Regression

```bash
python -m examples.regression_1d.main # use --help for cli arg descriptions
```

### 2d Classification 

```bash
python -m examples.classification_2d.main # use --help for cli arg descriptions
```

## Project layout

```
src/numpygrad/
├── core/           # Array, autograd (Function, backward), dispatch, device
├── ops/            # Operators: elementwise, linalg, transforms, reductions, etc.
├── nn/             # Linear, MLP (and other modules)
├── optim/          # SGD and optimizer base
├── utils/          # Logging, I/O, visualizations
└── configuration.py
```

Tests live in `tests/` and use Hypothesis plus PyTorch to check gradients against a reference.

## Development

Run tests:

```bash
pytest
# or with optional deps
pip install -e ".[tests]" && pytest
```

## License

MIT License. See [LICENSE](LICENSE) for details.
