# NumpyGrad

A small **autograd** and neural network library with a **PyTorch-like API**, built on **NumPy** only. No GPU, no C++ extensions—just Python and NumPy. Useful for learning how backprop and frameworks like PyTorch work, or for lightweight experiments where a full framework is overkill.

## Features

- **NumPy-only** — Single dependency: NumPy. Runs anywhere Python runs.
- **Define-by-run autograd** — Build a computation graph as you run ops; backward pass computes gradients via the chain rule.
- **Familiar array API** — `npg.array` with `shape`, `ndim`, `dtype` 
- **Familiar array creation** - `npg.ones`, `npg.arange`, `npg.randn`, etc.
- **Familiar NN API** - `.backward()`, `requires_grad`, `.grad`.
- **Basic NN Modules and Optimizers** - `Linear`, `MLP`, `SGD`, etc.
- **Broadcasting & batched ops** — Matmul, reductions, transforms, and elementwise ops support batched and broadcasted shapes.
- **Familiar special methods** - `x[:, 2, ::2]`, `x @ y`, `x[x > 0] = 1.`, etc.

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
import numpy as np
import numpygrad as npg
from numpygrad.nn import MLP

# Arrays and gradients
x = npg.array([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # gradients of sum(x²) w.r.t. x

# Small MLP
net = MLP(input_dim=1, hidden_sizes=[8, 8], output_dim=1)
optimizer = npg.optim.SGD(net.parameters(), step_size=1e-1)

x = npg.randn(32, 1)
targets = npg.randn(32, 1)
out = net(x)
loss = ((out - targets) ** 2).mean()
loss.backward()
optimizer.step()
```

## Example: 1D regression

The `examples/` directory includes a regression demo that fits an MLP to a noisy sine wave (with an optional PyTorch baseline for comparison):

```bash
python main.py
# or
python examples/regression_1d.py
```

This trains a small MLP and saves a plot of the fit under `media/`.

## Project layout

```
src/numpygrad/
├── core/          # Array, autograd (Function, backward), dispatch, device
├── ops/            # Operators: elementwise, linalg, reductions, etc.
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
