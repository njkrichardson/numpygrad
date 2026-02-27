from hypothesis import given, strategies as st
import numpy as np
import torch

import numpygrad as npg
from numpygrad.nn.mlp import MLP

def _linear_modules(sequential):
    """Collect Linear submodules from a Sequential in order. Linears are at indices 0, 2, 4, ... (ReLU at 1, 3, ...)."""
    modules = list(sequential._modules.values())
    # Sequential is [Linear, ReLU, Linear, ReLU, ..., Linear]
    return [modules[i] for i in range(0, len(modules), 2)]

from tests.configuration import (
    check_equality,
    FLOAT_DISTRIBUTION,
)

MIN_DIM_SIZE: int = 1
MAX_DIM_SIZE: int = 16
MAX_NUM_BATCH_DIMS: int = 3
MAX_BATCH_DIM_SIZE: int = 16
MIN_NUM_HIDDEN_LAYERS: int = 1
MAX_NUM_HIDDEN_LAYERS: int = 8

DIM_SIZE = st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)

@st.composite
def configuration(draw):
    num_inputs = draw(DIM_SIZE)
    num_outputs = draw(DIM_SIZE)
    hidden_sizes = draw(st.lists(DIM_SIZE, min_size=MIN_NUM_HIDDEN_LAYERS, max_size=MAX_NUM_HIDDEN_LAYERS))
    num_batch_dims = draw(st.integers(min_value=0, max_value=MAX_NUM_BATCH_DIMS))
    batch_dim_sizes = tuple(draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_BATCH_DIM_SIZE)) for _ in range(num_batch_dims))
    input_shape = tuple(batch_dim_sizes) + (num_inputs,)
    output_shape = tuple(batch_dim_sizes) + (num_outputs,)
    return input_shape, output_shape, hidden_sizes

@given(configuration())
def test_mlp_forward(config):
    input_shape, output_shape, hidden_sizes = config
    num_inputs, num_outputs = input_shape[-1], output_shape[-1]
    hidden_sizes = [int(size) for size in hidden_sizes]
    mlp = MLP(num_inputs, hidden_sizes, num_outputs)

    layers = []
    sizes = [num_inputs] + hidden_sizes + [num_outputs]
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        weight = FLOAT_DISTRIBUTION((out_dim, in_dim)).astype(np.float64)
        bias = FLOAT_DISTRIBUTION((out_dim,)).astype(np.float64)
        layers.append((weight, bias))

    linears = _linear_modules(mlp.layers)
    for i, (weight, bias) in enumerate(layers):
        linears[i].weight.data = weight
        linears[i].bias.data = bias

    class TorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(sizes[:-1], sizes[1:])])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers[:-1]:
                x = torch.nn.functional.relu(layer(x))
            return self.layers[-1](x)

    torch_mlp = TorchMLP()
    for i, (weight, bias) in enumerate(layers):
        torch_mlp.layers[i].weight.data = torch.from_numpy(weight)
        torch_mlp.layers[i].bias.data = torch.from_numpy(bias)

    x = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    y = mlp(npg.array(x))
    yt = torch_mlp(torch.from_numpy(x).to(torch.float64))

    # NumPy vs PyTorch matmul can differ in the last ~10 bits due to reduction order
    check_equality(y.data, yt.detach().numpy(), rtol=1e-8, atol=1e-10)

@given(configuration())
def test_mlp_backward(config):
    input_shape, output_shape, hidden_sizes = config
    num_inputs, num_outputs = input_shape[-1], output_shape[-1]
    hidden_sizes = [int(size) for size in hidden_sizes]
    mlp = MLP(num_inputs, hidden_sizes, num_outputs)

    layers = []
    sizes = [num_inputs] + hidden_sizes + [num_outputs]
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        weight = FLOAT_DISTRIBUTION((out_dim, in_dim)).astype(np.float64)
        bias = FLOAT_DISTRIBUTION((out_dim,)).astype(np.float64)
        layers.append((weight, bias))

    linears = _linear_modules(mlp.layers)
    for i, (weight, bias) in enumerate(layers):
        linears[i].weight.data = weight
        linears[i].bias.data = bias

    class TorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(sizes[:-1], sizes[1:])])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers[:-1]:
                x = torch.nn.functional.relu(layer(x))
            return self.layers[-1](x)

    torch_mlp = TorchMLP()
    for i, (weight, bias) in enumerate(layers):
        torch_mlp.layers[i].weight.data = torch.from_numpy(weight)
        torch_mlp.layers[i].bias.data = torch.from_numpy(bias)

    x = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    y = mlp(npg.array(x))
    y.backward()

    x_t = torch.from_numpy(x).to(torch.float64)
    yt = torch_mlp(x_t)
    torch_params = [
        p for layer in torch_mlp.layers for p in (layer.weight, layer.bias)
    ]
    grad_outs = torch.autograd.grad(
        outputs=yt,
        inputs=tuple(torch_params), # type: ignore
        grad_outputs=torch.ones_like(yt),
    )
    grad_iter = iter(grad_outs)
    for ngp_layer, torch_layer in zip(linears, torch_mlp.layers):
        gwt = next(grad_iter)
        gbt = next(grad_iter)
        assert ngp_layer.weight.grad is not None
        check_equality(ngp_layer.weight.grad, gwt.detach().numpy(), rtol=1e-8, atol=1e-10)
        assert ngp_layer.bias.grad is not None
        check_equality(ngp_layer.bias.grad, gbt.detach().numpy(), rtol=1e-8, atol=1e-10)

    