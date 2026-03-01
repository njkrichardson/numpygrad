import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import numpygrad as npg
from numpygrad.nn.linear import Linear
from tests.configuration import (
    FLOAT_DISTRIBUTION,
    check_equality,
)

MIN_DIM_SIZE: int = 1
MAX_DIM_SIZE: int = 16
MAX_NUM_BATCH_DIMS: int = 3
MAX_BATCH_DIM_SIZE: int = 16

DIM_SIZE = st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)


@st.composite
def configuration(draw):
    num_inputs = draw(DIM_SIZE)
    num_outputs = draw(DIM_SIZE)
    num_batch_dims = draw(st.integers(min_value=0, max_value=MAX_NUM_BATCH_DIMS))
    batch_dim_sizes = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_BATCH_DIM_SIZE))
        for _ in range(num_batch_dims)
    )
    input_shape = tuple(batch_dim_sizes) + (num_inputs,)
    output_shape = tuple(batch_dim_sizes) + (num_outputs,)
    return input_shape, output_shape


@given(configuration())
def test_linear_forward(config):
    input_shape, output_shape = config
    num_inputs, num_outputs = input_shape[-1], output_shape[-1]
    weight = FLOAT_DISTRIBUTION((num_outputs, num_inputs)).astype(np.float64)

    torch_linear = torch.nn.Linear(num_inputs, num_outputs, bias=False)
    torch_linear.weight.data = torch.from_numpy(weight)
    linear = Linear(num_inputs, num_outputs)
    linear.weight.data = weight
    linear.bias.data = np.zeros(num_outputs)

    x = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    y = linear(npg.array(x))
    yt = torch_linear(torch.from_numpy(x))

    check_equality(y.data, yt.detach().numpy())

    bias = FLOAT_DISTRIBUTION((num_outputs,)).astype(np.float64)

    torch_linear = torch.nn.Linear(num_inputs, num_outputs, bias=True)
    torch_linear.weight.data = torch.from_numpy(weight)
    torch_linear.bias.data = torch.from_numpy(bias)
    linear2 = Linear(num_inputs, num_outputs)
    linear2.weight.data = weight
    linear2.bias.data = bias

    x = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    y = linear2(npg.array(x))
    yt = torch_linear(torch.from_numpy(x))

    check_equality(y.data, yt.detach().numpy())


@settings(deadline=None)
@given(configuration())
def test_linear_backward(config):
    input_shape, output_shape = config
    num_inputs, num_outputs = input_shape[-1], output_shape[-1]
    weight = FLOAT_DISTRIBUTION((num_outputs, num_inputs)).astype(np.float64)

    torch_linear = torch.nn.Linear(num_inputs, num_outputs, bias=False)
    torch_linear.weight.data = torch.from_numpy(weight)
    linear = Linear(num_inputs, num_outputs)
    linear.weight.data = weight
    linear.bias.data = np.zeros(num_outputs)

    x = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    y = linear(npg.array(x))
    y.backward()

    yt = torch_linear(torch.from_numpy(x))
    gwt = torch.autograd.grad(
        outputs=yt,
        inputs=(torch_linear.weight,),
        grad_outputs=torch.ones_like(yt),
    )[0]
    assert linear.weight.grad is not None
    check_equality(linear.weight.grad, gwt.detach().numpy())
