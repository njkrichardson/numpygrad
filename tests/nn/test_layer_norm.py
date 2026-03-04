import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import numpygrad as npg
from numpygrad.nn.layer_norm import LayerNorm
from tests.configuration import FLOAT_DISTRIBUTION, check_equality


@st.composite
def configuration(draw):
    n_batch_dims = draw(st.integers(min_value=1, max_value=2))
    batch_shape = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(n_batch_dims))
    n_norm_dims = draw(st.integers(min_value=1, max_value=2))
    norm_shape = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(n_norm_dims))
    return batch_shape + norm_shape, norm_shape


@settings(deadline=None)
@given(configuration())
def test_forward(shapes):
    input_shape, normalized_shape = shapes
    x_np = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    w_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)
    b_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)

    ln = LayerNorm(normalized_shape)
    ln.weight.data = w_np
    ln.bias.data = b_np
    y = ln(npg.array(x_np))

    torch_ln = torch.nn.LayerNorm(normalized_shape, dtype=torch.float64)
    torch_ln.weight.data = torch.from_numpy(w_np)
    torch_ln.bias.data = torch.from_numpy(b_np)
    yt = torch_ln(torch.from_numpy(x_np))

    check_equality(y.data, yt.detach().numpy(), rtol=1e-10, atol=1e-10)


@settings(deadline=None)
@given(configuration())
def test_backward_x(shapes):
    input_shape, normalized_shape = shapes
    x_np = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    w_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)
    b_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)

    ln = LayerNorm(normalized_shape)
    ln.weight.data = w_np
    ln.bias.data = b_np
    x = npg.array(x_np, requires_grad=True)
    y = ln(x)
    y.backward()

    xt = torch.from_numpy(x_np).requires_grad_(True)
    torch_ln = torch.nn.LayerNorm(normalized_shape, dtype=torch.float64)
    torch_ln.weight.data = torch.from_numpy(w_np)
    torch_ln.bias.data = torch.from_numpy(b_np)
    yt = torch_ln(xt)
    yt.sum().backward()

    check_equality(x.grad, xt.grad.numpy(), rtol=1e-8, atol=1e-8)


@settings(deadline=None)
@given(configuration())
def test_backward_weight_bias(shapes):
    input_shape, normalized_shape = shapes
    x_np = FLOAT_DISTRIBUTION(input_shape).astype(np.float64)
    w_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)
    b_np = FLOAT_DISTRIBUTION(normalized_shape).astype(np.float64)

    ln = LayerNorm(normalized_shape)
    ln.weight.data = w_np
    ln.bias.data = b_np
    x = npg.array(x_np)
    y = ln(x)
    y.backward()

    torch_ln = torch.nn.LayerNorm(normalized_shape, dtype=torch.float64)
    torch_ln.weight.data = torch.from_numpy(w_np).requires_grad_(True)
    torch_ln.bias.data = torch.from_numpy(b_np).requires_grad_(True)
    yt = torch_ln(torch.from_numpy(x_np))
    yt.sum().backward()

    check_equality(ln.weight.grad, torch_ln.weight.grad.numpy(), rtol=1e-8, atol=1e-8)
    check_equality(ln.bias.grad, torch_ln.bias.grad.numpy(), rtol=1e-8, atol=1e-8)


def test_no_affine_forward():
    x_np = FLOAT_DISTRIBUTION((2, 4)).astype(np.float64)
    ln = LayerNorm(4, elementwise_affine=False)
    y = ln(npg.array(x_np))
    torch_ln = torch.nn.LayerNorm(4, elementwise_affine=False, dtype=torch.float64)
    yt = torch_ln(torch.from_numpy(x_np))
    check_equality(y.data, yt.detach().numpy(), rtol=1e-10, atol=1e-10)


def test_no_affine_no_parameters():
    ln = LayerNorm(4, elementwise_affine=False)
    assert ln.parameters() == []


def test_repr():
    assert repr(LayerNorm(4)) == "LayerNorm((4,), eps=1e-05, elementwise_affine=True)"
    assert repr(LayerNorm((3, 4), elementwise_affine=False)) == (
        "LayerNorm((3, 4), eps=1e-05, elementwise_affine=False)"
    )
