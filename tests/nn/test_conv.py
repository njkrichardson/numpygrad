import numpy as np
import torch

import numpygrad as npg
import numpygrad.nn as nn
from tests.configuration import check_equality


def test_conv2d_parameter_shapes():
    layer = nn.Conv2d(3, 8, kernel_size=3)
    assert layer.weight.shape == (8, 3, 3, 3)
    assert layer.bias is not None
    assert layer.bias.shape == (8,)
    params = list(layer.parameters())
    assert len(params) == 2


def test_conv2d_no_bias():
    layer = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    assert layer.bias is None
    params = list(layer.parameters())
    assert len(params) == 1


def test_conv2d_forward_matches_torch():
    np.random.seed(42)
    layer = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)

    x = np.random.randn(1, 2, 5, 5).astype(np.float64)
    xn = npg.array(x)

    out = layer(xn)

    # copy weights into torch layer
    torch_layer = torch.nn.Conv2d(2, 4, 3, stride=1, padding=1, bias=True)
    torch_layer.weight = torch.nn.Parameter(torch.from_numpy(layer.weight.data))
    torch_layer.bias = torch.nn.Parameter(torch.from_numpy(layer.bias.data))

    ref = torch_layer(torch.from_numpy(x)).detach().numpy()
    check_equality(out.data, ref, rtol=1e-10)


def test_conv2d_backward():
    np.random.seed(7)
    layer = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=0)

    x = np.random.randn(2, 2, 5, 5).astype(np.float64)
    xn = npg.array(x, requires_grad=True)

    out = layer(xn)
    out.backward()

    assert xn.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None

    # reference with torch
    torch_layer = torch.nn.Conv2d(2, 3, 3, stride=1, padding=0, bias=True)
    torch_layer.weight = torch.nn.Parameter(torch.from_numpy(layer.weight.data))
    torch_layer.bias = torch.nn.Parameter(torch.from_numpy(layer.bias.data))

    xt = torch.from_numpy(x).requires_grad_(True)
    outt = torch_layer(xt)
    outt.backward(torch.ones_like(outt))

    check_equality(xn.grad, xt.grad.numpy(), rtol=1e-10)
    check_equality(layer.weight.grad, torch_layer.weight.grad.numpy(), rtol=1e-10)
    check_equality(layer.bias.grad, torch_layer.bias.grad.numpy(), rtol=1e-10)


def test_conv2d_repr():
    layer = nn.Conv2d(3, 8, kernel_size=3)
    assert (
        repr(layer) == "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), "
        "bias=True, dilation=1, groups=1 [not supported])"
    )
