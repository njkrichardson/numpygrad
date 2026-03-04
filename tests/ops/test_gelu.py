import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import given

import numpygrad as npg
from tests.configuration import check_equality
from tests.strategies import FLOAT_DTYPES, generic_array


def test_gelu_forward_basic():
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = npg.array(arr)
    out = npg.gelu(x)

    xt = torch.tensor(arr)
    ref = F.gelu(xt, approximate="tanh").numpy()
    check_equality(out.data, ref, atol=1e-6)


def test_gelu_no_grad():
    arr = np.array([1.0, 2.0, 3.0])
    x = npg.array(arr, requires_grad=False)
    out = npg.gelu(x)
    assert not out.requires_grad


def test_gelu_backward_basic():
    arr = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    x = npg.array(arr, requires_grad=True)
    out = npg.gelu(x)
    out.backward()

    xt = torch.tensor(arr, requires_grad=True)
    F.gelu(xt, approximate="tanh").sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy(), atol=1e-6)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_gelu_backward_hypothesis(arr):
    arr = arr.astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    out = npg.gelu(x)
    out.backward()

    xt = torch.tensor(arr, requires_grad=True)
    F.gelu(xt, approximate="tanh").sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy(), atol=1e-6)


def test_nn_gelu_module():
    arr = np.array([-1.0, 0.0, 1.0, 2.0])
    x = npg.array(arr)
    out = npg.nn.GELU()(x)

    xt = torch.tensor(arr)
    ref = F.gelu(xt, approximate="tanh").numpy()
    check_equality(out.data, ref, atol=1e-6)
