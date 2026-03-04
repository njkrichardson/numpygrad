import numpy as np
import torch
from hypothesis import given

import numpygrad as npg
from tests.configuration import FLOAT_DTYPES, check_equality
from tests.strategies import generic_array


def test_masked_fill_basic():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = np.array([[True, False, True], [False, True, False]])
    x = npg.array(arr)
    out = x.masked_fill(npg.array(mask), -1e9)
    ref = torch.from_numpy(arr).masked_fill(torch.from_numpy(mask), -1e9).numpy()
    check_equality(out.data, ref)


def test_masked_fill_no_grad():
    x = npg.array(np.ones((3, 3)))
    m = npg.array(np.eye(3, dtype=bool))
    out = x.masked_fill(m, 0.0)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_masked_fill_backward_basic():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [False, True]])
    x = npg.array(arr, requires_grad=True)
    x.masked_fill(npg.array(mask), -1e9).sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    xt.masked_fill(torch.from_numpy(mask), -1e9).sum().backward()
    check_equality(x.grad, xt.grad.numpy())


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_masked_fill_backward_hypothesis(arr):
    if arr.ndim == 0:
        arr = arr.reshape(1)
    mask = np.zeros(arr.shape, dtype=bool)
    mask.flat[::2] = True

    x = npg.array(arr, requires_grad=True)
    x.masked_fill(npg.array(mask), 0.0).sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    xt.masked_fill(torch.from_numpy(mask), 0.0).sum().backward()
    check_equality(x.grad, xt.grad.numpy())


def test_npg_masked_fill_functional():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    mask = np.array([[True, False, True], [False, False, True]])
    out = npg.masked_fill(npg.array(arr), npg.array(mask), -999.0)
    ref = torch.from_numpy(arr).masked_fill(torch.from_numpy(mask), -999.0).numpy()
    check_equality(out.data, ref)
