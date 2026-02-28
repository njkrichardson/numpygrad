"""Tests for ops/activations.py — Softmax and LogSoftmax."""

import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import given

import numpygrad as npg
from tests.configuration import check_equality
from tests.strategies import FLOAT_DTYPES, generic_array

npg.manual_seed(0)


# --- Softmax ---


def test_softmax_forward_basic():
    data = np.array([1.0, 2.0, 3.0])
    x = npg.array(data)
    out = npg.softmax(x)

    xt = torch.from_numpy(data)
    ref = F.softmax(xt, dim=-1).numpy()
    check_equality(out.data, ref)


def test_softmax_no_grad():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = npg.array(data, requires_grad=False)
    out = npg.softmax(x)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_softmax_backward_basic():
    data = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    x = npg.array(data, requires_grad=True)
    out = npg.softmax(x)
    out.sum().backward()

    xt = torch.from_numpy(data).requires_grad_(True)
    F.softmax(xt, dim=-1).sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


def test_softmax_axis0():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = npg.array(data, requires_grad=True)
    out = npg.softmax(x, axis=0)
    out.sum().backward()

    xt = torch.from_numpy(data).requires_grad_(True)
    F.softmax(xt, dim=0).sum().backward()

    assert x.grad is not None
    check_equality(out.data, F.softmax(torch.from_numpy(data), dim=0).numpy())
    check_equality(x.grad, xt.grad.numpy())


@given(generic_array(shape=(3, 4), dtypes=FLOAT_DTYPES))
def test_softmax_backward_hypothesis(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.softmax(x).sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    F.softmax(xt, dim=-1).sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy(), rtol=1e-10)


# --- LogSoftmax ---


def test_log_softmax_forward_basic():
    data = np.array([1.0, 2.0, 3.0])
    x = npg.array(data)
    out = npg.log_softmax(x)

    xt = torch.from_numpy(data)
    ref = F.log_softmax(xt, dim=-1).numpy()
    check_equality(out.data, ref)


def test_log_softmax_no_grad():
    data = np.array([[0.1, 0.9], [0.4, 0.6]])
    x = npg.array(data, requires_grad=False)
    out = npg.log_softmax(x)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_log_softmax_backward_basic():
    data = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    x = npg.array(data, requires_grad=True)
    npg.log_softmax(x).sum().backward()

    xt = torch.from_numpy(data).requires_grad_(True)
    F.log_softmax(xt, dim=-1).sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


@given(generic_array(shape=(3, 4), dtypes=FLOAT_DTYPES))
def test_log_softmax_backward_hypothesis(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.log_softmax(x).sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    F.log_softmax(xt, dim=-1).sum().backward()

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy(), rtol=1e-10)
