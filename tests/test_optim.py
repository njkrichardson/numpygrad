"""Tests for optim/optimizer.py and optim/sgd.py."""

import numpy as np
import pytest

from numpygrad.nn.module import Parameter
from numpygrad.optim.optimizer import Optimizer
from numpygrad.optim.sgd import SGD


def _make_params():
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.array([4.0, 5.0]))
    # Simulate gradients being set by backward
    p1.grad = np.array([0.1, 0.2, 0.3])
    p2.grad = np.array([0.4, 0.5])
    return [p1, p2]


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------


def test_sgd_step_updates_params():
    params = _make_params()
    lr = 0.1
    sgd = SGD(params, step_size=lr)
    sgd.step()

    np.testing.assert_allclose(
        params[0].data, np.array([1.0, 2.0, 3.0]) - lr * np.array([0.1, 0.2, 0.3])
    )
    np.testing.assert_allclose(params[1].data, np.array([4.0, 5.0]) - lr * np.array([0.4, 0.5]))


def test_sgd_zero_grad():
    params = _make_params()
    sgd = SGD(params)
    sgd.zero_grad()

    for p in params:
        np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))


def test_sgd_default_step_size():
    params = _make_params()
    sgd = SGD(params)
    assert sgd.step_size == 1e-3


# ---------------------------------------------------------------------------
# Optimizer base
# ---------------------------------------------------------------------------


def test_optimizer_step_not_implemented():
    params = _make_params()
    opt = Optimizer(params)
    with pytest.raises(NotImplementedError):
        opt.step()


def test_optimizer_stores_params():
    params = _make_params()
    opt = Optimizer(params)
    assert opt.params is params


def test_optimizer_zero_grad():
    params = _make_params()
    opt = Optimizer(params)
    opt.zero_grad()
    for p in params:
        np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))
