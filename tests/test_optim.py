"""Tests for optim/optimizer.py and optim/sgd.py."""

import numpy as np
import pytest
import torch

from numpygrad.nn.module import Parameter
from numpygrad.optim.adamw import AdamW
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


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------


def _torch_adamw_reference(data, grads, lr, betas, eps, wd):
    p = torch.tensor(data.copy(), requires_grad=True)
    opt = torch.optim.AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for g in grads:
        opt.zero_grad()
        p.grad = torch.tensor(g)
        opt.step()
    return p.detach().numpy()


def test_adamw_defaults():
    params = _make_params()
    opt = AdamW(params)
    assert opt.lr == 1e-3
    assert opt.betas == (0.9, 0.999)
    assert opt.eps == 1e-8
    assert opt.weight_decay == 1e-2


def test_adamw_single_step():
    data = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])
    lr, betas, eps, wd = 1e-3, (0.9, 0.999), 1e-8, 1e-2

    param = Parameter(data.copy())
    param.grad = grad.copy()
    opt = AdamW([param], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    opt.step()

    expected = _torch_adamw_reference(data, [grad], lr, betas, eps, wd)
    np.testing.assert_allclose(param.data, expected, rtol=1e-6)


def test_adamw_multi_step():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(4)
    grads = [rng.standard_normal(4) for _ in range(10)]
    lr, betas, eps, wd = 1e-2, (0.9, 0.999), 1e-8, 1e-2

    param = Parameter(data.copy())
    opt = AdamW([param], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for g in grads:
        param.grad = g.copy()
        opt.step()

    expected = _torch_adamw_reference(data, grads, lr, betas, eps, wd)
    np.testing.assert_allclose(param.data, expected, rtol=1e-6)


def test_adamw_zero_weight_decay():
    rng = np.random.default_rng(7)
    data = rng.standard_normal(3)
    grads = [rng.standard_normal(3) for _ in range(5)]
    lr, betas, eps, wd = 1e-3, (0.9, 0.999), 1e-8, 0.0

    param = Parameter(data.copy())
    opt = AdamW([param], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for g in grads:
        param.grad = g.copy()
        opt.step()

    expected = _torch_adamw_reference(data, grads, lr, betas, eps, wd)
    np.testing.assert_allclose(param.data, expected, rtol=1e-6)
