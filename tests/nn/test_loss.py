"""Tests for nn/loss.py."""

import numpy as np
import pytest
import torch

import numpygrad as npg
from numpygrad.nn.loss import mse
from tests.configuration import check_equality


def test_mse_mean_reduction():
    preds = npg.array(np.array([1.0, 2.0, 3.0]))
    targets = npg.array(np.array([1.5, 2.5, 3.5]))
    loss = mse(preds, targets)
    expected = np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) ** 2)
    np.testing.assert_allclose(loss.data, expected)


def test_mse_sum_reduction():
    preds = npg.array(np.array([1.0, 2.0, 3.0]))
    targets = npg.array(np.array([1.5, 2.5, 3.5]))
    loss = mse(preds, targets, reduction="sum")
    expected = np.sum((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) ** 2)
    np.testing.assert_allclose(loss.data, expected)


def test_mse_invalid_reduction():
    preds = npg.array(np.array([1.0, 2.0]))
    targets = npg.array(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Invalid reduction"):
        mse(preds, targets, reduction="none")


def test_mse_with_weight():
    preds = npg.array(np.array([1.0, 2.0, 3.0]))
    targets = npg.array(np.array([0.0, 0.0, 0.0]))
    weight = npg.array(np.array([2.0, 1.0, 0.5]))
    loss = mse(preds, targets, weight=weight)
    # weighted: ((1*2 - 0*2)^2 + (2*1 - 0*1)^2 + (3*0.5 - 0*0.5)^2) / 3
    expected = np.mean((np.array([1.0, 2.0, 3.0]) * np.array([2.0, 1.0, 0.5])) ** 2)
    np.testing.assert_allclose(loss.data, expected)


def test_mse_gradient_vs_torch():
    p = np.array([1.0, 2.0, 3.0])
    t = np.array([0.5, 1.5, 2.5])

    preds = npg.array(p, requires_grad=True)
    targets = npg.array(t)
    loss = mse(preds, targets)
    loss.backward()

    pt = torch.from_numpy(p).requires_grad_(True)
    tt = torch.from_numpy(t)
    lt = torch.nn.functional.mse_loss(pt, tt)
    lt.backward()

    assert preds.grad is not None
    check_equality(preds.grad, pt.grad.numpy())
