import numpy as np
import pytest

import numpygrad as npg
from numpygrad.nn.dropout import Dropout


def test_eval_passthrough():
    d = Dropout(p=0.5)
    d.eval()
    x = npg.array(np.ones((4, 4)))
    np.testing.assert_array_equal(d(x).data, x.data)


def test_zero_p_passthrough():
    d = Dropout(p=0.0)
    x = npg.array(np.ones((4, 4)))
    np.testing.assert_array_equal(d(x).data, x.data)


def test_sparsity():
    np.random.seed(42)
    d = Dropout(p=0.5)
    y = d(npg.array(np.ones((1000,))))
    zero_frac = np.mean(y.data == 0)
    assert 0.4 < zero_frac < 0.6


def test_scale():
    # Non-dropped elements must be scaled by 1/(1-p)
    np.random.seed(0)
    d = Dropout(p=0.5)
    y = d(npg.array(np.ones((100,))))
    nonzero = y.data[y.data != 0]
    np.testing.assert_allclose(nonzero, 2.0)


def test_backward():
    # grad is zero where output was zeroed, 1/(1-p) where kept
    np.random.seed(1)
    d = Dropout(p=0.5)
    x = npg.array(np.ones((10,)), requires_grad=True)
    y = d(x)
    y.backward()
    # y = x * mask/(1-p); backward seeds all-ones upstream → x.grad = mask/(1-p) = y.data
    np.testing.assert_allclose(x.grad, y.data)


def test_invalid_p():
    with pytest.raises(ValueError):
        Dropout(p=1.0)
    with pytest.raises(ValueError):
        Dropout(p=-0.1)


def test_repr():
    assert repr(Dropout(p=0.3)) == "Dropout(p=0.3)"
