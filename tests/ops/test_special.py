import numpy as np
import pytest
from hypothesis import given, strategies as st

import numpygrad as npg
import numpygrad.ops as ops
from tests.strategies import array_pair, generic_array


# --- Comparison ops (GT, LT, GE, LE, EQ, NE): forward-only, no backward ---


@given(array_pair(same_shape=True))
def test_gt_basic(data):
    a, b = data
    z = npg.array(a) > npg.array(b)
    reference = a > b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(same_shape=True))
def test_lt_basic(data):
    a, b = data
    z = npg.array(a) < npg.array(b)
    reference = a < b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(same_shape=True))
def test_ge_basic(data):
    a, b = data
    z = npg.array(a) >= npg.array(b)
    reference = a >= b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(same_shape=True))
def test_le_basic(data):
    a, b = data
    z = npg.array(a) <= npg.array(b)
    reference = a <= b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(same_shape=True))
def test_eq_basic(data):
    a, b = data
    z = npg.array(a) == npg.array(b)
    reference = a == b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(same_shape=True))
def test_ne_basic(data):
    a, b = data
    z = npg.array(a) != npg.array(b)
    reference = a != b
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(broadcastable=True))
def test_gt_broadcast(data):
    a, b = data
    z = npg.array(a) > npg.array(b)
    reference = np.broadcast_to(a > b, np.broadcast_shapes(a.shape, b.shape))
    np.testing.assert_array_equal(z.data, reference)


@given(array_pair(broadcastable=True))
def test_lt_broadcast(data):
    a, b = data
    z = npg.array(a) < npg.array(b)
    reference = np.broadcast_to(a < b, np.broadcast_shapes(a.shape, b.shape))
    np.testing.assert_array_equal(z.data, reference)


# --- setitem (existing + optional forward property test) ---


def test_setitem_functional_backward_basic():
    x = npg.ones((3, 3), requires_grad=True)
    b = npg.ones((3,), requires_grad=True)

    y = ops.setitem(x, (slice(None), 0), b)
    out = y.sum()
    out.backward()

    assert x.grad is not None
    assert b.grad is not None

    expected_x_grad = np.ones_like(x.data)
    expected_x_grad[:, 0] = 0

    np.testing.assert_allclose(x.grad, expected_x_grad)
    np.testing.assert_allclose(b.grad, np.ones_like(b.data))


def test_array_setitem_requires_grad_raises():
    x = npg.ones((3, 3), requires_grad=True)
    b = npg.ones((3,), requires_grad=True)

    with pytest.raises(RuntimeError, match="__setitem__ on Arrays that require grad is not supported"):
        x[:, 0] = b


@st.composite
def setitem_args(draw):
    """Strategy for setitem forward: (arr, key, value) where key is (slice(None), col)."""
    n = draw(st.integers(1, 8))
    m = draw(st.integers(1, 8))
    shape = (n, m)
    arr = draw(generic_array(shape=shape))
    col = draw(st.integers(0, m - 1))
    key = (slice(None), col)
    value_shape = (n,)
    value = draw(generic_array(shape=value_shape))
    return arr, key, value


@given(setitem_args())
def test_setitem_forward(data):
    arr, key, value = data
    out = ops.setitem(npg.array(arr), key, npg.array(value))
    expected = arr.copy()
    expected[key] = value
    np.testing.assert_array_equal(out.data, expected)

