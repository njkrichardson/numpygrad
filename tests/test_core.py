"""Tests for core array, array_creation, contexts, function, and transforms."""

import numpy as np
import pytest

import numpygrad as npg
from numpygrad.core.array import Array
from numpygrad.core.array_creation import arange, eye, linspace, randint, randn, zeros_like
from numpygrad.core.contexts import no_grad

# ---------------------------------------------------------------------------
# Array construction
# ---------------------------------------------------------------------------


def test_array_numpy():
    data = np.array([1.0, 2.0, 3.0])
    x = Array(data)
    result = x.numpy()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)


def test_array_from_list():
    x = Array([1.0, 2.0, 3.0])
    assert x.shape == (3,)
    np.testing.assert_array_equal(x.data, [1.0, 2.0, 3.0])


def test_array_from_tuple():
    x = Array((4, 5, 6))
    assert x.shape == (3,)


def test_array_with_dtype():
    x = Array(np.array([1, 2, 3]), dtype=np.float32)
    assert x.dtype == np.float32


def test_array_requires_grad_non_float_raises():
    with pytest.raises(ValueError, match="floating point"):
        Array(np.array([1, 2, 3], dtype=np.int32), requires_grad=True)


def test_array_from_array_unwraps():
    inner = Array(np.array([1.0, 2.0]))
    outer = Array(inner)
    assert isinstance(outer.data, np.ndarray)
    np.testing.assert_array_equal(outer.data, inner.data)


# ---------------------------------------------------------------------------
# Properties: nbytes, size, item
# ---------------------------------------------------------------------------


def test_nbytes():
    x = Array(np.ones((3, 4), dtype=np.float64))
    assert x.nbytes == 3 * 4 * 8


def test_size():
    x = Array(np.ones((2, 5)))
    assert x.size == 10


def test_item():
    x = Array(np.array(42.0))
    assert x.item() == 42.0


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_basic():
    x = Array(np.array([1.0]))
    r = repr(x)
    assert "Array" in r


def test_repr_with_label():
    x = Array(np.array([1.0]), label="my_var")
    r = repr(x)
    assert "my_var" in r


def test_repr_with_grad_fn():
    x = npg.array(np.array([1.0, 2.0]), requires_grad=True)
    y = x + x  # creates a node with grad_fn
    r = repr(y)
    assert "grad_fn" in r


# ---------------------------------------------------------------------------
# __setitem__ and setitem
# ---------------------------------------------------------------------------


def test_setitem_mutation():
    x = npg.array(np.array([1.0, 2.0, 3.0]))
    x[1] = 99.0
    assert x.data[1] == 99.0


def test_setitem_raises_for_requires_grad():
    x = npg.array(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(RuntimeError, match="require grad"):
        x[0] = 5.0


def test_functional_setitem():
    x = npg.array(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    result = x.setitem(1, npg.array(np.array(99.0)))
    assert result.data[1] == 99.0


# ---------------------------------------------------------------------------
# Arithmetic operators: __rmul__, __neg__, __sub__
# ---------------------------------------------------------------------------


def test_rmul():
    x = npg.array(np.array([2.0, 3.0]))
    y = 4.0 * x  # triggers __rmul__
    np.testing.assert_array_equal(y.data, [8.0, 12.0])


def test_neg():
    x = npg.array(np.array([1.0, -2.0]))
    y = -x
    np.testing.assert_array_equal(y.data, [-1.0, 2.0])


def test_sub():
    x = npg.array(np.array([5.0, 3.0]))
    y = npg.array(np.array([1.0, 2.0]))
    z = x - y
    np.testing.assert_array_equal(z.data, [4.0, 1.0])


# ---------------------------------------------------------------------------
# reshape and view
# ---------------------------------------------------------------------------


def test_reshape_tuple_arg():
    x = npg.array(np.arange(6.0))
    y = x.reshape((2, 3))
    assert y.shape == (2, 3)


def test_reshape_list_arg():
    x = npg.array(np.arange(6.0))
    y = x.reshape([3, 2])
    assert y.shape == (3, 2)


def test_view():
    x = npg.array(np.arange(6.0))
    y = x.view((2, 3))
    assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# Array-as-index (transforms.py normalize_key, line 82)
# ---------------------------------------------------------------------------


def test_array_as_index():
    x = npg.array(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    idx = npg.array(np.array([0, 2, 4]))
    result = x[idx]
    np.testing.assert_array_equal(result.data, [10.0, 30.0, 50.0])


# ---------------------------------------------------------------------------
# array_creation functions
# ---------------------------------------------------------------------------


def test_zeros_like_array():
    x = npg.array(np.ones((3, 4)))
    z = zeros_like(x)
    assert z.shape == (3, 4)
    np.testing.assert_array_equal(z.data, np.zeros((3, 4)))


def test_zeros_like_ndarray():
    x = np.ones((2, 3))
    z = zeros_like(x)
    assert z.shape == (2, 3)
    np.testing.assert_array_equal(z.data, np.zeros((2, 3)))


def test_arange_single_arg():
    x = arange(5)
    np.testing.assert_array_equal(x.data, [0, 1, 2, 3, 4])


def test_arange_two_args():
    x = arange(2, 7)
    np.testing.assert_array_equal(x.data, [2, 3, 4, 5, 6])


def test_linspace():
    x = linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(x.data, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_eye():
    x = eye(3)
    np.testing.assert_array_equal(x.data, np.eye(3))


def test_randn_int_shape():
    x = randn(4)
    assert x.shape == (4,)


def test_randint_no_high():
    # randint(high) with no low → low=0, high=high
    x = randint(10, size=5)
    assert x.shape == (5,)
    assert np.all(x.data >= 0) and np.all(x.data < 10)


def test_randint_int_size():
    x = randint(0, 5, size=3)
    assert x.shape == (3,)


# ---------------------------------------------------------------------------
# no_grad (context manager and decorator)
# ---------------------------------------------------------------------------


def test_no_grad_disables_graph():
    x = npg.array(np.array([1.0, 2.0]), requires_grad=True)
    with no_grad():
        y = x + x
    assert y.grad_fn is None
    assert not y.requires_grad


def test_no_grad_restores_state():
    x = npg.array(np.array([1.0]), requires_grad=True)
    with no_grad():
        pass
    y = x + x
    assert y.grad_fn is not None
    assert y.requires_grad


def test_no_grad_decorator():
    x = npg.array(np.array([1.0, 2.0]), requires_grad=True)

    @npg.no_grad()
    def f(x):
        return x + x

    y = f(x)
    assert y.grad_fn is None
    assert not y.requires_grad
