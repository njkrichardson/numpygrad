"""Tests for sin, cos, tan, floor, ceil, sign, copy, where, argmin, r-ops,
array creation (full/ones_like/empty_like), and constants."""

import numpy as np
import numpy.random as npr
import torch
from hypothesis import given
from hypothesis import strategies as st

import numpygrad as npg
from tests.configuration import check_equality
from tests.strategies import FLOAT_DTYPES, generic_array, shape_nd

npg.manual_seed(0)
npr.seed(0)


# ---------------------------------------------------------------------------
# sin
# ---------------------------------------------------------------------------


@given(generic_array())
def test_sin_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.sin(x).data, np.sin(arr))


@given(generic_array())
def test_sin_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.sin().data, np.sin(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_sin_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.sin(x).backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    torch.sin(xt).backward(torch.ones_like(xt))

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


# ---------------------------------------------------------------------------
# cos
# ---------------------------------------------------------------------------


@given(generic_array())
def test_cos_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.cos(x).data, np.cos(arr))


@given(generic_array())
def test_cos_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.cos().data, np.cos(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_cos_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.cos(x).backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    torch.cos(xt).backward(torch.ones_like(xt))

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


# ---------------------------------------------------------------------------
# tan  (avoid |x| near pi/2 to prevent sec^2 blow-up)
# ---------------------------------------------------------------------------


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_tan_forward(arr: np.ndarray):
    arr = np.clip(np.atleast_1d(arr).astype(np.float32), -1.0, 1.0)
    x = npg.array(arr)
    check_equality(npg.tan(x).data, np.tan(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_tan_method(arr: np.ndarray):
    arr = np.clip(np.atleast_1d(arr).astype(np.float32), -1.0, 1.0)
    x = npg.array(arr)
    check_equality(x.tan().data, np.tan(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_tan_backward(arr: np.ndarray):
    arr = np.clip(np.atleast_1d(arr).astype(np.float32), -1.0, 1.0)
    x = npg.array(arr, requires_grad=True)
    npg.tan(x).backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    torch.tan(xt).backward(torch.ones_like(xt))

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


# ---------------------------------------------------------------------------
# floor / ceil / sign — forward only (zero-gradient ops)
# ---------------------------------------------------------------------------


@given(generic_array())
def test_floor_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.floor(x).data, np.floor(arr))


@given(generic_array())
def test_floor_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.floor().data, np.floor(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_floor_backward_is_zero(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.floor(x).backward()
    assert x.grad is not None
    assert np.all(x.grad == 0)


@given(generic_array())
def test_ceil_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.ceil(x).data, np.ceil(arr))


@given(generic_array())
def test_ceil_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.ceil().data, np.ceil(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_ceil_backward_is_zero(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.ceil(x).backward()
    assert x.grad is not None
    assert np.all(x.grad == 0)


@given(generic_array())
def test_sign_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.sign(x).data, np.sign(arr))


@given(generic_array())
def test_sign_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.sign().data, np.sign(arr))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_sign_backward_is_zero(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.sign(x).backward()
    assert x.grad is not None
    assert np.all(x.grad == 0)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


@given(generic_array())
def test_copy_forward(arr: np.ndarray):
    x = npg.array(arr)
    y = npg.copy(x)
    check_equality(y.data, arr)
    # must be an independent copy
    x.data[...] = 0
    assert not np.all(y.data == 0) or np.all(arr == 0)


@given(generic_array())
def test_copy_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.copy().data, arr)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_copy_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    npg.copy(x).backward()
    assert x.grad is not None
    check_equality(x.grad, np.ones_like(arr))


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_where_forward(A: np.ndarray):
    A = np.atleast_1d(A)
    B = npr.randn(*A.shape).astype(A.dtype)
    cond = A > 0
    x = npg.array(A)
    y = npg.array(B)
    c = npg.array(cond)
    check_equality(npg.where(c, x, y).data, np.where(cond, A, B))


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_where_backward(A: np.ndarray):
    A = np.atleast_1d(A)
    B = npr.randn(*A.shape).astype(A.dtype)
    cond = A > 0

    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    npg.where(npg.array(cond), x, y).backward()

    At = torch.from_numpy(A).requires_grad_(True)
    Bt = torch.from_numpy(B).requires_grad_(True)
    ct = torch.from_numpy(cond)
    torch.where(ct, At, Bt).backward(torch.ones_like(At))

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, At.grad.numpy())
    check_equality(y.grad, Bt.grad.numpy())


# ---------------------------------------------------------------------------
# argmin
# ---------------------------------------------------------------------------


@given(generic_array())
def test_argmin_forward(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(npg.argmin(x).data, np.argmin(arr))


@given(generic_array())
def test_argmin_method(arr: np.ndarray):
    x = npg.array(arr)
    check_equality(x.argmin().data, np.argmin(arr))


@given(shape_nd(min_num_dims=2, max_num_dims=3), st.integers(0, 1))
def test_argmin_axis(shape, axis):
    if isinstance(shape, int):
        shape = (shape, shape)
    arr = npr.randn(*shape).astype(np.float32)
    x = npg.array(arr)
    check_equality(npg.argmin(x, axis=axis).data, np.argmin(arr, axis=axis))


# ---------------------------------------------------------------------------
# r-ops: __radd__, __rsub__, __rtruediv__
# ---------------------------------------------------------------------------


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_radd(arr: np.ndarray):
    x = npg.array(arr)
    check_equality((5 + x).data, 5 + arr)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_rsub(arr: np.ndarray):
    x = npg.array(arr)
    check_equality((5 - x).data, 5 - arr)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_rtruediv(arr: np.ndarray):
    # avoid division by near-zero
    arr = np.where(np.abs(arr) < 0.1, 0.5, arr)
    x = npg.array(arr)
    check_equality((2.0 / x).data, 2.0 / arr)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_radd_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    (5.0 + x).backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    (5.0 + xt).backward(torch.ones_like(xt))

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_rsub_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    (5.0 - x).backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    (5.0 - xt).backward(torch.ones_like(xt))

    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


# ---------------------------------------------------------------------------
# Array creation: full, ones_like, empty_like
# ---------------------------------------------------------------------------


def test_full():
    x = npg.full((2, 3), 7.0)
    assert x.shape == (2, 3)
    assert np.all(x.data == 7.0)


def test_ones_like():
    base = npg.array(np.zeros((3, 4), dtype=np.float32))
    x = npg.ones_like(base)
    assert x.shape == (3, 4)
    assert np.all(x.data == 1.0)


def test_ones_like_ndarray():
    base = np.zeros((2, 5), dtype=np.float64)
    x = npg.ones_like(base)
    assert x.shape == (2, 5)
    assert np.all(x.data == 1.0)


def test_empty_like():
    base = npg.array(np.zeros((3, 4), dtype=np.float32))
    x = npg.empty_like(base)
    assert x.shape == (3, 4)
    assert x.dtype == np.float32


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert npg.pi == np.pi
    assert npg.e == np.e
    assert np.isinf(npg.inf)
    assert np.isnan(npg.nan)
    assert npg.newaxis is None


# ---------------------------------------------------------------------------
# isnan / isinf / isfinite
# ---------------------------------------------------------------------------


def test_isnan():
    arr = np.array([1.0, float("nan"), 0.0])
    x = npg.array(arr)
    check_equality(npg.isnan(x).data, np.isnan(arr))


def test_isinf():
    arr = np.array([1.0, float("inf"), float("-inf")])
    x = npg.array(arr)
    check_equality(npg.isinf(x).data, np.isinf(arr))


def test_isfinite():
    arr = np.array([1.0, float("inf"), float("nan")])
    x = npg.array(arr)
    check_equality(npg.isfinite(x).data, np.isfinite(arr))


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------


def test_concatenate_alias():
    a = npg.array(np.ones((2, 3)))
    b = npg.array(np.zeros((2, 3)))
    c = npg.concatenate((a, b), axis=0)
    assert c.shape == (4, 3)


def test_expand_dims_alias():
    a = npg.array(np.ones((3,)))
    b = npg.expand_dims(a, axis=0)
    assert b.shape == (1, 3)
