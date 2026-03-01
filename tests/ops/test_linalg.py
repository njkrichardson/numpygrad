import numpy as np
import numpy.random as npr
import torch
from hypothesis import given

import numpygrad as npg
from tests.configuration import (
    check_equality,
)
from tests.strategies import (
    FLOAT_DTYPES,
    array_pair,
    dot_1d_pair,
    generic_array,
    reduction_args,
)

npg.manual_seed(0)
npr.seed(0)


@given(array_pair(min_dims=2, max_dims=2, mm_broadcastable=True))
def test_mm_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x @ y

    reference = A @ B
    check_equality(z.data, reference)


@given(array_pair(min_dims=2, max_dims=2, mm_broadcastable=True))
def test_mm_api(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.mm(x, y)

    reference = A @ B
    check_equality(z.data, reference)


@given(array_pair(min_dims=3, max_dims=3, mm_broadcastable=True))
def test_mm_batched(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x @ y

    reference = A @ B
    check_equality(z.data, reference)


@given(array_pair(min_dims=3, mm_broadcastable=True))
def test_mm_batched_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x @ y

    reference = A @ B
    check_equality(z.data, reference)


@given(array_pair(min_dims=2, max_dims=2, mm_broadcastable=True, dtypes=FLOAT_DTYPES))
def test_mm_basic_bwd(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x @ y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt @ yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None

    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(min_dims=3, max_dims=3, mm_broadcastable=True, dtypes=FLOAT_DTYPES))
def test_mm_bwd_batched(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x @ y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt @ yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(min_dims=3, mm_broadcastable=True, dtypes=FLOAT_DTYPES))
def test_mm_bwd_batched_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x @ y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt @ yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


# --- dot (1D inner product, 2D matrix product) ---


@given(dot_1d_pair())
def test_dot_1d_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.dot(x, y)
    reference = np.dot(A, B)
    check_equality(z.data, reference)


@given(dot_1d_pair(dtypes=FLOAT_DTYPES))
def test_dot_1d_backward(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = npg.dot(x, y)
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = torch.dot(xt, yt)
    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(min_dims=2, max_dims=2, mm_broadcastable=True))
def test_dot_2d_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.dot(x, y)
    reference = np.dot(A, B)
    check_equality(z.data, reference)


# Dot 2D backward not tested: numpygrad Dot backward is for scalar output (1D-1D) only.
# 2D dot forward uses np.dot (matrix product) but backward uses grad_out * b.data which
# assumes scalar grad_out.


# --- matmul with 1D vectors ---


def test_mm_1d_vec_times_matrix():
    """1D @ 2D: vector-matrix product."""
    A = np.random.randn(3).astype(np.float64)
    B = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(A)
    y = npg.array(B)
    z = x @ y
    check_equality(z.data, A @ B)


def test_mm_matrix_times_1d_vec():
    """2D @ 1D: matrix-vector product."""
    A = np.random.randn(4, 3).astype(np.float64)
    B = np.random.randn(3).astype(np.float64)
    x = npg.array(A)
    y = npg.array(B)
    z = x @ y
    check_equality(z.data, A @ B)


def test_mm_1d_vec_times_matrix_backward():
    """Backward through 1D @ 2D."""
    A = np.random.randn(3).astype(np.float64)
    B = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x @ y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt @ yt
    gxt, gyt = torch.autograd.grad(zt, (xt, yt), grad_outputs=torch.ones_like(zt))
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


def test_mm_matrix_times_1d_vec_backward():
    """Backward through 2D @ 1D."""
    A = np.random.randn(4, 3).astype(np.float64)
    B = np.random.randn(3).astype(np.float64)
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x @ y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt @ yt
    gxt, gyt = torch.autograd.grad(zt, (xt, yt), grad_outputs=torch.ones_like(zt))
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


# --- norm ---


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_norm_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.norm(x)
    reference = np.linalg.norm(arr)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_norm_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr)
    z = npg.norm(x, axis=axis, keepdims=keepdims)
    reference = np.linalg.norm(arr, axis=axis, keepdims=keepdims)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_norm_backward(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = npg.norm(x, axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    if axis is None:
        zt = xt.norm()
    else:
        zt = xt.norm(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- diagonal ---


def test_diagonal_basic():
    arr = np.random.randn(4, 5).astype(np.float64)
    x = npg.array(arr)
    check_equality(x.diagonal().data, np.diagonal(arr))


def test_diagonal_offset_positive():
    arr = np.random.randn(4, 5).astype(np.float64)
    x = npg.array(arr)
    check_equality(x.diagonal(offset=1).data, np.diagonal(arr, offset=1))


def test_diagonal_offset_negative():
    arr = np.random.randn(4, 5).astype(np.float64)
    x = npg.array(arr)
    check_equality(x.diagonal(offset=-1).data, np.diagonal(arr, offset=-1))


def test_diagonal_backward():
    arr = np.random.randn(4, 5).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    y = x.diagonal()
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.diagonal()
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


def test_diagonal_backward_offset():
    arr = np.random.randn(4, 5).astype(np.float64)
    for offset in [-2, -1, 0, 1, 2]:
        x = npg.array(arr, requires_grad=True)
        y = x.diagonal(offset=offset)
        y.sum().backward()

        xt = torch.from_numpy(arr).requires_grad_(True)
        yt = xt.diagonal(offset=offset)
        yt.sum().backward()
        assert x.grad is not None
        check_equality(x.grad, xt.grad.numpy())


def test_diagonal_3d():
    arr = np.random.randn(3, 4, 5).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    y = x.diagonal(axis1=1, axis2=2)
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.diagonal(dim1=1, dim2=2)
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


def test_diagonal_functional():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.diagonal(x).data, np.diagonal(arr))


def test_trace_method():
    arr = np.random.randn(4, 4).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    z = x.trace()
    z.backward()

    check_equality(z.data, np.trace(arr))
    # gradient of trace is identity matrix
    check_equality(x.grad, np.eye(4))


def test_trace_functional():
    arr = np.random.randn(3, 3).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.trace(x).data, np.trace(arr))
