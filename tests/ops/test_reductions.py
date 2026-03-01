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
    generic_array,
    prod_safe_array,
    reduction_args,
)

npg.manual_seed(0)
npr.seed(0)


@given(generic_array())
def test_sum_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.sum()

    reference = arr.sum()
    check_equality(z.data, reference)


@given(generic_array())
def test_sum_api(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.sum(x)

    reference = arr.sum()
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_sum_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = x.sum()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.sum()

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]

    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


@given(generic_array())
def test_mean_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.mean()

    reference = arr.mean()
    check_equality(z.data, reference)


@given(generic_array())
def test_mean_api(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.mean(x)

    reference = arr.mean()
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_mean_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = x.mean()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.mean()

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]

    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


@given(reduction_args())
def test_sum_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr)
    z = x.sum(axis=axis, keepdims=keepdims)
    reference = arr.sum(axis=axis, keepdims=keepdims)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_sum_backward_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = x.sum(axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.sum(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


@given(reduction_args())
def test_mean_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr)
    z = x.mean(axis=axis, keepdims=keepdims)
    reference = arr.mean(axis=axis, keepdims=keepdims)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_mean_backward_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = x.mean(axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.mean(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- max ---


@given(generic_array())
def test_max_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.max(x)
    reference = np.max(arr)
    check_equality(z.data, reference)


@given(generic_array())
def test_max_api(arr: np.ndarray):
    x = npg.array(arr)
    z = x.max()
    reference = np.max(arr)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_max_backward(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = npg.max(x, axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    if axis is None:
        zt = xt.amax()
    else:
        zt = xt.amax(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- min ---


@given(generic_array())
def test_min_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.min(x)
    reference = np.min(arr)
    check_equality(z.data, reference)


@given(generic_array())
def test_min_api(arr: np.ndarray):
    x = npg.array(arr)
    z = x.min()
    reference = np.min(arr)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_min_backward(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = npg.min(x, axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    if axis is None:
        zt = xt.amin()
    else:
        zt = xt.amin(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- prod ---


@given(prod_safe_array())
def test_prod_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.prod(x)
    reference = np.prod(arr)
    check_equality(z.data, reference)


@given(prod_safe_array())
def test_prod_api(arr: np.ndarray):
    x = npg.array(arr)
    z = x.prod()
    reference = np.prod(arr)
    check_equality(z.data, reference)


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_prod_backward(data):
    arr, axis, keepdims = data
    # Use prod-safe values to avoid overflow and zeros for backward
    arr = npr.uniform(0.1, 1.0, size=arr.shape).astype(arr.dtype)
    x = npg.array(arr, requires_grad=True)
    z = npg.prod(x, axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    if axis is None:
        zt = xt.prod()
    else:
        zt = xt.prod(dim=axis, keepdim=keepdims)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- argmax (forward only, non-differentiable) ---


@given(generic_array())
def test_argmax_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.argmax(x)
    reference = np.argmax(arr)
    check_equality(z.data, reference)


@given(generic_array())
def test_argmax_api(arr: np.ndarray):
    x = npg.array(arr)
    z = x.argmax()
    reference = np.argmax(arr)
    check_equality(z.data, reference)


@given(reduction_args(with_axis=True))
def test_argmax_axis(data):
    arr, axis, keepdims = data
    x = npg.array(arr)
    z = npg.argmax(x, axis=axis, keepdims=keepdims)
    reference = np.argmax(arr, axis=axis, keepdims=keepdims)
    check_equality(z.data, reference)


# --- var ---


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_var_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.var()
    check_equality(z.data, np.var(arr))


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_var_axis_keepdims(data):
    arr, axis, keepdims = data
    x = npg.array(arr)
    z = x.var(axis=axis, keepdims=keepdims)
    check_equality(z.data, np.var(arr, axis=axis, keepdims=keepdims))


@given(reduction_args(dtypes=FLOAT_DTYPES))
def test_var_backward(data):
    arr, axis, keepdims = data
    x = npg.array(arr, requires_grad=True)
    z = x.var(axis=axis, keepdims=keepdims)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    if axis is None:
        zt = xt.var(unbiased=False)
    else:
        zt = xt.var(dim=axis, unbiased=False, keepdim=keepdims)
    gxt = torch.autograd.grad(zt, xt, grad_outputs=torch.ones_like(zt))[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


def test_var_ddof():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    x = npg.array(arr)
    check_equality(x.var(ddof=1).data, np.var(arr, ddof=1))


def test_var_functional():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.var(x).data, np.var(arr))


# --- cumsum ---


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_cumsum_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.cumsum()
    check_equality(z.data, np.cumsum(arr))


@given(reduction_args(dtypes=FLOAT_DTYPES, with_axis=True))
def test_cumsum_axis(data):
    arr, axis, _ = data
    x = npg.array(arr)
    z = x.cumsum(axis=axis)
    check_equality(z.data, np.cumsum(arr, axis=axis))


def test_cumsum_backward_axis():
    arr = np.random.randn(3, 4).astype(np.float64)
    for axis in [0, 1]:
        x = npg.array(arr, requires_grad=True)
        y = x.cumsum(axis=axis)
        y.sum().backward()

        xt = torch.from_numpy(arr).requires_grad_(True)
        yt = xt.cumsum(dim=axis)
        yt.sum().backward()
        assert x.grad is not None
        check_equality(x.grad, xt.grad.numpy())


def test_cumsum_backward_flat():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    y = x.cumsum()
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.flatten().cumsum(dim=0)
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy().reshape(arr.shape))


def test_cumsum_functional():
    arr = np.random.randn(4).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.cumsum(x).data, np.cumsum(arr))


# --- cumprod ---


@given(prod_safe_array())
def test_cumprod_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.cumprod()
    check_equality(z.data, np.cumprod(arr))


@given(reduction_args(dtypes=FLOAT_DTYPES, with_axis=True))
def test_cumprod_axis(data):
    arr, axis, _ = data
    # Use positive values to avoid overflow/zero issues in cumprod
    arr = np.abs(arr).astype(arr.dtype) + 0.1
    x = npg.array(arr)
    z = x.cumprod(axis=axis)
    check_equality(z.data, np.cumprod(arr, axis=axis))


def test_cumprod_backward_axis():
    # Positive non-zero values to avoid undefined grad at zero
    arr = np.abs(np.random.randn(3, 4)).astype(np.float64) + 0.1
    for axis in [0, 1]:
        x = npg.array(arr, requires_grad=True)
        y = x.cumprod(axis=axis)
        y.sum().backward()

        xt = torch.from_numpy(arr).requires_grad_(True)
        yt = xt.cumprod(dim=axis)
        yt.sum().backward()
        assert x.grad is not None
        check_equality(x.grad, xt.grad.numpy())


def test_cumprod_backward_flat():
    arr = np.abs(np.random.randn(3, 4)).astype(np.float64) + 0.1
    x = npg.array(arr, requires_grad=True)
    y = x.cumprod()
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.flatten().cumprod(dim=0)
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy().reshape(arr.shape))


def test_cumprod_functional():
    arr = np.abs(np.random.randn(4)).astype(np.float64) + 0.1
    x = npg.array(arr)
    check_equality(npg.cumprod(x).data, np.cumprod(arr))
