from hypothesis import given
import numpy as np
import numpy.random as npr
import torch

import numpygrad as npg
from tests.configuration import (
    check_equality,
)
from tests.strategies import (
    generic_array,
    positive_array,
    prod_safe_array,
    reduction_args,
    FLOAT_DTYPES,
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