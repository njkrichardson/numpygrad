from hypothesis import given, strategies as st
import numpy as np
import numpy.random as npr
import torch

import numpygrad as npg
from tests.strategies import (
    generic_array,
    shape_nd,
    array_pair,
    FLOAT_DTYPES,
)
from tests.configuration import (
    check_equality,
    VALUE_RANGE,
    POW_RANGE,
)

npg.manual_seed(0)
npr.seed(0)


@given(generic_array(), st.integers(*VALUE_RANGE))
def test_add_constant(arr: np.ndarray, constant: int):
    x = npg.array(arr)
    z = x + constant

    reference = arr + constant
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_add_ndarray(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = B
    z = x + y

    reference = A + B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_add_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x + y

    reference = A + B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_add_api(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.add(x, y)

    reference = A + B
    check_equality(z.data, reference)


@given(array_pair(broadcastable=True))
def test_add_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs

    x = npg.array(A)
    y = npg.array(B)
    z = x + y

    reference = A + B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True, dtypes=FLOAT_DTYPES))
def test_add_backward_basic(arrs):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x + y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt + yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(dtypes=FLOAT_DTYPES, broadcastable=True))
def test_add_backward_bcast(arrs):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x + y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt + yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(generic_array(), st.integers(*VALUE_RANGE))
def test_mul_constant(arr: np.ndarray, constant: int):
    x = npg.array(arr)
    z = x * constant

    reference = arr * constant
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_mul_ndarray(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    z = x * B

    reference = A * B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_mul_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x * y

    reference = A * B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True))
def test_mul_api(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.mul(x, y)

    reference = A * B
    check_equality(z.data, reference)


@given(array_pair(broadcastable=True))
def test_mul_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = x * y

    reference = A * B
    check_equality(z.data, reference)


@given(array_pair(dtypes=FLOAT_DTYPES, same_shape=True))
def test_mul_backward_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x * y
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = xt * yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(generic_array(dtypes=FLOAT_DTYPES), st.integers(*POW_RANGE))
def test_pow_basic(arr: np.ndarray, constant: int):
    x = npg.array(arr)
    z = x**constant

    reference = arr**constant
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES), st.integers(*POW_RANGE))
def test_pow_backward(arr: np.ndarray, constant: int):
    x = npg.array(arr, requires_grad=True)
    z = x**constant
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt**constant

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


def _make_safe_for_div(A: np.ndarray, margin: float = 1e-2) -> np.ndarray:
    return np.where(
        A >= 0, np.maximum(A, margin), np.minimum(A, -margin)
    )  # avoids extreme values


@given(generic_array())
def test_div_constant(arr: np.ndarray):
    arr = _make_safe_for_div(arr)  # avoid extreme values
    constant = _make_safe_for_div(npr.uniform(low=-8, high=8, size=(1,))).item()
    x = npg.array(arr)  # avoid extreme values
    z = x / constant
    reference = arr / constant
    check_equality(z.data, reference)


@given(shape_nd())
def test_div_ndarray(shape):
    A = _make_safe_for_div(npr.uniform(*VALUE_RANGE, size=shape))
    B = _make_safe_for_div(npr.uniform(*VALUE_RANGE, size=shape))
    x = npg.array(A)
    z = x / B

    reference = A / B
    check_equality(z.data, reference)


@given(array_pair(same_shape=True, dtypes=FLOAT_DTYPES))
def test_div_backward_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    A = _make_safe_for_div(A)  # avoid extreme values
    B = _make_safe_for_div(B)  # avoid division by zero
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x / y
    z.backward()

    xt = torch.from_numpy(np.array(A)).requires_grad_(True)
    yt = torch.from_numpy(np.array(B)).requires_grad_(True)
    zt = xt / yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(broadcastable=True, dtypes=FLOAT_DTYPES))
def test_div_backward_bcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    A = _make_safe_for_div(A)  # avoid extreme values
    B = _make_safe_for_div(B)  # avoid division by zero
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = x / y
    z.backward()

    xt = torch.from_numpy(np.array(A)).requires_grad_(True)
    yt = torch.from_numpy(np.array(B)).requires_grad_(True)
    zt = xt / yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(generic_array())
def test_relu_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.relu(x)

    reference = np.maximum(0.0, arr)
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_relu_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    y = npg.relu(x)
    z = y.sum()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = torch.nn.functional.relu(xt)
    zt = yt.sum()

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())
