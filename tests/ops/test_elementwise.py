from hypothesis import given, strategies as st
import numpy as np
import numpy.random as npr
import torch

import numpygrad as npg
from tests.strategies import (
    generic_array,
    shape_nd,
    array_pair,
    positive_array,
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


# --- exp ---


@given(generic_array())
def test_exp_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.exp()
    reference = np.exp(arr)
    check_equality(z.data, reference)


@given(generic_array())
def test_exp_api(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.exp(x)
    reference = np.exp(arr)
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_exp_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = x.exp()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.exp()
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- log ---


@given(positive_array())
def test_log_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.log()
    reference = np.log(arr)
    check_equality(z.data, reference)


@given(positive_array())
def test_log_api(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.log(x)
    reference = np.log(arr)
    check_equality(z.data, reference)


@given(positive_array())
def test_log_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = x.log()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.log()
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- abs ---


@given(generic_array())
def test_abs_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = x.abs()
    reference = np.abs(arr)
    check_equality(z.data, reference)


@given(generic_array())
def test_abs_api(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.abs(x)
    reference = np.abs(arr)
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_abs_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = x.abs()
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.abs()
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- clip ---


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_clip_basic(arr: np.ndarray):
    low = float(np.min(arr)) - 1.0
    high = float(np.max(arr)) + 1.0
    x = npg.array(arr)
    z = npg.clip(x, low, high)
    reference = np.clip(arr, low, high)
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_clip_backward(arr: np.ndarray):
    low = float(np.min(arr)) - 1.0
    high = float(np.max(arr)) + 1.0
    x = npg.array(arr, requires_grad=True)
    z = npg.clip(x, low, high)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = torch.clamp(xt, low, high)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- maximum ---


@given(array_pair(same_shape=True))
def test_maximum_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.maximum(x, y)
    reference = np.maximum(A, B)
    check_equality(z.data, reference)


@given(array_pair(broadcastable=True))
def test_maximum_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.maximum(x, y)
    reference = np.maximum(A, B)
    check_equality(z.data, reference)


@given(array_pair(same_shape=True, dtypes=FLOAT_DTYPES))
def test_maximum_backward_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = npg.maximum(x, y)
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = torch.maximum(xt, yt)
    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(broadcastable=True, dtypes=FLOAT_DTYPES))
def test_maximum_backward_bcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = npg.maximum(x, y)
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = torch.maximum(xt, yt)
    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


# --- minimum ---


@given(array_pair(same_shape=True))
def test_minimum_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.minimum(x, y)
    reference = np.minimum(A, B)
    check_equality(z.data, reference)


@given(array_pair(broadcastable=True))
def test_minimum_broadcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A)
    y = npg.array(B)
    z = npg.minimum(x, y)
    reference = np.minimum(A, B)
    check_equality(z.data, reference)


@given(array_pair(same_shape=True, dtypes=FLOAT_DTYPES))
def test_minimum_backward_basic(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = npg.minimum(x, y)
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = torch.minimum(xt, yt)
    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())


@given(array_pair(broadcastable=True, dtypes=FLOAT_DTYPES))
def test_minimum_backward_bcast(arrs: tuple[np.ndarray, np.ndarray]):
    A, B = arrs
    x = npg.array(A, requires_grad=True)
    y = npg.array(B, requires_grad=True)
    z = npg.minimum(x, y)
    z.backward()

    xt = torch.from_numpy(A).requires_grad_(True)
    yt = torch.from_numpy(B).requires_grad_(True)
    zt = torch.minimum(xt, yt)
    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )
    assert x.grad is not None and y.grad is not None
    check_equality(x.grad, gxt.numpy())
    check_equality(y.grad, gyt.numpy())
