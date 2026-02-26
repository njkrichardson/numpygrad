from hypothesis import given, strategies as st
import numpy as np
import torch

import numpygrad as npg
from tests.configuration import (
    check_equality,
)
from tests.strategies import (
    shape_nd,
    array_pair,
    FLOAT_DTYPES,
)
npg.manual_seed(0)


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
