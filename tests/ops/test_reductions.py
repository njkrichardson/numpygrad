from hypothesis import given
import numpy as np
import torch

import numpygrad as npg
from tests.configuration import (
    check_equality,
)
from tests.strategies import (
    generic_array,
    FLOAT_DTYPES,
)

npg.manual_seed(0)



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