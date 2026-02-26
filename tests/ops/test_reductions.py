import numpy as np
import torch

import numpygrad as npg

npg.manual_seed(0)


def test_sum_basic():
    xshape = (2,)
    x = npg.ones(xshape)
    z = x.sum()

    reference = np.ones(xshape).sum()
    np.testing.assert_array_equal(z.data, reference)


def test_sum_api():
    xshape = (2,)
    x = npg.ones(xshape)
    z = npg.sum(x)

    reference = np.ones(xshape).sum()
    np.testing.assert_array_equal(z.data, reference)


def test_sum_backward():
    xshape = (2,)
    x = npg.ones(xshape, requires_grad=True)
    z = x.sum()
    z.backward()

    xt = torch.ones(xshape, requires_grad=True)
    zt = xt.sum()

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]

    np.testing.assert_array_equal(x.grad, gxt.numpy())


def test_mean_basic():
    xshape = (2,)
    x = npg.ones(xshape)
    z = x.mean()

    reference = np.ones(xshape).mean()
    np.testing.assert_array_equal(z.data, reference)


def test_mean_api():
    xshape = (2,)
    x = npg.ones(xshape)
    z = npg.mean(x)

    reference = np.ones(xshape).mean()
    np.testing.assert_array_equal(z.data, reference)


def test_mean_backward():
    xshape = (2,)
    x = npg.ones(xshape, requires_grad=True)
    z = x.mean()
    z.backward()

    xt = torch.ones(xshape, requires_grad=True)
    zt = xt.mean()

    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]

    np.testing.assert_array_equal(x.grad, gxt.numpy())
