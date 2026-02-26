import numpy as np
import torch

import numpygrad as npg

npg.manual_seed(0)


def test_add_basic():
    xshape = (2,)
    yshape = (2,)

    x = npg.ones(xshape)
    y = npg.ones(yshape)
    z = x + y

    reference = np.ones(xshape) + np.ones(yshape)
    np.testing.assert_array_equal(z.data, reference)


def test_add_broadcast():
    xshape = (2, 1)
    yshape = (1, 2)

    x = npg.ones(xshape)
    y = npg.ones(yshape)
    z = x + y

    reference = np.ones(xshape) + np.ones(yshape)
    np.testing.assert_array_equal(z.data, reference)


def test_add_backward():
    xshape = (2, 1)
    yshape = (1, 2)

    x = npg.ones(xshape, requires_grad=True)
    y = npg.ones(yshape, requires_grad=True)
    z = x + y
    z.backward()

    xt = torch.ones(xshape, requires_grad=True)
    yt = torch.ones(yshape, requires_grad=True)
    zt = xt + yt

    gxt, gyt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt, yt),
        grad_outputs=torch.ones_like(zt),
    )

    np.testing.assert_array_equal(x.grad, gxt.numpy())
    np.testing.assert_array_equal(y.grad, gyt.numpy())
