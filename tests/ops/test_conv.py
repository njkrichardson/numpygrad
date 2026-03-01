import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

import numpygrad as npg
from tests.configuration import check_equality

# ------------------------------------------------------------------
# Hypothesis strategy: small conv2d problem
# ------------------------------------------------------------------

_small = st.integers(min_value=1, max_value=3)
_spatial = st.integers(min_value=1, max_value=8)
_channels = st.integers(min_value=1, max_value=3)


@st.composite
def conv2d_config(draw, with_bias: bool = False):
    """Draw (x_np, w_np, b_np_or_None, stride, padding) for a valid conv."""
    N = draw(st.integers(min_value=1, max_value=2))
    C_in = draw(_channels)
    C_out = draw(_channels)
    H = draw(_spatial)
    W = draw(_spatial)
    kH = draw(st.integers(min_value=1, max_value=min(3, H)))
    kW = draw(st.integers(min_value=1, max_value=min(3, W)))
    stride = draw(st.integers(min_value=1, max_value=2))
    padding = draw(st.integers(min_value=0, max_value=1))

    # ensure output is at least 1x1
    out_h = (H + 2 * padding - kH) // stride + 1
    out_w = (W + 2 * padding - kW) // stride + 1
    if out_h < 1 or out_w < 1:
        # fall back to stride=1, padding=0 to guarantee valid output
        stride, padding = 1, 0

    x = np.random.randn(N, C_in, H, W).astype(np.float64)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float64)
    b = np.random.randn(C_out).astype(np.float64) if with_bias else None
    return x, w, b, stride, padding


# ------------------------------------------------------------------
# Forward tests (no grad)
# ------------------------------------------------------------------


@settings(deadline=None)
@given(conv2d_config(with_bias=False))
def test_conv2d_fwd_no_bias(cfg):
    x, w, _, stride, padding = cfg
    out = npg.conv2d(npg.array(x), npg.array(w), stride=stride, padding=padding)
    ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), stride=stride, padding=padding).numpy()
    check_equality(out.data, ref, rtol=1e-10)


@settings(deadline=None)
@given(conv2d_config(with_bias=True))
def test_conv2d_fwd_with_bias(cfg):
    x, w, b, stride, padding = cfg
    out = npg.conv2d(npg.array(x), npg.array(w), npg.array(b), stride=stride, padding=padding)
    ref = F.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(w),
        torch.from_numpy(b),
        stride=stride,
        padding=padding,
    ).numpy()
    check_equality(out.data, ref, rtol=1e-10)


# ------------------------------------------------------------------
# Backward tests (grad check vs PyTorch)
# ------------------------------------------------------------------


@settings(deadline=None)
@given(conv2d_config(with_bias=False))
def test_conv2d_bwd_no_bias(cfg):
    x, w, _, stride, padding = cfg
    xn = npg.array(x, requires_grad=True)
    wn = npg.array(w, requires_grad=True)
    out = npg.conv2d(xn, wn, stride=stride, padding=padding)
    out.backward()

    xt = torch.from_numpy(x).requires_grad_(True)
    wt = torch.from_numpy(w).requires_grad_(True)
    outt = F.conv2d(xt, wt, stride=stride, padding=padding)
    gx, gw = torch.autograd.grad(outt, (xt, wt), grad_outputs=torch.ones_like(outt))

    assert xn.grad is not None
    assert wn.grad is not None
    check_equality(xn.grad, gx.numpy(), rtol=1e-10)
    check_equality(wn.grad, gw.numpy(), rtol=1e-10)


@settings(deadline=None)
@given(conv2d_config(with_bias=True))
def test_conv2d_bwd_with_bias(cfg):
    x, w, b, stride, padding = cfg
    xn = npg.array(x, requires_grad=True)
    wn = npg.array(w, requires_grad=True)
    bn = npg.array(b, requires_grad=True)
    out = npg.conv2d(xn, wn, bn, stride=stride, padding=padding)
    out.backward()

    xt = torch.from_numpy(x).requires_grad_(True)
    wt = torch.from_numpy(w).requires_grad_(True)
    bt = torch.from_numpy(b).requires_grad_(True)
    outt = F.conv2d(xt, wt, bt, stride=stride, padding=padding)
    gx, gw, gb = torch.autograd.grad(outt, (xt, wt, bt), grad_outputs=torch.ones_like(outt))

    assert xn.grad is not None
    assert wn.grad is not None
    assert bn.grad is not None
    check_equality(xn.grad, gx.numpy(), rtol=1e-10)
    check_equality(wn.grad, gw.numpy(), rtol=1e-10)
    check_equality(bn.grad, gb.numpy(), rtol=1e-10)
