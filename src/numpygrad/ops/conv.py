from typing import cast

import numpy as np

from numpygrad.core.array import Array, ArrayCoercible
from numpygrad.core.function import Context, Function
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, register
from numpygrad.ops.core import ensure_array


def _im2col(
    x_pad: np.ndarray,
    kH: int,
    kW: int,
    sH: int,
    sW: int,
) -> np.ndarray:
    """Extract sliding patches; returns (N*out_h*out_w, C*kH*kW)."""
    N, C, H_pad, W_pad = x_pad.shape
    out_h = (H_pad - kH) // sH + 1
    out_w = (W_pad - kW) // sW + 1
    col = np.zeros((N, C, kH, kW, out_h, out_w), dtype=x_pad.dtype)
    for y in range(kH):
        for x in range(kW):
            col[:, :, y, x, :, :] = x_pad[:, :, y : y + sH * out_h : sH, x : x + sW * out_w : sW]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def _col2im(
    cols: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kH: int,
    kW: int,
    sH: int,
    sW: int,
    pH: int,
    pW: int,
) -> np.ndarray:
    """Scatter gradient patches back to padded input, then unpad."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pH - kH) // sH + 1
    out_w = (W + 2 * pW - kW) // sW + 1
    x_pad = np.zeros((N, C, H + 2 * pH, W + 2 * pW), dtype=cols.dtype)
    cols_r = cols.reshape(N, out_h, out_w, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    for y in range(kH):
        for x in range(kW):
            x_pad[:, :, y : y + sH * out_h : sH, x : x + sW * out_w : sW] += cols_r[
                :, :, y, x, :, :
            ]
    return x_pad[:, :, pH : pH + H, pW : pW + W]


class Conv2dOp(Function):
    @staticmethod
    def forward(
        ctx: Context,
        inp: Array,
        weight: Array,
        bias: Array | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Array:
        inp, weight = ensure_array(inp), ensure_array(weight)
        N, C_in, H, W = inp.data.shape
        C_out, _, kH, kW = weight.data.shape
        sH, sW = stride
        pH, pW = padding

        x_pad = np.pad(inp.data, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
        col = _im2col(x_pad, kH, kW, sH, sW)  # (N*out_h*out_w, C_in*kH*kW)
        W_mat = weight.data.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        out = col @ W_mat.T  # (N*out_h*out_w, C_out)

        out_h = (H + 2 * pH - kH) // sH + 1
        out_w = (W + 2 * pW - kW) // sW + 1
        out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)  # (N, C_out, out_h, out_w)

        if bias is not None:
            bias_arr = ensure_array(bias)
            out = out + bias_arr.data[None, :, None, None]
            ctx.store(inp, weight, bias_arr)
        else:
            ctx.store(inp, weight)
        has_bias = bias is not None

        ctx.input_shape = inp.data.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx._col = col  # type: ignore[attr-defined]
        ctx._has_bias = has_bias  # type: ignore[attr-defined]

        return Array(
            out, device=inp.device, requires_grad=inp.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx: Context, grad_out: np.ndarray):
        has_bias: bool = ctx._has_bias  # type: ignore[attr-defined]
        _, weight = ctx.saved_arrays[0], ctx.saved_arrays[1]
        col: np.ndarray = ctx._col  # type: ignore[attr-defined]
        sH, sW = ctx.stride
        pH, pW = ctx.padding
        C_out, C_in, kH, kW = weight.data.shape
        N = ctx.input_shape[0]
        out_h, out_w = grad_out.shape[2], grad_out.shape[3]

        # (N, C_out, out_h, out_w) → (N*out_h*out_w, C_out)
        g = grad_out.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, C_out)

        grad_weight = (g.T @ col).reshape(C_out, C_in, kH, kW)
        grad_col = g @ weight.data.reshape(C_out, -1)
        grad_inp = _col2im(
            grad_col, cast(tuple[int, int, int, int], ctx.input_shape), kH, kW, sH, sW, pH, pW
        )
        grad_bias = grad_out.sum(axis=(0, 2, 3)) if has_bias else None

        return grad_inp, grad_weight, grad_bias


@register(OperatorId.CONV2D)
def conv2d_cpu(
    inp: ArrayCoercible,
    weight: ArrayCoercible,
    bias: ArrayCoercible | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> Array:
    inp, weight = ensure_array(inp), ensure_array(weight)
    N, C_in, H, W = inp.data.shape
    C_out, _, kH, kW = weight.data.shape
    sH, sW = stride
    pH, pW = padding

    x_pad = np.pad(inp.data, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
    col = _im2col(x_pad, kH, kW, sH, sW)
    W_mat = weight.data.reshape(C_out, -1)
    out = col @ W_mat.T

    out_h = (H + 2 * pH - kH) // sH + 1
    out_w = (W + 2 * pW - kW) // sW + 1
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    if bias is not None:
        bias_arr = ensure_array(bias)
        out = out + bias_arr.data[None, :, None, None]

    return Array(out, device=inp.device, requires_grad=False)


@register(OperatorId.CONV2D, op_requirements=OperatorRequirements.Autograd)
def conv2d_autograd(
    inp: ArrayCoercible,
    weight: ArrayCoercible,
    bias: ArrayCoercible | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> Array:
    return Conv2dOp.apply(inp, weight, bias, stride, padding)
