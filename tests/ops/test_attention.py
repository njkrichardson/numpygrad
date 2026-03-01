import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import numpygrad as npg
from numpygrad.ops.attention import MultiHeadAttentionOp
from tests.configuration import check_equality


@st.composite
def mha_config(draw):
    B = draw(st.integers(1, 2))
    T = draw(st.integers(1, 6))
    num_heads = draw(st.sampled_from([1, 2, 4]))
    k = draw(st.integers(1, 4))
    d_model = k * num_heads
    q = np.random.randn(B, T, d_model).astype(np.float64)
    k_arr = np.random.randn(B, T, d_model).astype(np.float64)
    v = np.random.randn(B, T, d_model).astype(np.float64)
    W_q = np.random.randn(d_model, d_model).astype(np.float64)
    W_k = np.random.randn(d_model, d_model).astype(np.float64)
    W_v = np.random.randn(d_model, d_model).astype(np.float64)
    W_o = np.random.randn(d_model, d_model).astype(np.float64)
    return B, T, d_model, num_heads, q, k_arr, v, W_q, W_k, W_v, W_o


def _torch_mha_no_bias(d_model, num_heads, W_q, W_k, W_v, W_o):
    mha_t = torch.nn.MultiheadAttention(
        d_model, num_heads, bias=False, batch_first=True, dtype=torch.float64
    )
    with torch.no_grad():
        mha_t.in_proj_weight.copy_(torch.from_numpy(np.vstack([W_q, W_k, W_v])))
        mha_t.out_proj.weight.copy_(torch.from_numpy(W_o))
    return mha_t


@settings(deadline=None)
@given(mha_config())
def test_mha_fwd_no_bias(cfg):
    B, T, d_model, num_heads, q, k, v, W_q, W_k, W_v, W_o = cfg

    out_n = MultiHeadAttentionOp.apply(
        npg.array(q),
        npg.array(k),
        npg.array(v),
        npg.array(W_q),
        npg.array(W_k),
        npg.array(W_v),
        npg.array(W_o),
        None,
        None,
        None,
        None,
        num_heads,
        None,
    )

    mha_t = _torch_mha_no_bias(d_model, num_heads, W_q, W_k, W_v, W_o)
    out_t, _ = mha_t(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(v),
        need_weights=False,
    )
    check_equality(out_n.data, out_t.detach().numpy(), rtol=1e-5)


@settings(deadline=None)
@given(mha_config())
def test_mha_bwd_no_bias(cfg):
    B, T, d_model, num_heads, q, k, v, W_q, W_k, W_v, W_o = cfg

    q_n = npg.array(q, requires_grad=True)
    k_n = npg.array(k, requires_grad=True)
    v_n = npg.array(v, requires_grad=True)
    W_q_n = npg.array(W_q, requires_grad=True)
    W_k_n = npg.array(W_k, requires_grad=True)
    W_v_n = npg.array(W_v, requires_grad=True)
    W_o_n = npg.array(W_o, requires_grad=True)

    out_n = MultiHeadAttentionOp.apply(
        q_n,
        k_n,
        v_n,
        W_q_n,
        W_k_n,
        W_v_n,
        W_o_n,
        None,
        None,
        None,
        None,
        num_heads,
        None,
    )
    out_n.backward()

    mha_t = _torch_mha_no_bias(d_model, num_heads, W_q, W_k, W_v, W_o)
    q_t = torch.from_numpy(q).requires_grad_(True)
    k_t = torch.from_numpy(k).requires_grad_(True)
    v_t = torch.from_numpy(v).requires_grad_(True)
    out_t, _ = mha_t(q_t, k_t, v_t, need_weights=False)
    grads_t = torch.autograd.grad(
        out_t,
        (q_t, k_t, v_t, mha_t.in_proj_weight, mha_t.out_proj.weight),
        grad_outputs=torch.ones_like(out_t),
    )

    check_equality(q_n.grad, grads_t[0].numpy(), rtol=1e-4)
    check_equality(k_n.grad, grads_t[1].numpy(), rtol=1e-4)
    check_equality(v_n.grad, grads_t[2].numpy(), rtol=1e-4)
    in_proj_grad = grads_t[3].numpy()
    check_equality(W_q_n.grad, in_proj_grad[:d_model], rtol=1e-4)
    check_equality(W_k_n.grad, in_proj_grad[d_model : 2 * d_model], rtol=1e-4)
    check_equality(W_v_n.grad, in_proj_grad[2 * d_model :], rtol=1e-4)
    check_equality(W_o_n.grad, grads_t[4].numpy(), rtol=1e-4)
