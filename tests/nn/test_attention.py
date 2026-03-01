import numpy as np
import torch

import numpygrad as npg
import numpygrad.nn as nn
from tests.configuration import check_equality


def _set_torch_weights(mha_t, mha_n, d_model):
    """Copy numpygrad MHA weights into a PyTorch MHA module."""
    W_q = mha_n.W_q.data
    W_k = mha_n.W_k.data
    W_v = mha_n.W_v.data
    W_o = mha_n.W_o.data
    with torch.no_grad():
        mha_t.in_proj_weight.copy_(torch.from_numpy(np.vstack([W_q, W_k, W_v])))
        mha_t.out_proj.weight.copy_(torch.from_numpy(W_o))
        if mha_n.b_q is not None:
            b_q = mha_n.b_q.data
            b_k = mha_n.b_k.data
            b_v = mha_n.b_v.data
            b_o = mha_n.b_o.data
            mha_t.in_proj_bias.copy_(torch.from_numpy(np.concatenate([b_q, b_k, b_v])))
            mha_t.out_proj.bias.copy_(torch.from_numpy(b_o))


def test_mha_parameter_count_with_bias():
    mha = nn.MultiHeadAttention(8, 2)
    assert len(list(mha.parameters())) == 8  # W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o


def test_mha_parameter_count_no_bias():
    mha = nn.MultiHeadAttention(8, 2, bias=False)
    assert len(list(mha.parameters())) == 4  # W_q, W_k, W_v, W_o


def test_mha_forward_matches_torch():
    np.random.seed(42)
    d_model, num_heads = 8, 2
    B, T = 2, 4

    mha_n = nn.MultiHeadAttention(d_model, num_heads)
    mha_t = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, dtype=torch.float64)
    _set_torch_weights(mha_t, mha_n, d_model)

    x = np.random.randn(B, T, d_model).astype(np.float64)
    q = np.random.randn(B, T, d_model).astype(np.float64)

    out_n = mha_n(npg.array(x), npg.array(q), npg.array(x))

    out_t, _ = mha_t(
        torch.from_numpy(x),
        torch.from_numpy(q),
        torch.from_numpy(x),
        need_weights=False,
    )
    check_equality(out_n.data, out_t.detach().numpy(), rtol=1e-5)


def test_mha_backward_grads_match():
    np.random.seed(7)
    d_model, num_heads = 8, 2
    B, T = 2, 4

    mha_n = nn.MultiHeadAttention(d_model, num_heads)
    mha_t = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, dtype=torch.float64)
    _set_torch_weights(mha_t, mha_n, d_model)

    x = np.random.randn(B, T, d_model).astype(np.float64)
    q_inp = np.random.randn(B, T, d_model).astype(np.float64)

    q_n = npg.array(x, requires_grad=True)
    out_n = mha_n(q_n, npg.array(q_inp), q_n)
    out_n.backward()

    q_t = torch.from_numpy(x).requires_grad_(True)
    out_t, _ = mha_t(
        q_t,
        torch.from_numpy(q_inp),
        q_t,
        need_weights=False,
    )
    out_t.sum().backward()

    check_equality(q_n.grad, q_t.grad.numpy(), rtol=1e-4)
    check_equality(
        mha_n.W_q.grad,
        mha_t.in_proj_weight.grad[:d_model].numpy(),
        rtol=1e-4,
    )
    check_equality(
        mha_n.b_q.grad,
        mha_t.in_proj_bias.grad[:d_model].numpy(),
        rtol=1e-4,
    )
