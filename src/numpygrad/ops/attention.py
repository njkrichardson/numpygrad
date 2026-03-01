import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.function import Context, Function
from numpygrad.ops.core import ensure_array


class MultiHeadAttentionOp(Function):
    @staticmethod
    def forward(
        ctx: Context,
        q: Array,
        k: Array,
        v: Array,
        W_q: Array,
        W_k: Array,
        W_v: Array,
        W_o: Array,
        b_q: Array | None,
        b_k: Array | None,
        b_v: Array | None,
        b_o: Array | None,
        num_heads: int,
        attn_mask: np.ndarray | None,
    ) -> Array:
        q, k, v = ensure_array(q), ensure_array(k), ensure_array(v)
        W_q, W_k, W_v, W_o = (
            ensure_array(W_q),
            ensure_array(W_k),
            ensure_array(W_v),
            ensure_array(W_o),
        )
        has_b_q = b_q is not None
        has_b_k = b_k is not None
        has_b_v = b_v is not None
        has_b_o = b_o is not None

        B, T, d_model = q.data.shape
        H = num_heads
        d_k = d_model // H
        scale = 1.0 / np.sqrt(d_k)

        # 1. Project
        Q = q.data @ W_q.data.T  # (B, T, d_model)
        K = k.data @ W_k.data.T
        V = v.data @ W_v.data.T
        if has_b_q:
            Q += ensure_array(b_q).data  # type: ignore[arg-type]
        if has_b_k:
            K += ensure_array(b_k).data  # type: ignore[arg-type]
        if has_b_v:
            V += ensure_array(b_v).data  # type: ignore[arg-type]

        # 2. Split heads → (B, H, T, d_k)
        Q_h = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K_h = K.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V_h = V.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        # 3. Scaled dot-product attention
        scores = Q_h @ K_h.transpose(0, 1, 3, 2) * scale  # (B, H, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp / exp.sum(axis=-1, keepdims=True)  # (B, H, T, T)
        attn_out = attn @ V_h  # (B, H, T, d_k)

        # 4. Merge heads → (B, T, d_model)
        merged = attn_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)

        # 5. Output projection
        out = merged @ W_o.data.T  # (B, T, d_model)
        if has_b_o:
            out = out + ensure_array(b_o).data  # type: ignore[arg-type]

        ctx.store(q, k, v, W_q, W_k, W_v, W_o)
        ctx._Q_h = Q_h  # type: ignore[attr-defined]
        ctx._K_h = K_h  # type: ignore[attr-defined]
        ctx._V_h = V_h  # type: ignore[attr-defined]
        ctx._attn = attn  # type: ignore[attr-defined]
        ctx._merged = merged  # type: ignore[attr-defined]
        ctx._B = B  # type: ignore[attr-defined]
        ctx._T = T  # type: ignore[attr-defined]
        ctx._H = H  # type: ignore[attr-defined]
        ctx._d_k = d_k  # type: ignore[attr-defined]
        ctx._d_model = d_model  # type: ignore[attr-defined]
        ctx._has_b_q = has_b_q  # type: ignore[attr-defined]
        ctx._has_b_k = has_b_k  # type: ignore[attr-defined]
        ctx._has_b_v = has_b_v  # type: ignore[attr-defined]
        ctx._has_b_o = has_b_o  # type: ignore[attr-defined]

        return Array(
            out,
            device=q.device,
            requires_grad=(
                q.requires_grad
                or k.requires_grad
                or v.requires_grad
                or W_q.requires_grad
                or W_k.requires_grad
                or W_v.requires_grad
                or W_o.requires_grad
            ),
        )

    @staticmethod
    def backward(ctx: Context, grad_out: np.ndarray):
        q, k, v, W_q, W_k, W_v, W_o = ctx.saved_arrays
        B, T, H, d_k, d_model = (
            ctx._B,  # type: ignore[attr-defined]
            ctx._T,  # type: ignore[attr-defined]
            ctx._H,  # type: ignore[attr-defined]
            ctx._d_k,  # type: ignore[attr-defined]
            ctx._d_model,  # type: ignore[attr-defined]
        )
        scale = 1.0 / np.sqrt(d_k)

        # Step 5 ← output projection
        grad_merged = grad_out @ W_o.data  # (B, T, d_model)
        grad_W_o = np.einsum("bti,btj->ij", grad_out, ctx._merged)  # type: ignore[attr-defined]
        grad_b_o = grad_out.sum(axis=(0, 1)) if ctx._has_b_o else None  # type: ignore[attr-defined]

        # Step 4 ← merge heads (permutation (0,2,1,3) is self-inverse)
        grad_attn_out = grad_merged.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # (B, H, T, d_k)

        # Step 3b ← attn_out = attn @ V_h
        grad_attn_raw = grad_attn_out @ ctx._V_h.transpose(0, 1, 3, 2)  # type: ignore[attr-defined]  # (B, H, T, T)
        grad_V_h = ctx._attn.transpose(0, 1, 3, 2) @ grad_attn_out  # type: ignore[attr-defined]  # (B, H, T, d_k)

        # Step 3a ← softmax
        dot = (ctx._attn * grad_attn_raw).sum(axis=-1, keepdims=True)  # type: ignore[attr-defined]  # (B, H, T, 1)
        grad_scores = ctx._attn * (grad_attn_raw - dot)  # type: ignore[attr-defined]  # (B, H, T, T)

        # Step 3c ← scores = Q_h @ K_h.T * scale
        g = grad_scores * scale
        grad_Q_h = g @ ctx._K_h  # type: ignore[attr-defined]  # (B, H, T, d_k)
        grad_K_h = g.transpose(0, 1, 3, 2) @ ctx._Q_h  # type: ignore[attr-defined]  # (B, H, T, d_k)

        # Step 2 ← un-split heads
        grad_Q_proj = grad_Q_h.transpose(0, 2, 1, 3).reshape(B, T, d_model)
        grad_K_proj = grad_K_h.transpose(0, 2, 1, 3).reshape(B, T, d_model)
        grad_V_proj = grad_V_h.transpose(0, 2, 1, 3).reshape(B, T, d_model)

        # Step 1 ← linear projections
        grad_q = grad_Q_proj @ W_q.data
        grad_W_q = np.einsum("bti,btj->ij", grad_Q_proj, q.data)
        grad_b_q = grad_Q_proj.sum(axis=(0, 1)) if ctx._has_b_q else None  # type: ignore[attr-defined]

        grad_k = grad_K_proj @ W_k.data
        grad_W_k = np.einsum("bti,btj->ij", grad_K_proj, k.data)
        grad_b_k = grad_K_proj.sum(axis=(0, 1)) if ctx._has_b_k else None  # type: ignore[attr-defined]

        grad_v = grad_V_proj @ W_v.data
        grad_W_v = np.einsum("bti,btj->ij", grad_V_proj, v.data)
        grad_b_v = grad_V_proj.sum(axis=(0, 1)) if ctx._has_b_v else None  # type: ignore[attr-defined]

        return (
            grad_q,
            grad_k,
            grad_v,
            grad_W_q,
            grad_W_k,
            grad_W_v,
            grad_W_o,
            grad_b_q,
            grad_b_k,
            grad_b_v,
            grad_b_o,
            None,  # num_heads: non-differentiable
            None,  # attn_mask: non-differentiable
        )
