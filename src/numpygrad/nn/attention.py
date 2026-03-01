import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.array_creation import zeros
from numpygrad.nn.module import Module, Parameter
from numpygrad.ops.attention import MultiHeadAttentionOp


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        std = np.sqrt(2 / d_model)
        for name in ("W_q", "W_k", "W_v", "W_o"):
            setattr(
                self,
                name,
                Parameter(Array(np.random.randn(d_model, d_model) * std, requires_grad=True)),
            )
        if bias:
            for name in ("b_q", "b_k", "b_v", "b_o"):
                setattr(self, name, Parameter(zeros(d_model, requires_grad=True)))
        else:
            for name in ("b_q", "b_k", "b_v", "b_o"):
                object.__setattr__(self, name, None)

    def forward(  # type: ignore[override]
        self,
        q: Array,
        k: Array,
        v: Array,
        attn_mask: np.ndarray | None = None,
    ) -> Array:
        return MultiHeadAttentionOp.apply(
            q,
            k,
            v,
            self.W_q,
            self.W_k,
            self.W_v,
            self.W_o,
            self.b_q,
            self.b_k,
            self.b_v,
            self.b_o,
            self.num_heads,
            attn_mask,
        )

    def __call__(  # type: ignore[override]
        self,
        q: Array,
        k: Array,
        v: Array,
        attn_mask: np.ndarray | None = None,
    ) -> Array:
        return self.forward(q, k, v, attn_mask)

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_model={self.d_model}, "
            f"num_heads={self.num_heads}, bias={self.b_q is not None})"
        )
