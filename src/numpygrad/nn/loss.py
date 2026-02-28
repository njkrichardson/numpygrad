import numpy as np

import numpygrad as npg
from numpygrad.core.array import Array
from numpygrad.core.function import Context, Function
from numpygrad.ops.core import ArrayCoercible, ensure_array


class CrossEntropy(Function):
    @staticmethod
    def forward(
        ctx: Context, logits: ArrayCoercible, targets: ArrayCoercible, reduction: str
    ) -> Array:
        logits = ensure_array(logits)
        targets = ensure_array(targets)
        ctx.store(logits)
        ctx.targets = targets.data.astype(np.intp)
        ctx.reduction = reduction
        x = logits.data - logits.data.max(axis=-1, keepdims=True)
        lsm = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        ctx.log_probs = lsm
        N = logits.shape[0]
        nll = -lsm[np.arange(N), ctx.targets]
        out = np.mean(nll) if reduction == "mean" else np.sum(nll)
        return Array(out, device=logits.device, requires_grad=logits.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        assert ctx.log_probs is not None and ctx.targets is not None
        logits = ctx.saved_arrays[0]
        N = logits.shape[0]
        s = np.exp(ctx.log_probs)
        one_hot = np.zeros_like(s)
        one_hot[np.arange(N), ctx.targets] = 1.0
        scale = 1.0 / N if ctx.reduction == "mean" else 1.0
        return grad * (s - one_hot) * scale, None, None  # type: ignore[return-value]


def cross_entropy_loss(
    logits: npg.array,
    targets: npg.array,
    reduction: str = "mean",
) -> npg.array:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction}")
    return CrossEntropy.apply(logits, targets, reduction)


def mse(
    predictions: npg.array,
    targets: npg.array,
    reduction: str = "mean",
    weight: npg.array | None = None,
) -> npg.array:
    if weight is not None:
        predictions = predictions * weight
        targets = targets * weight
    if reduction == "mean":
        return ((predictions - targets) ** 2).mean()
    elif reduction == "sum":
        return ((predictions - targets) ** 2).sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
