import numpy as np

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
        ctx.logits_shape = logits.shape
        ctx.reduction = reduction
        num_classes = logits.shape[-1]
        lgt = logits.data.reshape(-1, num_classes)
        tgt = targets.data.reshape(-1).astype(np.intp)
        ctx.targets = tgt
        x = lgt - lgt.max(axis=-1, keepdims=True)
        lsm = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        ctx.log_probs = lsm
        N = lgt.shape[0]
        nll = -lsm[np.arange(N), tgt]
        out = np.mean(nll) if reduction == "mean" else np.sum(nll)
        return Array(out, device=logits.device, requires_grad=logits.requires_grad)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray, ...]:
        assert ctx.log_probs is not None and ctx.targets is not None
        N = ctx.log_probs.shape[0]
        s = np.exp(ctx.log_probs)
        one_hot = np.zeros_like(s)
        one_hot[np.arange(N), ctx.targets] = 1.0
        scale = 1.0 / N if ctx.reduction == "mean" else 1.0
        grad_2d = grad * (s - one_hot) * scale
        return grad_2d.reshape(ctx.logits_shape), None, None  # type: ignore[return-value]


def cross_entropy_loss(
    logits: Array,
    targets: Array,
    reduction: str = "mean",
) -> Array:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction}")
    return CrossEntropy.apply(logits, targets, reduction)


def mse(
    predictions: Array,
    targets: Array,
    reduction: str = "mean",
    weight: Array | None = None,
) -> Array:
    if weight is not None:
        predictions = predictions * weight
        targets = targets * weight
    if reduction == "mean":
        return ((predictions - targets) ** 2).mean()
    elif reduction == "sum":
        return ((predictions - targets) ** 2).sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
