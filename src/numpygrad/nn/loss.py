import numpygrad as npg


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
