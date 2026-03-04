import numpy as np

from numpygrad.core.array import Array


def _calculate_fan(tensor: Array) -> tuple[int, int]:
    """Returns (fan_in, fan_out). Supports 2D (Linear) and 4D (Conv2d)."""
    if tensor.data.ndim == 2:
        return tensor.shape[0], tensor.shape[1]
    elif tensor.data.ndim == 4:
        receptive = tensor.shape[2] * tensor.shape[3]
        return tensor.shape[1] * receptive, tensor.shape[0] * receptive
    raise ValueError(f"fan requires 2D or 4D tensor, got ndim={tensor.data.ndim}")


_GAINS: dict[str, float] = {
    "relu": float(np.sqrt(2.0)),
    "leaky_relu": float(np.sqrt(2.0 / (1 + 0.01**2))),
    "tanh": 5.0 / 3.0,
    "sigmoid": 1.0,
    "gelu": float(np.sqrt(2.0)),
    "linear": 1.0,
    "identity": 1.0,
}


def uniform_(tensor: Array, low: float = -1.0, high: float = 1.0) -> Array:
    tensor.data[:] = np.random.uniform(low, high, tensor.shape)
    return tensor


def normal_(tensor: Array, mean: float = 0.0, std: float = 1.0) -> Array:
    tensor.data[:] = np.random.normal(mean, std, tensor.shape)
    return tensor


def zeros_(tensor: Array) -> Array:
    tensor.data[:] = 0.0
    return tensor


def ones_(tensor: Array) -> Array:
    tensor.data[:] = 1.0
    return tensor


def kaiming_uniform_(tensor: Array, mode: str = "fan_in", nonlinearity: str = "relu") -> Array:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _GAINS.get(nonlinearity, 1.0)
    bound = np.sqrt(3.0) * gain / np.sqrt(fan)
    tensor.data[:] = np.random.uniform(-bound, bound, tensor.shape)
    return tensor


def kaiming_normal_(tensor: Array, mode: str = "fan_in", nonlinearity: str = "relu") -> Array:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = _GAINS.get(nonlinearity, 1.0)
    std = gain / np.sqrt(fan)
    tensor.data[:] = np.random.normal(0.0, std, tensor.shape)
    return tensor


def xavier_uniform_(tensor: Array, gain: float = 1.0) -> Array:
    fan_in, fan_out = _calculate_fan(tensor)
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data[:] = np.random.uniform(-bound, bound, tensor.shape)
    return tensor


def xavier_normal_(tensor: Array, gain: float = 1.0) -> Array:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data[:] = np.random.normal(0.0, std, tensor.shape)
    return tensor
