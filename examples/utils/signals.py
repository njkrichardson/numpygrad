import numpy as np


def signal_power(x: np.ndarray) -> float:
    return np.mean(x**2).item()


def noise_power(signal: np.ndarray, snr_db: float) -> float:
    signal_pwr = signal_power(signal)
    snr_linear = 10 ** (snr_db / 10)
    return signal_pwr / snr_linear
