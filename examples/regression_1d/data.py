import numpy as np

from examples.utils.signals import noise_power
from numpygrad.utils.data import TensorDataset


class RegressionDataset(TensorDataset):
    def __init__(self, num_examples: int, snr_db: float):
        inputs = np.linspace(-2 * np.pi, 2 * np.pi, num_examples).reshape(-1, 1)
        inputs_norm = inputs / (2 * np.pi)
        targets = np.sin(inputs)
        noise = np.random.normal(0, np.sqrt(noise_power(targets, snr_db)), size=inputs.shape)
        targets = targets + noise
        super().__init__(inputs_norm, targets)
        # TODO deprecate normalization
        self.inputs_unnormalized = inputs
