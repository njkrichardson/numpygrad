import numpy as np
import numpy.random as npr

from numpygrad.utils.data import TensorDataset


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    inputs = 10 * np.einsum("ti,tij->tj", features, rotations)
    perm = npr.permutation(len(inputs))
    return inputs[perm], labels[perm]


class PinwheelDataset(TensorDataset):
    def __init__(
        self,
        num_classes: int,
        samples_per_class: int,
        radial_std: float = 0.3,
        tangential_std: float = 0.05,
        rate: float = 0.025,
        snr_db: float = 20.0,
    ):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.radial_std = radial_std
        self.tangential_std = tangential_std
        self.rate = rate
        self.snr_db = snr_db

        inputs, targets = make_pinwheel_data(
            radial_std, tangential_std, num_classes, samples_per_class, rate
        )

        super().__init__(inputs, targets)
