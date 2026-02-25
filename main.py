import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

import numpygrad as npg
import numpygrad.nn as nn

npg.manual_seed(0)
Log = npg.Log(__name__)


def mse(x: list[npg.ndarray] | npg.ndarray) -> npg.ndarray:
    if not isinstance(x, list):
        x = [x]

    out = npg.zeros(1)
    for _x in x:
        out += _x**2

    return out


def signal_power(signal: np.ndarray) -> float:
    return np.mean(signal**2).item()


def snr_to_noise_power(snr_db: float, signal: np.ndarray) -> float:
    return signal_power(signal) / (10 ** (snr_db / 10))


def main():
    num_inputs = 1
    num_outputs = 1

    num_examples: int = 256
    batch_size: int = 8
    inputs = np.linspace(-2 * np.pi, 2 * np.pi, num=num_examples)
    targets = np.sin(inputs)

    snr_db = 10
    noise = npr.randn(num_examples) * np.sqrt(snr_to_noise_power(snr_db, targets))
    targets += noise

    def get_batch() -> tuple[list[npg.ndarray], list[npg.ndarray]]:
        batch_inputs: list[npg.ndarray] = []
        batch_targets: list[npg.ndarray] = []
        for _ in range(batch_size):
            idx = npr.choice(num_examples)
            batch_inputs.append(npg.array(inputs[idx]))
            batch_targets.append(npg.array(targets[idx]))

        return batch_inputs, batch_targets

    num_steps: int = 800
    net = nn.MLP(num_inputs, [16], num_outputs)

    optimizer = npg.optim.SGD(net.parameters())
    print_every: int = 10

    def estimate_loss(num_batches: int = 200) -> float:
        total = 0.0

        for _ in range(num_batches):
            X, Y = get_batch()
            for x, y in zip(X, Y):
                out = net(x)
                error = out - y  # type: ignore
                L = mse(error)
                total += L.data.item()

        average = total / (num_batches * batch_size)
        optimizer.zero_grad()
        return average

    init_loss = estimate_loss()
    Log.info(f"Init: {init_loss:.5f}")

    for i in range(num_steps):
        X, Y = get_batch()
        optimizer.zero_grad()
        for x, y in zip(X, Y):
            out = net(x)
            assert isinstance(out, npg.ndarray)
            error = out - y
            L = mse(error)
            L.backward()

        optimizer.step()

        if (i % print_every) == 0:
            Log.info(f"Iteration [{i:03d}/{num_steps:03d}]: {estimate_loss():.5f}")

    Log.info(f"Final: {estimate_loss():.5f}")

    net_outs = []
    for x in inputs:
        net_outs.append(net(npg.array(x)).data.item())

    net_outs = np.array(net_outs)

    plt.figure(figsize=(16, 12))
    plt.scatter(inputs, targets, c="tab:blue", s=5)
    plt.plot(inputs, net_outs, c="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(npg.configuration.MEDIA_DIR / "fit")
    plt.close()


if __name__ == "__main__":
    main()
