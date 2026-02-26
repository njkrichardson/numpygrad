import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import numpygrad as npg
from numpygrad.nn import MLP

npg.manual_seed(0)
Log = npg.Log(__name__)


def mse(predictions: npg.array, targets: npg.array) -> npg.array:
    return ((predictions - targets) ** 2).mean()


def signal_power(x: np.ndarray) -> float:
    return np.mean(x**2).item()


def noise_power(signal: np.ndarray, snr_db: float) -> float:
    signal_pwr = signal_power(signal)
    snr_linear = 10 ** (snr_db / 10)
    return signal_pwr / snr_linear


def main():
    writer = SummaryWriter(log_dir=npg.configuration.TB_DIR)
    hidden_sizes = [32] * 8
    input_dim = 1
    output_dim = 1
    net = MLP(input_dim, hidden_sizes, output_dim)

    optimizer = npg.optim.SGD(net.parameters(), step_size=1e-1)

    num_examples: int = 2_048
    inputs = np.linspace(-2 * np.pi, 2 * np.pi, num_examples).reshape(-1, 1)
    inputs_norm = inputs / (2 * np.pi)
    snr_db = 10

    targets = np.sin(inputs)
    noise = np.random.normal(
        0, np.sqrt(noise_power(targets, snr_db)), size=inputs.shape
    )
    targets += noise

    def get_batch(batch_size: int):
        idx = np.random.choice(num_examples, batch_size, replace=False)
        return npg.array(inputs_norm[idx]), npg.array(targets[idx])

    batch_size: int = 64
    num_steps: int = 1_000
    report_every: int = 25

    for step in range(num_steps):
        x, y = get_batch(batch_size)
        optimizer.zero_grad()
        out = net(x)
        L = mse(out, y)
        L.backward()
        optimizer.step()
        if step % report_every == 0:
            loss = mse(net(npg.array(inputs_norm)), npg.array(targets)).data.item()
            print(f"Step {step}: loss={loss:.4f}")
            writer.add_scalar("Loss/train", loss, step)

    print("Final loss:", mse(net(npg.array(inputs_norm)), npg.array(targets)).data)
    writer.close()

    _x_raw = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    _x = npg.array(_x_raw / (2 * np.pi))
    plt.figure(figsize=(18, 10))
    plt.scatter(inputs, targets, c="tab:blue", alpha=0.5)
    plt.plot(_x_raw, net(_x).data, c="tab:orange", linewidth=3)
    plt.tight_layout()
    plt.savefig(npg.configuration.MEDIA_DIR / "mlp_fit.png")
    plt.close()


if __name__ == "__main__":
    main()
