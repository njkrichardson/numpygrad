import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch 
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

class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int):
        super().__init__()
        size = [input_dim] + hidden_sizes + [output_dim]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(size[:-1], size[1:])])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        return self.layers[-1](x)

def main():
    use_torch = False
    writer = SummaryWriter(log_dir=npg.configuration.TB_DIR)
    hidden_sizes = [8] * 8
    input_dim = 1
    output_dim = 1

    if use_torch:
        torch_net = TorchMLP(input_dim, hidden_sizes, output_dim)
        torch_optimizer = torch.optim.SGD(torch_net.parameters(), lr=1e-1)
    else:
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

    batch_size: int = 128
    num_steps: int = 10_000
    report_every: int = 100

    for step in range(num_steps):
        x, y = get_batch(batch_size)
        if use_torch:
            torch_optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        if use_torch:
            out = torch_net(torch.from_numpy(x.data).to(torch.float32))
        else:
            out = net(x)

        if use_torch:
            L = torch.nn.functional.mse_loss(out, torch.from_numpy(y.data).to(torch.float32))
        else:
            L = mse(out, y)

        L.backward()

        if use_torch:
            torch_optimizer.step()
        else:
            optimizer.step()

        if step % report_every == 0:
            if use_torch:
                loss = mse(torch_net(torch.from_numpy(inputs_norm).to(torch.float32)), torch.from_numpy(targets).to(torch.float32)).item()
            else:
                loss = mse(net(npg.array(inputs_norm)), npg.array(targets)).data.item()
            print(f"Step {step}: loss={loss:.4f}")
            writer.add_scalar("Loss/train", loss, step)

    if use_torch:
        print("Final loss:", mse(torch_net(torch.from_numpy(inputs_norm).to(torch.float32)), torch.from_numpy(targets).to(torch.float32)).item())
    else:
        print("Final loss:", mse(net(npg.array(inputs_norm)), npg.array(targets)).data)
    writer.close()

    _x_raw = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    _x = npg.array(_x_raw / (2 * np.pi))
    plt.figure(figsize=(18, 10))
    plt.scatter(inputs, targets, c="tab:blue", alpha=0.5)
    if use_torch:
        plt.plot(_x_raw, torch_net(torch.from_numpy(_x_raw / (2*np.pi)).to(torch.float32)).detach().numpy(), c="tab:orange", linewidth=3)
    else:
        plt.plot(_x_raw, net(_x).data, c="tab:orange", linewidth=3)
    plt.tight_layout()
    if use_torch:
        plt.savefig(npg.configuration.MEDIA_DIR / "mlp_fit_torch.png")
    else:
        plt.savefig(npg.configuration.MEDIA_DIR / "mlp_fit_numpygrad.png")
    plt.close()


if __name__ == "__main__":
    main()
