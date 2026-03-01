import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpygrad as np
import numpygrad.nn as nn
from examples.mnist.data import MNIST
from numpygrad.utils.data import DataLoader

MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)
Log = np.Log(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=5_00)
parser.add_argument("--step-size", type=float, default=1e-3)
parser.add_argument("--report-every", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--hidden-dim", type=int, default=32)
parser.add_argument("--num-estimate-loss-batches", type=int, default=32)
parser.add_argument("--plot", action="store_true")


def main(args: argparse.Namespace):
    train_dataset = MNIST(split="train")
    test_dataset = MNIST(split="test")
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    class MNISTClassifier(nn.Module):
        def __init__(self, input_shape: tuple[int, int, int], num_classes: int, hidden_dim: int):
            super().__init__()
            self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.linear_out = nn.Linear(hidden_dim * (input_shape[1] * input_shape[2]), num_classes)

        def forward(self, x: np.ndarray) -> np.ndarray:
            x = self.conv1(x)
            x = np.relu(x)
            x = self.conv2(x)
            x = np.relu(x)
            x = x.reshape(x.shape[0], -1)
            x = self.linear_out(x)
            return x

    net = MNISTClassifier((1, 28, 28), 10, args.hidden_dim)
    optimizer = np.optim.AdamW(net.parameters(), lr=args.step_size)

    @np.no_grad()
    def estimate_loss(num_batches: int, loader=None) -> float:
        if loader is None:
            loader = dataloader
        losses = []
        for _ in range(num_batches):
            x, y = next(iter(loader))
            out = net(x)
            loss = nn.cross_entropy_loss(out, y)
            losses.append(loss.item())
        return np.mean(np.array(losses)).item()

    @np.no_grad()
    def estimate_accuracy(num_batches: int, loader=None) -> float:
        if loader is None:
            loader = dataloader
        correct = 0
        total = 0
        for _ in range(num_batches):
            x, y = next(iter(loader))
            out = net(x)
            n = y.shape[0]
            correct += np.sum(np.argmax(out, axis=1) == y).item()
            total += n
        return correct / total

    Log.info(f"Init loss: {estimate_loss(args.num_estimate_loss_batches):0.2f}")
    Log.info(f"Init accuracy: {estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%")

    for step in range(args.num_steps):
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        out = net(x)
        loss = nn.cross_entropy_loss(out, y)
        loss.backward()
        optimizer.step()
        if step % args.report_every == 0:
            Log.info(
                f"Step {step}: loss={loss.item():0.2f} "
                "accuracy={estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%"
            )

    Log.info(f"Final train loss: {estimate_loss(args.num_estimate_loss_batches):0.2f}")
    Log.info(
        f"Final test loss: {estimate_loss(args.num_estimate_loss_batches, test_dataloader):0.2f}"
    )
    Log.info(
        f"Final train accuracy: {estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%"
    )
    Log.info(
        f"Final test accuracy: "
        f"{estimate_accuracy(args.num_estimate_loss_batches, test_dataloader) * 100:0.2f}%"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
