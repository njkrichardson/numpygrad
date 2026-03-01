import argparse

import numpy as onp

import numpygrad as np
import numpygrad.nn as nn
from examples.classification_2d.data import PinwheelDataset
from examples.classification_2d.visuals import (
    plot_dataset_scatter,
    plot_final_decision,
    plot_initial_decision,
)
from numpygrad.utils.data import DataLoader

np.manual_seed(0)
Log = np.Log(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=3_000)
parser.add_argument("--step-size", type=float, default=0.01)
parser.add_argument("--report-every", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--num-classes", type=int, default=5)
parser.add_argument("--samples-per-class", type=int, default=512)
parser.add_argument("--num-estimate-loss-batches", type=int, default=32)
parser.add_argument("--plot", action="store_true")
parser.add_argument(
    "--initial-plot-only",
    action="store_true",
    help="Only generate initial decision plot, no training or final plot",
)


def main(args: argparse.Namespace):
    dataset = PinwheelDataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        radial_std=0.35,
        tangential_std=0.09,
        rate=0.4,
        snr_db=10.0,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    class Classifier(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: np.ndarray) -> np.ndarray:
            x = self.linear1(x)
            x = np.tanh(x)
            x = self.linear2(x)
            x = np.tanh(x)
            x = self.linear3(x)
            return x

    @np.no_grad()
    def estimate_loss(num_batches: int) -> float:
        losses = []
        for _ in range(num_batches):
            x, y = next(iter(dataloader))
            out = net(x)
            loss = nn.cross_entropy_loss(out, y)
            losses.append(loss.item())
        return np.mean(np.array(losses)).item()

    @np.no_grad()
    def estimate_accuracy(num_batches: int) -> float:
        correct = 0
        for _ in range(num_batches):
            x, y = next(iter(dataloader))
            out = net(x)
            correct += np.sum(np.argmax(out, axis=1) == y).item()
        return correct / (num_batches * args.batch_size)

    # expected loss guessing at random
    expected_loss = np.log(np.array(5))
    expected_accuracy = 1.0 / args.num_classes
    Log.info(f"Expected init loss: {expected_loss.item():0.2f}")
    Log.info(f"Expected init accuracy: {expected_accuracy * 100:.2f}%")

    net = Classifier(2, args.num_classes, args.hidden_dim)
    optimizer = np.optim.AdamW(net.parameters())
    Log.info(f"Actual init loss: {estimate_loss(args.num_estimate_loss_batches):0.2f}")
    Log.info(
        f"Actual init accuracy: {estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%"
    )

    @np.no_grad()
    def predict_proba(X):
        """X: (N, 2) numpy or numpygrad; returns (N, K) numpy probabilities."""
        X_np = X.numpy() if hasattr(X, "numpy") else onp.asarray(X)
        logits = net(np.array(X_np))
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(1, keepdims=True)).numpy()

    X_train = dataset.data

    plot_initial_decision(X_train, predict_proba)
    plot_dataset_scatter(dataset, X_train, dataset.targets)
    if args.initial_plot_only:
        return

    for step in range(args.num_steps):
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        out = net(x)
        loss = nn.cross_entropy_loss(out, y)
        loss.backward()
        optimizer.step()
        if step % args.report_every == 0:
            Log.info(
                f"Step {step}: loss={loss.item():0.2f} \
                accuracy={estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%"
            )

    Log.info(f"Final loss: {estimate_loss(args.num_estimate_loss_batches):0.2f}")
    Log.info(f"Final accuracy: {estimate_accuracy(args.num_estimate_loss_batches) * 100:0.2f}%")

    plot_final_decision(X_train, predict_proba)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
