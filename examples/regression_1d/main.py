import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import numpygrad as npg
import numpygrad.nn as nn 
from numpygrad.utils.data import DataLoader
from examples.regression_1d.data import RegressionDataset

npg.manual_seed(0)
Log = npg.Log(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--create-plot", action="store_true")
parser.add_argument("--num-steps", type=int, default=10_000)
parser.add_argument("--report-every", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[8] * 8)
parser.add_argument("--input-dim", type=int, default=1)
parser.add_argument("--output-dim", type=int, default=1)
parser.add_argument("--snr-db", type=float, default=10)
parser.add_argument("--num-examples", type=int, default=2_048)
parser.add_argument("--num-estimate-loss-batches", type=int, default=32)

def main(args: argparse.Namespace):
    hidden_sizes = args.hidden_sizes
    input_dim = args.input_dim
    output_dim = args.output_dim

    net = nn.MLP(input_dim, hidden_sizes, output_dim)
    optimizer = npg.optim.SGD(net.parameters(), step_size=1e-1)

    dataset = RegressionDataset(args.num_examples, args.snr_db)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def estimate_loss(num_batches: int):
        losses = []
        for _ in range(num_batches):
            x, y = next(iter(dataloader))
            out = net(x)
            L = nn.mse(out, y)
            losses.append(L.data.item())
        optimizer.zero_grad()
        return np.mean(losses)

    for step in range(args.num_steps):
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        out = net(x)
        L = nn.mse(out, y)
        L.backward()
        optimizer.step()

        if step % args.report_every == 0:
            loss = estimate_loss(args.num_estimate_loss_batches)
            Log.info(f"Step {step}: loss={loss:.4f}")

    Log.info(f"Final loss: {estimate_loss(args.num_estimate_loss_batches):.4f}")

    if args.create_plot:
        save_path = npg.configuration.MEDIA_DIR / "mlp_fit_numpygrad.png"
        plt.figure(figsize=(18, 10))
        plt.scatter(dataset.inputs_unnormalized, dataset.targets.data, c="tab:blue", alpha=0.5)
        plt.plot(dataset.inputs_unnormalized, net(dataset.data).data, c="tab:orange", linewidth=3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        Log.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
