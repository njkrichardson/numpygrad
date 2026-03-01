import argparse

import numpygrad as np
import numpygrad.nn as nn
from examples.regression_1d.data import RegressionDataset
from examples.regression_1d.visuals import plot_fit
from numpygrad.utils.data import DataLoader, TensorDataset

np.manual_seed(0)
Log = np.Log(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=3000)
parser.add_argument("--report-every", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64] * 2)
parser.add_argument("--input-dim", type=int, default=1)
parser.add_argument("--output-dim", type=int, default=1)
parser.add_argument("--snr-db", type=float, default=12)
parser.add_argument("--num-examples", type=int, default=4096)
parser.add_argument("--num-estimate-loss-batches", type=int, default=32)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")
parser.add_argument("--plot-init-only", action="store_true")
parser.add_argument(
    "--test-split",
    type=float,
    default=0.2,
    help="Fraction of examples to hold out as the test set (0..1)",
)
parser.add_argument("--activation", type=str, choices=["relu", "tanh", "sigmoid"], default="tanh")


def main(args: argparse.Namespace):
    hidden_sizes = args.hidden_sizes
    input_dim = args.input_dim
    output_dim = args.output_dim

    net = nn.MLP(input_dim, hidden_sizes, output_dim, activation=args.activation)
    if args.optimizer == "sgd":
        optimizer = np.optim.SGD(net.parameters(), step_size=1e-1)
    elif args.optimizer == "adamw":
        optimizer = np.optim.AdamW(net.parameters())
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    dataset = RegressionDataset(args.num_examples, args.snr_db)
    n = len(dataset)
    n_test = int(n * args.test_split)
    n_train = n - n_test
    train_dataset = TensorDataset(dataset.data.data[:n_train], dataset.targets.data[:n_train])
    test_dataset = TensorDataset(dataset.data.data[n_train:], dataset.targets.data[n_train:])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    @np.no_grad()
    def estimate_loss(num_batches: int, loader=None):
        if loader is None:
            loader = train_dataloader
        losses = []
        for _ in range(num_batches):
            x, y = next(iter(loader))
            out = net(x)
            L = nn.mse(out, y)
            losses.append(L.data.item())
        optimizer.zero_grad()
        return np.mean(np.array(losses)).item()

    if args.plot_init_only:
        plot_fit(dataset, net, name="regression_1d_init")
        return

    for step in range(args.num_steps):
        x, y = next(iter(train_dataloader))
        optimizer.zero_grad()
        out = net(x)
        L = nn.mse(out, y)
        L.backward()
        optimizer.step()

        if step % args.report_every == 0:
            loss = estimate_loss(args.num_estimate_loss_batches)
            Log.info(f"Step {step}: loss={loss:.4f}")

    Log.info(f"Final train loss: {estimate_loss(args.num_estimate_loss_batches):.4f}")
    Log.info(
        f"Final test loss: {estimate_loss(args.num_estimate_loss_batches, test_dataloader):.4f}"
    )

    plot_fit(dataset, net)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
