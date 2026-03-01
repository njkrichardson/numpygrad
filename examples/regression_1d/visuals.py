from pathlib import Path

import matplotlib.pyplot as plt

import numpygrad as np

Log = np.Log(__name__)
MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)


def plot_fit(dataset, net, name: str = "regression_1d_fit"):
    save_path = MEDIA_DIR / f"{name}.png"
    plt.figure(figsize=(18, 10))
    plt.scatter(dataset.inputs_unnormalized, dataset.targets.data, c="tab:blue", alpha=0.5)
    plt.plot(dataset.inputs_unnormalized, net(dataset.data).data, c="tab:orange", linewidth=3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    Log.info(f"Plot saved to {save_path}")
