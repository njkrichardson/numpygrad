from pathlib import Path

import numpygrad as np
from examples.utils.optional_mpl import HAS_MATPLOTLIB, plt

Log = np.Log(__name__)
MEDIA_DIR = Path(__file__).parent / "media"


def plot_fit(dataset, net, name: str = "regression_1d_fit"):
    if not HAS_MATPLOTLIB:
        return
    MEDIA_DIR.mkdir(exist_ok=True)
    save_path = MEDIA_DIR / f"{name}.png"
    plt.figure(figsize=(18, 10))
    plt.scatter(dataset.inputs_unnormalized, dataset.targets.data, c="tab:blue", alpha=0.5)
    plt.plot(dataset.inputs_unnormalized, net(dataset.data).data, c="tab:orange", linewidth=3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    Log.info(f"Plot saved to {save_path}")
