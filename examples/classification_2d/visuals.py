from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import numpygrad as np

Log = np.Log(__name__)
EXPERIMENT_DIR = Path(__file__).parent
MEDIA_DIR = EXPERIMENT_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)


# Same color palette as original (plain numpy for plotting)
COLORS = (
    onp.array(
        [
            [106, 61, 154],
            [31, 120, 180],
            [51, 160, 44],
            [227, 26, 28],
            [255, 127, 0],
            [166, 206, 227],
            [178, 223, 138],
            [251, 154, 153],
            [253, 191, 111],
            [202, 178, 214],
        ]
    )
    / 256.0
)


def get_hexbin_coords(ax, xlims, ylims, gridsize):
    x_corners = [xlims[0], xlims[0], xlims[1], xlims[1]]
    y_corners = [ylims[0], ylims[1], ylims[0], ylims[1]]
    hb = ax.hexbin(x_corners, y_corners, gridsize=gridsize, extent=(*xlims, *ylims))
    coords = onp.asarray(hb.get_offsets())
    hb.remove()
    return coords


def plot_classifier_hexbin(ax, predict_proba, xlims, ylims, gridsize=75):
    """
    predict_proba: callable (N, 2) -> (N, K) softmax probabilities (numpy in/out).
    Colors each hex cell by argmax class, alpha by confidence (max prob).
    """
    coords = get_hexbin_coords(ax, xlims, ylims, gridsize)  # numpy
    probs = predict_proba(coords)  # (N, K) numpy
    probs = onp.asarray(probs)
    class_ids = onp.argmax(probs, axis=1)
    confidence = onp.max(probs, axis=1)

    for k in range(probs.shape[1]):
        mask = class_ids == k
        if not mask.any():
            continue
        color = COLORS[k % len(COLORS)]
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        cdict = {
            "red": [(0.0, r, r), (1.0, r, r)],
            "green": [(0.0, g, g), (1.0, g, g)],
            "blue": [(0.0, b, b), (1.0, b, b)],
            "alpha": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        }
        cmap = LinearSegmentedColormap(f"cls_{k}", cdict)
        x, y = coords[mask].T
        ax.hexbin(
            x,
            y,
            C=confidence[mask],
            cmap=cmap,
            gridsize=gridsize,
            extent=(*xlims, *ylims),
            linewidths=0.0,
            edgecolors="none",
            vmin=0.0,
            vmax=1.0,
            zorder=1,
        )


def plot_decision(ax, X, predict_proba, gridsize=75, border=0.15):
    X_np = X.numpy() if hasattr(X, "numpy") else onp.asarray(X)
    xlims = (X_np[:, 0].min() - border, X_np[:, 0].max() + border)
    ylims = (X_np[:, 1].min() - border, X_np[:, 1].max() + border)

    plot_classifier_hexbin(ax, predict_proba, xlims, ylims, gridsize)
    ax.scatter(X_np[:, 0], X_np[:, 1], c="gray", s=6, alpha=0.4, zorder=2, linewidths=0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


def plot_initial_decision(X_train, predict_proba):
    save_path = MEDIA_DIR / "classification_2d_initial.png"
    _, ax = plt.subplots(figsize=(5, 5))
    plot_decision(ax, X_train, predict_proba)
    plt.title("Initial classifier decision plot")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)
    plt.close()
    Log.info(f"Plot saved to {str(save_path)}")


def plot_dataset_scatter(dataset, X_train, y_train):
    # Dark palette (RGB 0-255 → 0-1)
    colors = [
        [106 / 255, 61 / 255, 154 / 255],
        [31 / 255, 120 / 255, 180 / 255],
        [51 / 255, 160 / 255, 44 / 255],
        [227 / 255, 26 / 255, 28 / 255],
        [255 / 255, 127 / 255, 0 / 255],
    ]
    cmap = ListedColormap(colors)

    plt.figure(figsize=(12, 12))
    plt.scatter(
        dataset.data[:, 0].numpy(),
        dataset.data[:, 1].numpy(),
        c=dataset.targets.numpy(),
        cmap=cmap,
    )
    plt.title("Dataset scatter plot")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    save_path = MEDIA_DIR / "classification_2d.png"
    plt.savefig(save_path)
    plt.close()
    Log.info(f"Plot saved to {str(save_path)}")


def plot_final_decision(X_train, predict_proba):
    save_path_final = MEDIA_DIR / "classification_2d_final.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_decision(ax, X_train, predict_proba)
    plt.title("Final classifier decision plot")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path_final)
    plt.close()
    Log.info(f"Final classifier plot saved to {str(save_path_final)}")
