from pathlib import Path

import numpy as onp

import numpygrad as np
from examples.utils.optional_mpl import (
    HAS_MATPLOTLIB,
    LinearSegmentedColormap,
    ListedColormap,
    plt,
)

Log = np.Log(__name__)
EXPERIMENT_DIR = Path(__file__).parent
MEDIA_DIR = EXPERIMENT_DIR / "media"


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


def start_aggregate(X_train, y_train, predict_proba, partial=False):
    if not HAS_MATPLOTLIB:
        return None
    MEDIA_DIR.mkdir(exist_ok=True)
    n = 2 if partial else 3
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    cmap = ListedColormap(COLORS)
    X_np = X_train.numpy() if hasattr(X_train, "numpy") else onp.asarray(X_train)
    axes[0].scatter(X_np[:, 0], X_np[:, 1], c=y_train, cmap=cmap, marker="o", s=5)
    plot_decision(axes[1], X_train, predict_proba)
    plt.subplots_adjust(wspace=0.04)
    if partial:
        save_path = MEDIA_DIR / "classification_2d.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        Log.info(f"Plot saved to {str(save_path)}")
        return None
    return fig


def finish_aggregate(fig, X_train, predict_proba):
    if fig is None:
        return
    plot_decision(fig.axes[2], X_train, predict_proba)
    plt.tight_layout()
    save_path = MEDIA_DIR / "classification_2d.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    Log.info(f"Plot saved to {str(save_path)}")
