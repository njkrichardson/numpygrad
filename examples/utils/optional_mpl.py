"""
Optional matplotlib for examples. When matplotlib is not installed, examples
still run; plot functions become no-ops and no figures are emitted.
"""

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]
    LinearSegmentedColormap = None  # type: ignore[assignment, misc]
    ListedColormap = None  # type: ignore[assignment, misc]
