Utilities (``utils``)
=====================

Data loading
------------

``utils.Dataset``
~~~~~~~~~~~~~~~~~

Abstract base class for datasets. Subclass it to wrap custom data sources::

    from numpygrad.utils import Dataset
    import numpygrad as npg

    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, index):
            return npg.array(self.X[index]), npg.array(self.y[index])

``utils.TensorDataset``
~~~~~~~~~~~~~~~~~~~~~~~~

Wraps two arrays (data and targets) as a dataset. Internally stores them as
``Array`` objects and uses slice-based batching for efficiency::

    from numpygrad.utils import TensorDataset
    import numpy as np

    X = np.random.randn(500, 8).astype(np.float32)
    y = np.random.randint(0, 4, size=(500,))

    dataset = TensorDataset(X, y)
    print(len(dataset))         # 500
    x_i, y_i = dataset[0]      # single sample

``utils.DataLoader``
~~~~~~~~~~~~~~~~~~~~

Iterates over a dataset in batches::

    from numpygrad.utils import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

    for x_batch, y_batch in loader:
        ...  # x_batch: Array (32, 8), y_batch: Array (32,)

Parameters:

- ``batch_size`` — number of samples per batch
- ``shuffle`` — whether to shuffle at the start of each epoch (default ``True``)
- ``drop_last`` — discard the final batch if it is smaller than ``batch_size``
  (default ``False``)

For ``TensorDataset``, batching is done with a single slice operation. For
custom ``Dataset`` subclasses, samples are fetched one at a time and stacked.

Visualisation
-------------

``utils.draw_computation_graph(root, save_path=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Renders the computation graph rooted at ``root`` as a PNG using Graphviz::

    from numpygrad.utils import draw_computation_graph
    from pathlib import Path

    x = npg.array([1.0, 2.0], requires_grad=True)
    y = npg.array([3.0, 4.0], requires_grad=True)
    z = (x * y).sum()

    draw_computation_graph(z, save_path=Path("graph.png"))

Requires the optional ``graphviz`` package::

    pip install graphviz

If ``save_path`` is ``None``, the graph is displayed inline (in a Jupyter
notebook) or saved to a temporary file.

Seeding
-------

For ``npg.manual_seed`` and the full ``npg.random`` namespace see :doc:`random`.
