Custom Modules
==============

All trainable components in numpygrad inherit from ``nn.Module``. Subclassing
it is the standard way to define new layers, loss functions, or complete
architectures.

Minimal example
---------------

Override ``forward`` with the computation your module performs::

    import numpygrad as npg
    import numpygrad.nn as nn

    class Affine(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(npg.random.randn((in_features, out_features)))
            self.bias   = nn.Parameter(npg.zeros((out_features,)))

        def forward(self, x: npg.array) -> npg.array:
            return x @ self.weight + self.bias

    layer = Affine(4, 8)
    out = layer(npg.random.randn((2, 4)))  # shape (2, 8)

Any attribute assigned as a ``Parameter`` is automatically included in
``module.parameters()`` and therefore in the optimizer's update step.

Composing modules
-----------------

Assign child modules as attributes and they are tracked recursively::

    class TwoLayer(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.fc1 = Affine(dim, dim)
            self.fc2 = Affine(dim, dim)

        def forward(self, x: npg.array) -> npg.array:
            return self.fc2(npg.relu(self.fc1(x)))

    net = TwoLayer(16)
    print(len(list(net.parameters())))  # 4 — weight + bias for each layer

``parameters()`` walks the full module tree recursively, so you can nest
modules arbitrarily deep.

Using ``Sequential``
--------------------

For a simple chain of modules, ``nn.Sequential`` avoids boilerplate::

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    out = model(x)   # applies each module in order

Buffers
-------

If you need a non-trainable array stored on the module (e.g. a running mean),
assign it as a plain ``Array`` — it will not appear in ``parameters()`` but
is still accessible as an attribute::

    class BatchNorm1d(nn.Module):
        def __init__(self, num_features: int) -> None:
            super().__init__()
            self.scale  = nn.Parameter(npg.ones((num_features,)))
            self.shift  = nn.Parameter(npg.zeros((num_features,)))
            self.running_mean = npg.zeros((num_features,))  # not a Parameter

        def forward(self, x: npg.array) -> npg.array:
            mean = x.mean(axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            x_norm = (x - mean) / (x.var(axis=0) ** 0.5 + 1e-5)
            return self.scale * x_norm + self.shift

Inspecting parameters
---------------------

``state_dict()`` returns a flat ``dict`` mapping parameter names to their
underlying NumPy arrays — useful for checkpointing::

    sd = model.state_dict()
    # {'fc1.weight': array(...), 'fc1.bias': array(...), ...}

    import numpy as np
    np.savez("checkpoint.npz", **sd)
