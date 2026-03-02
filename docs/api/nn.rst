Neural Network (``nn``)
=======================

The ``numpygrad.nn`` module provides layers, activation modules, and loss
functions. Import it as::

    import numpygrad.nn as nn

Module system
-------------

``nn.Module``
~~~~~~~~~~~~~

Base class for all layers and models. Subclass it and override ``forward``::

    class MyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(npg.randn((4, 4)))

        def forward(self, x):
            return x @ self.weight

Key methods:

- ``module(x)`` — calls ``forward(x)`` (via ``__call__``)
- ``module.parameters()`` — iterator over all ``Parameter`` objects in the
  module and its children, recursively
- ``module.state_dict()`` — ``dict`` mapping parameter names to NumPy arrays

``nn.Parameter``
~~~~~~~~~~~~~~~~

A subclass of ``Array`` that always has ``requires_grad=True``. Assigning a
``Parameter`` as a module attribute automatically registers it with
``parameters()``::

    self.bias = nn.Parameter(npg.zeros((8,)))

``nn.Sequential``
~~~~~~~~~~~~~~~~~

Chains modules in order::

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    out = model(x)

Layers
------

``nn.Linear(num_inputs, num_outputs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fully connected layer: ``y = x @ W.T + b``.

- ``weight``: ``Parameter`` of shape ``(num_outputs, num_inputs)``
- ``bias``: ``Parameter`` of shape ``(num_outputs,)``

::

    layer = nn.Linear(8, 4)
    out = layer(npg.randn((16, 8)))   # (16, 4)

``nn.MLP(input_dim, hidden_sizes, output_dim, activation="relu")``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-layer perceptron: stacked ``Linear`` layers separated by the chosen
activation. ``activation`` can be ``"relu"``, ``"tanh"``, or ``"sigmoid"``::

    model = nn.MLP(
        input_dim=784,
        hidden_sizes=[256, 128],
        output_dim=10,
        activation="relu",
    )

``nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2D convolutional layer. ``kernel_size``, ``stride``, and ``padding`` each
accept an int or a ``(H, W)`` tuple::

    conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    out = conv(npg.randn((8, 3, 28, 28)))   # (8, 32, 28, 28)

``nn.MultiHeadAttention(d_model, num_heads, bias=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-head scaled dot-product attention. ``d_model`` must be divisible by
``num_heads``::

    attn = nn.MultiHeadAttention(d_model=64, num_heads=8)
    out = attn(q, k, v)                 # q, k, v: (batch, seq, d_model)
    out = attn(q, k, v, attn_mask=mask) # optional additive mask

Activation modules
------------------

These wrap the functional activations as ``Module`` subclasses, useful inside
``Sequential``::

    nn.ReLU()
    nn.Sigmoid()
    nn.Tanh()
    nn.SoftPlus()

Loss functions
--------------

``nn.cross_entropy_loss(logits, targets, reduction="mean")``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-entropy loss for classification.

- ``logits``: shape ``(N, C)`` — raw (un-normalised) scores
- ``targets``: shape ``(N,)`` — integer class indices in ``[0, C)``
- ``reduction``: ``"mean"`` (default) or ``"sum"``

::

    logits = model(x_batch)              # (32, 10)
    loss = nn.cross_entropy_loss(logits, y_batch)
    loss.backward()

``nn.mse(predictions, targets, reduction="mean", weight=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean squared error loss for regression.

- ``predictions``, ``targets``: same shape
- ``weight``: optional per-sample weight array (same leading dimension)
- ``reduction``: ``"mean"`` (default) or ``"sum"``

::

    pred = model(x_batch)           # (32, 1)
    loss = nn.mse(pred, y_batch)
    loss.backward()
