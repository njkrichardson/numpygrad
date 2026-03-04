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
- ``module.train()`` / ``module.eval()`` — switch training mode on/off
  (affects ``Dropout``)

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

``nn.Linear(num_inputs, num_outputs, bias=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fully connected layer: ``y = x @ W + b``.

- ``weight``: ``Parameter`` of shape ``(num_inputs, num_outputs)``
- ``bias``: ``Parameter`` of shape ``(num_outputs,)``, or absent when ``bias=False``

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

``nn.Embedding(num_embeddings, embedding_dim)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lookup table mapping integer indices to dense vectors. Equivalent to an
indexed row lookup with full gradient support::

    embed = nn.Embedding(vocab_size, 64)
    x = npg.array([3, 1, 4, 1, 5])   # integer indices
    out = embed(x)                    # (5, 64)

``nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layer normalisation over the last ``len(normalized_shape)`` dimensions.
When ``elementwise_affine=True`` (the default), learnable ``weight`` (gamma)
and ``bias`` (beta) parameters are added::

    ln = nn.LayerNorm(512)
    out = ln(npg.randn((4, 16, 512)))   # (4, 16, 512), normalised over dim=-1

``nn.Dropout(p=0.5)``
~~~~~~~~~~~~~~~~~~~~~

Randomly zeros elements with probability ``p`` during training and rescales
the remaining values by ``1/(1-p)`` (inverted dropout). Dropout is a no-op
during evaluation (after calling ``model.eval()``)::

    drop = nn.Dropout(p=0.1)
    out = drop(x)   # during training: randomly mask 10% of values

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
    nn.GELU()
    nn.Sigmoid()
    nn.Tanh()
    nn.SoftPlus()

Loss functions
--------------

``nn.cross_entropy_loss(logits, targets, reduction="mean")``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-entropy loss for classification. Supports both 2D ``(N, C)`` and
higher-dimensional ``(*, C)`` logits — the extra dimensions are flattened
automatically::

    # 2D: standard (batch, classes)
    logits = model(x_batch)              # (32, 10)
    loss = nn.cross_entropy_loss(logits, y_batch)
    loss.backward()

    # 3D: sequence models (batch, seq_len, vocab_size)
    logits = lm(tokens)                  # (B, T, V)
    loss = nn.cross_entropy_loss(logits, targets)   # targets: (B, T)

- ``logits``: shape ``(*, C)`` — raw (un-normalised) scores
- ``targets``: shape ``(*,)`` — integer class indices in ``[0, C)``
- ``reduction``: ``"mean"`` (default) or ``"sum"``

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

Parameter initialisation (``nn.init``)
---------------------------------------

The ``nn.init`` submodule provides in-place parameter initialisation helpers,
following the same convention as ``torch.nn.init``. Every function modifies
``tensor.data`` in-place and returns the tensor::

    import numpygrad.nn as nn

    w = nn.Parameter(npg.zeros((128, 64)))
    nn.init.kaiming_uniform_(w, nonlinearity="relu")

Basic
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Description
   * - ``nn.init.uniform_(tensor, low=-1, high=1)``
     - Uniform distribution over ``[low, high]``
   * - ``nn.init.normal_(tensor, mean=0, std=1)``
     - Normal distribution
   * - ``nn.init.zeros_(tensor)``
     - Fill with zeros
   * - ``nn.init.ones_(tensor)``
     - Fill with ones

Kaiming (He)
~~~~~~~~~~~~

Variance-preserving initialisation for networks with ReLU-family activations:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Description
   * - ``nn.init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity="relu")``
     - Kaiming uniform: :math:`\mathcal{U}(-\text{bound}, \text{bound})` where :math:`\text{bound} = \sqrt{3} \cdot \text{gain} / \sqrt{\text{fan}}`
   * - ``nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu")``
     - Kaiming normal: :math:`\mathcal{N}(0, \text{gain}^2/\text{fan})`

Xavier (Glorot)
~~~~~~~~~~~~~~~

Variance-preserving initialisation for networks with symmetric activations
(tanh, sigmoid):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Description
   * - ``nn.init.xavier_uniform_(tensor, gain=1.0)``
     - Xavier uniform: :math:`\mathcal{U}(-b, b)` where :math:`b = \text{gain} \cdot \sqrt{6 / (\text{fan\_in} + \text{fan\_out})}`
   * - ``nn.init.xavier_normal_(tensor, gain=1.0)``
     - Xavier normal: :math:`\mathcal{N}(0, \text{gain}^2 \cdot 2 / (\text{fan\_in} + \text{fan\_out}))`

``mode`` controls whether fan_in or fan_out is used. ``nonlinearity`` sets
the gain; recognised values are ``"relu"``, ``"gelu"``, ``"tanh"``,
``"sigmoid"``, ``"leaky_relu"``, ``"linear"``, ``"identity"``.
Fan is computed from the tensor shape: ``(fan_in, fan_out)`` for 2D tensors,
``(C_in * KH * KW, C_out * KH * KW)`` for 4D (Conv) tensors.
