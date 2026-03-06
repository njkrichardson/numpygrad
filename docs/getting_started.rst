Getting Started
===============

Installation
------------

Install from PyPI::

    pip install numpygrad

Or install from source in editable mode::

    git clone https://github.com/njkrichardson/numpygrad.git
    cd numpygrad
    pip install -e .

To also install test dependencies::

    pip install -e ".[tests]"

Requirements: Python >= 3.12, NumPy >= 2.4.2.

Quickstart
----------

Create arrays with automatic differentiation::

    import numpygrad as npg

    x = npg.array([[1.0, 2.0, 3.0]], requires_grad=True)
    w = npg.array([[0.5], [0.5], [0.5]], requires_grad=True)

    y = x @ w          # matrix multiply
    loss = y.sum()
    loss.backward()

    print(x.grad)      # d(loss)/d(x) — shape (1, 3)
    print(w.grad)      # d(loss)/d(w) — shape (3, 1)

Train a small network in a few lines::

    import numpygrad as npg
    import numpygrad.nn as nn

    model = nn.MLP(input_dim=2, hidden_sizes=[16, 16], output_dim=1)
    optimizer = npg.optim.SGD(model.parameters(), step_size=1e-3)

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = nn.mse(pred, y_batch)
        loss.backward()
        optimizer.step()

See the :doc:`guide/training_loop` for a complete working example.
