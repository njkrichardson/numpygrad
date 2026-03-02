The Training Loop
=================

This page walks through a complete supervised-learning training loop using
numpygrad's dataset, model, loss, and optimizer APIs.

Data
----

Wrap your NumPy arrays in a ``TensorDataset`` and pass it to a
``DataLoader``::

    import numpy as np
    import numpygrad as npg
    from numpygrad.utils import TensorDataset, DataLoader

    X = np.random.randn(200, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

``DataLoader`` yields ``(x_batch, y_batch)`` tuples of ``Array`` objects on
each iteration. Set ``drop_last=True`` if your loss function is sensitive to
batch-size variation.

Model
-----

Use a built-in module or define your own (see :doc:`custom_modules`)::

    import numpygrad.nn as nn

    model = nn.MLP(
        input_dim=4,
        hidden_sizes=[32, 32],
        output_dim=2,
        activation="relu",
    )

Optimizer
---------

Pass ``model.parameters()`` to an optimizer::

    optimizer = npg.optim.AdamW(model.parameters(), lr=1e-3)

``SGD`` is also available for simpler experiments::

    optimizer = npg.optim.SGD(model.parameters(), step_size=1e-2)

The loop
--------

A standard training step has four parts: reset gradients, forward pass,
compute loss, backward pass, parameter update::

    for epoch in range(10):
        for x_batch, y_batch in loader:

            # 1. reset accumulated gradients
            optimizer.zero_grad()

            # 2. forward pass
            logits = model(x_batch)

            # 3. loss
            loss = nn.cross_entropy_loss(logits, y_batch)

            # 4. backward + update
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch}  loss {loss.item():.4f}")

Validation
----------

Wrap the validation pass in ``npg.no_grad()`` to skip graph construction::

    with npg.no_grad():
        val_logits = model(x_val)
        val_loss = nn.cross_entropy_loss(val_logits, y_val)
        correct = (val_logits.numpy().argmax(axis=1) == y_val.numpy()).mean()
        print(f"val loss {val_loss.item():.4f}  acc {correct:.2%}")

Putting it all together
-----------------------

::

    import numpy as np
    import numpygrad as npg
    import numpygrad.nn as nn
    from numpygrad.utils import TensorDataset, DataLoader

    # --- data ---
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    X_train, y_train = X[:320], y[:320]
    X_val, y_val = X[320:], y[320:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

    # --- model ---
    model = nn.MLP(input_dim=4, hidden_sizes=[32, 32], output_dim=2)
    optimizer = npg.optim.AdamW(model.parameters(), lr=1e-3)

    # --- train ---
    for epoch in range(20):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = nn.cross_entropy_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

    # --- evaluate ---
    with npg.no_grad():
        val_loss = nn.cross_entropy_loss(
            model(npg.array(X_val)), npg.array(y_val)
        )
        print(f"val loss: {val_loss.item():.4f}")
