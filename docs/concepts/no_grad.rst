Disabling Autograd
==================

By default, every operation involving an array with ``requires_grad=True``
records itself into the computation graph. During evaluation and inference you
usually don't need gradients — disabling autograd avoids the overhead and
memory of building the graph.

``npg.no_grad``
---------------

Use ``npg.no_grad`` as a context manager to suspend gradient tracking for a
block of code::

    import numpygrad as npg

    x = npg.random.randn((4, 8), requires_grad=True)
    model = ...

    with npg.no_grad():
        pred = model(x)   # no graph is built
        # pred.grad_fn is None

Operations inside the block still run and return correct numeric results —
gradients just aren't tracked.

Use as a decorator
------------------

``npg.no_grad`` also works as a function decorator::

    @npg.no_grad()
    def evaluate(model, x, y):
        pred = model(x)
        return nn.mse(pred, y)

    val_loss = evaluate(model, x_val, y_val)

Typical usage
-------------

The most common pattern is to use ``no_grad`` around your validation loop::

    # training step — gradients needed
    optimizer.zero_grad()
    loss = nn.cross_entropy_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()

    # validation — no gradients needed
    with npg.no_grad():
        val_loss = nn.cross_entropy_loss(model(x_val), y_val)
        print(f"val loss: {val_loss.item():.4f}")
