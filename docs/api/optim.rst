Optimizers (``optim``)
======================

Optimizers update model parameters using accumulated gradients. Import them
from ``numpygrad.optim``::

    import numpygrad.optim as optim

All optimizers share the same interface: construct with a parameter list, call
``zero_grad()`` before each forward pass, and call ``step()`` after
``backward()``::

    optimizer = optim.SGD(model.parameters(), step_size=1e-2)

    optimizer.zero_grad()   # reset .grad on all params
    loss.backward()         # accumulate gradients
    optimizer.step()        # update params

``optim.Optimizer`` (base class)
---------------------------------

Provides ``zero_grad()`` which sets ``param.grad = None`` for every parameter.
Subclasses must implement ``step()``.

``optim.SGD``
-------------

Vanilla stochastic gradient descent::

    optimizer = optim.SGD(model.parameters(), step_size=1e-3)

Each step applies ``param.data -= step_size * param.grad`` with no momentum or
weight decay. Good for quick experiments and small models.

``optim.AdamW``
---------------

Adam with decoupled weight decay::

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )

Parameters:

- ``lr`` — learning rate (default ``1e-3``)
- ``betas`` — exponential decay rates for the first and second moment
  estimates (default ``(0.9, 0.999)``)
- ``eps`` — numerical stability term added to the denominator (default ``1e-8``)
- ``weight_decay`` — L2 regularisation coefficient applied directly to weights,
  **decoupled** from the gradient update (default ``1e-2``)

AdamW is the recommended default for most tasks. Use ``SGD`` when you want
full control over the update rule or are studying optimisation dynamics.
