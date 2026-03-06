How Autograd Works
==================

numpygrad uses **define-by-run** (also called dynamic) automatic differentiation.
The computation graph is built implicitly as you run operations — there is no
separate "compilation" step.

The computation graph
---------------------

Every ``Array`` that participates in a differentiable computation holds a
reference to the ``Function`` that produced it (``array.grad_fn``) and to its
input ``Array`` nodes (``array.parents``). Together these form a directed
acyclic graph (DAG) where:

- **Nodes** are ``Array`` objects.
- **Edges** point from an output array back to its inputs.
- **Leaf nodes** are arrays you created directly (e.g. with ``npg.array(...)``
  or ``npg.random.randn(...)``). They have no ``grad_fn``.

A simple example::

    import numpygrad as npg

    a = npg.array([2.0], requires_grad=True)   # leaf
    b = npg.array([3.0], requires_grad=True)   # leaf
    c = a * b                                  # c.grad_fn = Mul
    d = c + a                                  # d.grad_fn = Add

The graph for ``d`` looks like::

    d (Add)
    ├── c (Mul)
    │   ├── a (leaf)
    │   └── b (leaf)
    └── a (leaf)

Calling ``backward()``
----------------------

``array.backward()`` traverses the graph in **reverse topological order**,
calling each ``Function``'s ``backward`` method and accumulating gradients
into ``array.grad`` for every leaf with ``requires_grad=True``::

    d.backward()
    print(a.grad)   # d(d)/d(a) = d/da[(a*b) + a] = b + 1 = 4.0
    print(b.grad)   # d(d)/d(b) = d/db[(a*b)] = a = 2.0

By default ``backward()`` starts with a scalar gradient of 1. For non-scalar
outputs pass an explicit gradient array::

    out = npg.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    result = out * 2
    result.backward(npg.ones_like(result).data)

Gradient accumulation
---------------------

Gradients **accumulate** in ``array.grad`` rather than being overwritten. This
matches PyTorch's behaviour and is required when an array appears multiple
times in a graph (like ``a`` in the example above). Call
``optimizer.zero_grad()`` (or set ``array.grad = None``) between training
steps to reset them.

Non-differentiable operations
------------------------------

Some ``Array`` methods — ``astype``, ``nonzero``, ``all``, ``any``, ``fill``,
``sort``, ``round`` — do not propagate gradients. They return a new array but
do not attach a ``grad_fn``.

Comparison operators (``>``, ``<``, ``==``, etc.) also return arrays without
gradient tracking, since they are not differentiable.

Broadcasting and gradients
--------------------------

When an operation broadcasts one operand to match the shape of another,
``backward()`` automatically sums the upstream gradient over the broadcasted
axes to recover the gradient of the original (smaller) shape. You do not need
to handle this manually.
