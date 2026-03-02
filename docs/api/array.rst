Array
=====

The ``Array`` class is numpygrad's core tensor type. It wraps a NumPy
``ndarray`` and optionally participates in the computation graph.

Creating arrays
---------------

**From data**

.. code-block:: python

    npg.array(data, *, requires_grad=False, dtype=None)

``data`` can be a Python list, NumPy array, or scalar. Use
``requires_grad=True`` for any array whose gradient you want to compute.

**Factory functions**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.zeros(shape, **kw)``
     - Array of zeros
   * - ``npg.ones(shape, **kw)``
     - Array of ones
   * - ``npg.zeros_like(x, **kw)``
     - Zeros with the same shape as ``x``
   * - ``npg.arange(start, stop, step, **kw)``
     - Evenly spaced values in ``[start, stop)``
   * - ``npg.linspace(start, stop, num)``
     - ``num`` evenly spaced values in ``[start, stop]``
   * - ``npg.eye(n)``
     - ``n×n`` identity matrix
   * - ``npg.randn(shape, **kw)``
     - Samples from a standard normal distribution
   * - ``npg.randint(low, high, size, **kw)``
     - Random integers in ``[low, high)``

All factory functions accept ``requires_grad`` and ``dtype`` keyword arguments.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``shape``
     - Tuple of dimension sizes
   * - ``ndim``
     - Number of dimensions
   * - ``dtype``
     - NumPy dtype (e.g. ``np.float32``)
   * - ``size``
     - Total number of elements
   * - ``nbytes``
     - Total bytes in the underlying buffer
   * - ``itemsize``
     - Bytes per element
   * - ``strides``
     - Byte strides of the underlying array
   * - ``T``
     - Transposed array (reverses all axes)
   * - ``requires_grad``
     - Whether this array accumulates gradients
   * - ``grad``
     - Accumulated gradient (``None`` until ``backward()`` is called)
   * - ``grad_fn``
     - The ``Function`` class that produced this array, or ``None``

Accessing data
--------------

``array.numpy()`` returns the underlying ``np.ndarray`` without a copy::

    x = npg.array([1.0, 2.0, 3.0])
    x.numpy()   # np.array([1., 2., 3.])

``array.item()`` extracts a scalar from a single-element array::

    loss = npg.array([0.42])
    loss.item()   # 0.42  (Python float)

``array.tolist()`` converts to a nested Python list (no gradient tracking).

Indexing
--------

Arrays support standard NumPy-style indexing. Getting a slice is
differentiable::

    x = npg.randn((4, 8), requires_grad=True)
    row = x[0]           # differentiable slice
    block = x[1:3, 2:5]  # differentiable slice

In-place assignment via ``__setitem__`` is also supported and increments the
internal version counter so that any ``backward()`` that saved the pre-mutation
array detects the conflict and raises an error::

    x = npg.zeros((3, 3))
    x[1, 1] = 5.0   # ok — in-place mutation

Arithmetic operators
--------------------

All standard arithmetic operators dispatch to the corresponding differentiable
op and return a new ``Array``:

``+``, ``-``, ``*``, ``/``, ``**``, ``@`` (matmul), unary ``-``.

Comparison operators (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``) return
boolean arrays **without** gradient tracking.

Methods
-------

**Reductions** — all accept ``axis`` and ``keepdims``

``sum``, ``mean``, ``max``, ``min``, ``prod``, ``argmax``, ``var``,
``std``, ``cumsum``, ``cumprod``

**Shape transforms**

``reshape(new_shape)``, ``view(new_shape)`` (alias), ``flatten()``,
``transpose(axes)``, ``squeeze(axis=None)``, ``repeat(repeats, axis=None)``

**Math**

``exp()``, ``log()``, ``abs()``, ``sqrt()``

**Linear algebra**

``diagonal(offset=0, axis1=0, axis2=1)``, ``trace(offset=0)``

**Non-differentiable utilities**

``astype(dtype)``, ``nonzero()``, ``tolist()``, ``all(axis)``,
``any(axis)``, ``fill(value)``, ``sort(axis=-1)``, ``round(decimals=0)``

Backward
--------

Call ``backward()`` on a scalar array to compute gradients for all upstream
leaf arrays with ``requires_grad=True``::

    a = npg.array([3.0], requires_grad=True)
    b = npg.array([4.0], requires_grad=True)
    c = (a ** 2 + b ** 2).sqrt()
    c.backward()
    print(a.grad)   # [0.6]  (= a/c)
    print(b.grad)   # [0.8]  (= b/c)

For non-scalar outputs, pass an explicit upstream gradient of the same shape::

    out = npg.randn((3, 4), requires_grad=True) * 2
    out.backward(npg.ones((3, 4)).data)

Dtypes
------

numpygrad re-exports four NumPy dtypes as module-level names:

``npg.float32``, ``npg.float64``, ``npg.int32``, ``npg.int64``

Use them anywhere NumPy accepts a dtype::

    x = npg.zeros((4,), dtype=npg.float64)
