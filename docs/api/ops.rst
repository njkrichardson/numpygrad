Operators
=========

All operators are available both as module-level functions (``npg.relu(x)``)
and as ``Array`` methods (``x.relu()`` where applicable). Every operator listed
here is differentiable — it records itself into the computation graph when any
input has ``requires_grad=True``.

Element-wise
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.add(a, b)``
     - Element-wise addition (also ``a + b``)
   * - ``npg.mul(a, b)``
     - Element-wise multiplication (also ``a * b``)
   * - ``npg.pow(a, exponent)``
     - Element-wise power (also ``a ** exponent``)
   * - ``npg.exp(a)``
     - Element-wise :math:`e^x`
   * - ``npg.log(a)``
     - Natural logarithm (undefined for non-positive values)
   * - ``npg.abs(a)``
     - Absolute value
   * - ``npg.relu(a)``
     - ``max(0, x)`` element-wise
   * - ``npg.clip(a, min, max)``
     - Clamp values to ``[min, max]``
   * - ``npg.maximum(a, b)``
     - Element-wise max of two arrays
   * - ``npg.minimum(a, b)``
     - Element-wise min of two arrays

Reductions
----------

All reduction functions accept ``axis=None`` (reduce all axes) and
``keepdims=False``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.sum(a, axis, keepdims)``
     - Sum of elements
   * - ``npg.mean(a, axis, keepdims)``
     - Mean of elements
   * - ``npg.prod(a, axis, keepdims)``
     - Product of elements
   * - ``npg.max(a, axis, keepdims)``
     - Maximum value
   * - ``npg.min(a, axis, keepdims)``
     - Minimum value
   * - ``npg.argmax(a, axis, keepdims)``
     - Index of maximum value (no gradient)
   * - ``npg.var(a, axis, ddof, keepdims)``
     - Variance. ``ddof=0`` (population) or ``ddof=1`` (sample)
   * - ``npg.std(a, axis, ddof, keepdims)``
     - Standard deviation (``sqrt(var(...))``)
   * - ``npg.cumsum(a, axis)``
     - Cumulative sum along ``axis``
   * - ``npg.cumprod(a, axis)``
     - Cumulative product along ``axis``

Activations
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.softmax(a, axis=-1)``
     - Softmax along ``axis``
   * - ``npg.log_softmax(a, axis=-1)``
     - Log-softmax (numerically stable)
   * - ``npg.sigmoid(a)``
     - :math:`\sigma(x) = 1 / (1 + e^{-x})`
   * - ``npg.tanh(a)``
     - Hyperbolic tangent
   * - ``npg.softplus(a)``
     - :math:`\log(1 + e^x)` (smooth approximation of ReLU)
   * - ``npg.relu(a)``
     - ``max(0, x)`` (also listed under element-wise)

Linear algebra
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.matmul(a, b)`` / ``npg.mm(a, b)``
     - Matrix multiplication. Handles 1D (dot product), 2D, and batched 3D inputs.
   * - ``npg.dot(a, b)``
     - Dot product of two 1D or 2D arrays
   * - ``npg.norm(a, axis, keepdims)``
     - Frobenius / L2 norm
   * - ``npg.diagonal(a, offset, axis1, axis2)``
     - Extract diagonal elements
   * - ``npg.trace(a, offset)``
     - Sum of diagonal elements (``diagonal(a, offset).sum()``)

Shape transforms
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``npg.reshape(a, new_shape)``
     - Change shape without changing data. Returns a view when possible.
   * - ``npg.transpose(a, axes)``
     - Permute dimensions. ``axes`` is a tuple; ``None`` reverses all.
   * - ``npg.flatten(a)``
     - Flatten to 1D (equivalent to ``reshape(a, (-1,))``)
   * - ``npg.unsqueeze(a, axis)``
     - Insert a size-1 dimension at ``axis``
   * - ``npg.squeeze(a, axis=None)``
     - Remove size-1 dimensions (all if ``axis=None``)
   * - ``npg.repeat(a, repeats, axis=None)``
     - Repeat elements along an axis
   * - ``npg.stack(arrays, axis=0)``
     - Stack a list of arrays along a new axis
   * - ``npg.cat(arrays, axis=0)``
     - Concatenate arrays along an existing axis

Convolution
-----------

``npg.conv2d(input, weight, bias=None, stride=1, padding=0)``

2D convolution with full backward support.

- ``input``: shape ``(N, C_in, H, W)``
- ``weight``: shape ``(C_out, C_in, KH, KW)``
- ``bias``: shape ``(C_out,)`` or ``None``
- ``stride`` and ``padding`` accept an int or a ``(H, W)`` tuple
- Output shape: ``(N, C_out, H_out, W_out)``

Example::

    import numpygrad as npg

    x = npg.randn((2, 3, 32, 32))                # batch of 2 RGB images
    w = npg.randn((16, 3, 3, 3), requires_grad=True)
    out = npg.conv2d(x, w, stride=1, padding=1)  # (2, 16, 32, 32)

Special
-------

``npg.setitem(a, key, value)``

Differentiable in-place assignment. Equivalent to ``a[key] = value`` but
records the operation in the computation graph, allowing gradients to flow
through the assignment::

    a = npg.zeros((4,), requires_grad=True)
    b = npg.setitem(a, 2, npg.array([5.0]))
    b.sum().backward()
    print(a.grad)   # [1., 1., 1., 1.]
