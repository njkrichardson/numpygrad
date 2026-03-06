Random (``npg.random``)
=======================

The ``npg.random`` module contains all random-array factories and the
seed utility.  Set the seed once at the top of your script for
reproducible results::

    import numpygrad as npg

    npg.random.manual_seed(0)       # or npg.manual_seed(0) — same effect

Random arrays
-------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - Description
   * - ``npg.random.rand(shape, **kw)``
     - Uniform samples in ``[0, 1)``
   * - ``npg.random.randn(shape, **kw)``
     - Standard normal samples
   * - ``npg.random.randint(low, high, size, **kw)``
     - Random integers in ``[low, high)``
   * - ``npg.random.uniform(low, high, size, **kw)``
     - Uniform samples in ``[low, high]``
   * - ``npg.random.normal(mean, std, size, **kw)``
     - Normal samples with given mean and std
   * - ``npg.random.randperm(n, **kw)``
     - Random permutation of ``0 … n-1``

All factory functions accept ``requires_grad`` and ``dtype`` keyword
arguments::

    w = npg.random.randn((4, 4), requires_grad=True)
    i = npg.random.randperm(1000)                    # shuffle indices

Seeding
-------

``npg.random.manual_seed(seed)`` (also available as ``npg.manual_seed``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the NumPy random seed globally for reproducible weight initialisation,
data shuffling, and dropout masks::

    npg.random.manual_seed(42)
    model = npg.nn.MLP(4, [32], 2)   # weights initialised with seed 42
