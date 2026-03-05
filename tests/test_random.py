import numpy as np

import numpygrad as npg


def test_rand_shape():
    a = npg.random.rand((2, 3))
    assert a.shape == (2, 3)


def test_rand_int_shape():
    a = npg.random.rand(5)
    assert a.shape == (5,)


def test_rand_values_in_range():
    a = npg.random.rand((1000,))
    assert np.all(a.data >= 0.0) and np.all(a.data < 1.0)


def test_rand_requires_grad():
    a = npg.random.rand((3,), requires_grad=True)
    assert a.requires_grad


def test_randn_shape():
    a = npg.random.randn((4, 5))
    assert a.shape == (4, 5)


def test_randn_int_shape():
    a = npg.random.randn(7)
    assert a.shape == (7,)


def test_randn_requires_grad():
    a = npg.random.randn((3,), requires_grad=True)
    assert a.requires_grad


def test_randint_shape():
    a = npg.random.randint(0, 10, (5,))
    assert a.shape == (5,)


def test_randint_int_size():
    a = npg.random.randint(0, 10, 5)
    assert a.shape == (5,)


def test_randint_single_arg():
    a = npg.random.randint(10, size=(4,))
    assert a.shape == (4,)
    assert np.all(a.data >= 0) and np.all(a.data < 10)


def test_uniform_shape():
    a = npg.random.uniform(-1.0, 1.0, (100,))
    assert a.shape == (100,)


def test_uniform_values_in_range():
    a = npg.random.uniform(-1.0, 1.0, (1000,))
    assert np.all(a.data >= -1.0) and np.all(a.data <= 1.0)


def test_uniform_requires_grad():
    a = npg.random.uniform(requires_grad=True)
    assert a.requires_grad


def test_normal_shape():
    a = npg.random.normal(0.0, 1.0, (1000,))
    assert a.shape == (1000,)


def test_normal_int_size():
    a = npg.random.normal(size=10)
    assert a.shape == (10,)


def test_normal_requires_grad():
    a = npg.random.normal(requires_grad=True)
    assert a.requires_grad


def test_randperm_is_permutation():
    p = npg.random.randperm(8)
    assert p.shape == (8,)
    assert sorted(p.data.tolist()) == list(range(8))


def test_randperm_kwargs():
    # randperm returns int indices; verify kwargs pass through (e.g. label)
    p = npg.random.randperm(4, label="perm")
    assert p.shape == (4,)


def test_manual_seed_reexport():
    npg.random.manual_seed(42)
    a = npg.random.randn((3,))
    npg.random.manual_seed(42)
    b = npg.random.randn((3,))
    np.testing.assert_array_equal(a.data, b.data)


def test_randn_not_at_top_level():
    assert not hasattr(npg, "randn")


def test_randint_not_at_top_level():
    assert not hasattr(npg, "randint")
