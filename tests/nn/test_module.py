"""Tests for nn/module.py, nn/linear.py (repr), and nn/mlp.py (repr, bad activation)."""

import numpy as np
import pytest

import numpygrad as npg
from numpygrad.core.array import Array
from numpygrad.nn.linear import Linear
from numpygrad.nn.mlp import MLP
from numpygrad.nn.module import Module, Parameter, Sequential

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LeafModule(Module):
    """A module with a parameter, a buffer, and a child submodule."""

    def __init__(self):
        super().__init__()
        self.w = Parameter(Array(np.array([1.0, 2.0])))
        self.buf = Array(np.array([0.0, 0.0]))  # goes to _buffers

    def forward(self, x: Array) -> Array:
        return x * self.w


class ParentModule(Module):
    """A module that contains a child LeafModule."""

    def __init__(self):
        super().__init__()
        self.child = LeafModule()  # goes to _modules

    def forward(self, x: Array) -> Array:
        return self.child(x)


# ---------------------------------------------------------------------------
# __setattr__ — buffer registration
# ---------------------------------------------------------------------------


def test_buffer_registered():
    m = LeafModule()
    assert "buf" in m._buffers
    assert "buf" not in m._parameters


# ---------------------------------------------------------------------------
# __getattr__ — modules, buffers, AttributeError
# ---------------------------------------------------------------------------


def test_getattr_returns_submodule():
    m = ParentModule()
    child = m.child
    assert isinstance(child, LeafModule)


def test_getattr_returns_buffer():
    m = LeafModule()
    buf = m.buf
    assert isinstance(buf, Array)


def test_getattr_raises_attribute_error():
    m = LeafModule()
    with pytest.raises(AttributeError, match="Module has no attribute"):
        _ = m.nonexistent


# ---------------------------------------------------------------------------
# state_dict — parameters, submodules, buffers
# ---------------------------------------------------------------------------


def test_state_dict_leaf():
    m = LeafModule()
    sd = m.state_dict()
    assert "w" in sd
    assert "buf" in sd
    np.testing.assert_array_equal(sd["w"], np.array([1.0, 2.0]))


def test_state_dict_nested():
    m = ParentModule()
    sd = m.state_dict()
    assert "child" in sd
    assert "w" in sd["child"]


# ---------------------------------------------------------------------------
# parameters() — recursive
# ---------------------------------------------------------------------------


def test_parameters_leaf():
    m = LeafModule()
    params = m.parameters()
    assert len(params) == 1
    assert isinstance(params[0], Parameter)


def test_parameters_nested():
    m = ParentModule()
    params = m.parameters()
    # ParentModule has no own parameters; LeafModule contributes 1
    assert len(params) == 1


# ---------------------------------------------------------------------------
# forward() raises NotImplementedError on bare Module
# ---------------------------------------------------------------------------


def test_bare_module_forward_not_implemented():
    class Bare(Module):
        pass

    m = Bare()
    with pytest.raises(NotImplementedError):
        m(npg.array(np.array([1.0])))


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


def test_sequential_forward():
    seq = Sequential(LeafModule(), LeafModule())
    x = npg.array(np.array([3.0, 4.0]))
    out = seq(x)
    # Each LeafModule multiplies by [1, 2], so [3*1*1, 4*2*2] = [3, 16]
    np.testing.assert_array_equal(out.data, np.array([3.0, 16.0]))


# ---------------------------------------------------------------------------
# Linear.__repr__
# ---------------------------------------------------------------------------


def test_linear_repr():
    layer = Linear(4, 8)
    r = repr(layer)
    assert "Linear" in r
    assert "4" in r
    assert "8" in r


# ---------------------------------------------------------------------------
# MLP.__repr__ and bad activation
# ---------------------------------------------------------------------------


def test_mlp_repr():
    model = MLP(4, [8, 8], 2)
    r = repr(model)
    assert "MLP" in r


def test_mlp_bad_activation():
    with pytest.raises(ValueError, match="not supported"):
        MLP(4, [8], 2, activation="tanh")
