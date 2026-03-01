"""Tests for in-place tensor mutation with version counter checks."""

import numpy as np
import pytest
import torch

import numpygrad as npg


def test_mutate_after_forward_before_backward_raises():
    """Test 1: x used in op, mutated before backward() → RuntimeError."""
    x = npg.array([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    x[0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        y.backward()


def test_mutate_before_op_no_error():
    """Test 2: x mutated before op is called → no error."""
    x = npg.array([1.0, 2.0, 3.0], requires_grad=True)
    x[0] = 99.0
    y = (x**2).sum()
    y.backward()  # should not raise


def test_mutate_after_backward_no_error():
    """Test 3: backward() completes, then x mutated → no error."""
    x = npg.array([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    y.backward()
    x[0] = 99.0  # should not raise


def test_mutate_no_grad_array_no_error():
    """Test 4: non-requires-grad array mutated via __setitem__ → no error."""
    x = npg.array([1.0, 2.0, 3.0], requires_grad=False)
    x[0] = 99.0  # should not raise
    assert x.data[0] == 99.0


def test_reshape_view_mutation_raises():
    """Test 5: y = x.reshape(...) (view); mutate x; backward through y → RuntimeError."""
    x = npg.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.reshape(4)
    loss = (y**2).sum()
    x[0, 0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        loss.backward()


def test_transpose_view_mutation_raises():
    """Test 6: y = npg.transpose(x, ...) (always a view); mutate x → RuntimeError."""
    x = npg.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose((1, 0))
    loss = (y**2).sum()
    x[0, 0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        loss.backward()


def test_fancy_index_copy_no_error():
    """Test 7: y = x[[0,1]] (fancy index → copy); mutate x; backward → no error."""
    x = npg.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = x[np.array([0, 1])]
    loss = (y**2).sum()
    x[2, 0] = 99.0
    loss.backward()  # should not raise


def test_chained_reshape_view_mutation_raises():
    """Test 8: z = y.reshape(…) where y = x.reshape(…) (chain); mutate x → RuntimeError."""
    x = npg.array([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = x.reshape((2, 2))
    z = y.reshape((4,))
    loss = (z**2).sum()
    x[0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        loss.backward()


def test_no_grad_mutation_no_error():
    """Test 9: inside no_grad context, mutation happens, no save → no error."""
    x = npg.array([1.0, 2.0, 3.0], requires_grad=True)
    with npg.no_grad():
        (x**2).sum()
    x[0] = 99.0  # no save happened; no error expected


def test_matmul_mutate_before_backward_raises():
    """Test 10: Matmul input mutated after forward, before backward → RuntimeError."""
    a = npg.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = npg.array([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    c = a @ b
    loss = c.sum()
    a[0, 0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        loss.backward()


def test_basic_slice_view_mutation_raises():
    """Test 11: y = x[0:2] (basic slice → view); mutate x; backward → RuntimeError."""
    x = npg.array([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = x[0:2]
    loss = (y**2).sum()
    x[0] = 99.0
    with pytest.raises(RuntimeError, match="mutated by an in-place operation"):
        loss.backward()


def test_unrelated_graph_mutation_no_error():
    """Test 12: mutate tensor in graph B; backward through graph A → no error."""
    a = npg.array([1.0, 2.0, 3.0], requires_grad=True)
    b = npg.array([4.0, 5.0, 6.0], requires_grad=True)

    loss_a = (a**2).sum()
    (b**2).sum()

    b[0] = 99.0  # mutate graph B

    loss_a.backward()  # should not raise — a was not mutated


# ---------------------------------------------------------------------------
# Gradient correctness for legal mutation cases
# ---------------------------------------------------------------------------


def test_mutate_before_op_grad_correct():
    """Mutate x, build graph, backward — grad reflects the mutated value."""
    arr = np.array([1.0, 2.0, 3.0])
    x = npg.array(arr.copy(), requires_grad=True)
    x[0] = 99.0
    (x**2).sum().backward()

    xt = torch.tensor(arr, requires_grad=True)
    xt.data[0] = 99.0
    (xt**2).sum().backward()

    np.testing.assert_allclose(x.grad, xt.grad.numpy())


def test_fancy_index_copy_grad_correct():
    """y = x[fancy] (copy); mutate an unindexed row; backward scatters grad to indexed rows only."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = npg.array(arr.copy(), requires_grad=True)
    y = x[np.array([0, 1])]
    (y**2).sum().backward()
    # x[2] was never sliced into y; grad there must be zero
    # x[0] and x[1] were copied into y; grad = 2 * original values
    expected = np.array([[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]])
    np.testing.assert_allclose(x.grad, expected)


def test_unrelated_graph_mutation_grad_correct():
    """Mutate tensor in graph B; backward through graph A returns correct grad."""
    arr_a = np.array([1.0, 2.0, 3.0])
    a = npg.array(arr_a.copy(), requires_grad=True)
    b = npg.array([4.0, 5.0, 6.0], requires_grad=True)

    loss_a = (a**2).sum()
    (b**2).sum()  # build graph B but don't backward yet

    b[0] = 99.0  # mutate B
    loss_a.backward()

    at = torch.tensor(arr_a, requires_grad=True)
    (at**2).sum().backward()

    np.testing.assert_allclose(a.grad, at.grad.numpy())
