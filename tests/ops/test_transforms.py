import numpy as np
import numpy.random as npr
import torch
from hypothesis import given

import numpygrad as npg
from tests.configuration import check_equality
from tests.strategies import (
    FLOAT_DTYPES,
    cat_arrays,
    generic_array,
    reshape_args,
    slice_args,
    split_args,
    stack_arrays,
    transpose_args,
    unsqueeze_args,
)

npg.manual_seed(0)
npr.seed(0)


# --- transpose ---


@given(transpose_args())
def test_transpose_basic(data):
    arr, axes = data
    x = npg.array(arr)
    z = npg.transpose(x, axes)
    reference = np.transpose(arr, axes=axes)
    check_equality(z.data, reference)


@given(transpose_args(dtypes=FLOAT_DTYPES))
def test_transpose_backward(data):
    arr, axes = data
    x = npg.array(arr, requires_grad=True)
    z = npg.transpose(x, axes)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.permute(*axes)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- reshape ---


@given(reshape_args())
def test_reshape_basic(data):
    arr, new_shape = data
    x = npg.array(arr)
    z = npg.reshape(x, new_shape)
    reference = np.reshape(arr, new_shape)
    check_equality(z.data, reference)


@given(reshape_args(dtypes=FLOAT_DTYPES))
def test_reshape_backward(data):
    arr, new_shape = data
    x = npg.array(arr, requires_grad=True)
    z = npg.reshape(x, new_shape)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.reshape(new_shape)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- slice ---


@given(slice_args())
def test_slice_basic(data):
    arr, key = data
    x = npg.array(arr)
    z = x[key]
    reference = arr[key]
    check_equality(z.data, reference)


@given(slice_args(dtypes=FLOAT_DTYPES, allow_negative_step=False))
def test_slice_backward(data):
    arr, key = data
    x = npg.array(arr, requires_grad=True)
    z = x[key]
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt[key]
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- unsqueeze ---


@given(unsqueeze_args())
def test_unsqueeze_basic(data):
    arr, axis = data
    x = npg.array(arr)
    z = npg.unsqueeze(x, axis)
    reference = np.expand_dims(arr, axis=axis)
    check_equality(z.data, reference)


@given(unsqueeze_args(dtypes=FLOAT_DTYPES))
def test_unsqueeze_backward(data):
    arr, axis = data
    x = npg.array(arr, requires_grad=True)
    z = npg.unsqueeze(x, axis)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.unsqueeze(axis)
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- flatten ---


@given(generic_array())
def test_flatten_basic(arr: np.ndarray):
    x = npg.array(arr)
    z = npg.flatten(x)
    reference = np.ravel(arr)
    check_equality(z.data, reference)


@given(generic_array(dtypes=FLOAT_DTYPES))
def test_flatten_backward(arr: np.ndarray):
    x = npg.array(arr, requires_grad=True)
    z = npg.flatten(x)
    z.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    zt = xt.flatten()
    gxt = torch.autograd.grad(
        outputs=zt,
        inputs=(xt,),
        grad_outputs=torch.ones_like(zt),
    )[0]
    assert x.grad is not None
    check_equality(x.grad, gxt.numpy())


# --- stack ---


@given(stack_arrays(n=2, min_dims=1, max_dims=4, dtypes=FLOAT_DTYPES))
def test_stack_basic(data):
    arrays, axis = data
    xs = [npg.array(a) for a in arrays]
    z = npg.stack(tuple(xs), axis=axis)
    reference = np.stack(arrays, axis=axis)
    check_equality(z.data, reference)


@given(stack_arrays(n=2, min_dims=1, max_dims=4, dtypes=FLOAT_DTYPES))
def test_stack_backward(data):
    arrays, axis = data
    xs = [npg.array(a, requires_grad=True) for a in arrays]
    z = npg.stack(tuple(xs), axis=axis)
    z.backward()

    xts = [torch.from_numpy(a).requires_grad_(True) for a in arrays]
    zt = torch.stack(xts, dim=axis)
    gxts = torch.autograd.grad(
        outputs=zt,
        inputs=xts,
        grad_outputs=torch.ones_like(zt),
    )
    for x, gxt in zip(xs, gxts, strict=False):
        assert x.grad is not None
        check_equality(x.grad, gxt.numpy())


# --- cat ---


@given(cat_arrays(n=2, min_dims=1, max_dims=4, dtypes=FLOAT_DTYPES))
def test_cat_basic(data):
    arrays, axis = data
    xs = [npg.array(a) for a in arrays]
    z = npg.cat(tuple(xs), axis=axis)
    reference = np.concatenate(arrays, axis=axis)
    check_equality(z.data, reference)


@given(cat_arrays(n=2, min_dims=1, max_dims=4, dtypes=FLOAT_DTYPES))
def test_cat_backward(data):
    arrays, axis = data
    xs = [npg.array(a, requires_grad=True) for a in arrays]
    z = npg.cat(tuple(xs), axis=axis)
    z.backward()

    xts = [torch.from_numpy(a).requires_grad_(True) for a in arrays]
    zt = torch.cat(xts, dim=axis)
    gxts = torch.autograd.grad(
        outputs=zt,
        inputs=xts,
        grad_outputs=torch.ones_like(zt),
    )
    for x, gxt in zip(xs, gxts, strict=False):
        assert x.grad is not None
        check_equality(x.grad, gxt.numpy())


# --- squeeze ---


def test_squeeze_basic():
    arr = np.random.randn(1, 3, 1, 4).astype(np.float64)
    x = npg.array(arr)
    z = x.squeeze()
    check_equality(z.data, np.squeeze(arr))
    assert z.shape == (3, 4)


def test_squeeze_axis():
    arr = np.random.randn(1, 3, 1, 4).astype(np.float64)
    x = npg.array(arr)
    z = x.squeeze(axis=0)
    check_equality(z.data, np.squeeze(arr, axis=0))
    assert z.shape == (3, 1, 4)


def test_squeeze_backward():
    arr = np.random.randn(1, 3, 1, 4).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    y = x.squeeze()
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.squeeze()
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy())


def test_squeeze_functional():
    arr = np.random.randn(1, 3).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.squeeze(x).data, np.squeeze(arr))


# --- repeat ---


def test_repeat_basic():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr)
    z = x.repeat(2, axis=0)
    check_equality(z.data, np.repeat(arr, 2, axis=0))


def test_repeat_flat():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr)
    z = x.repeat(3)
    check_equality(z.data, np.repeat(arr, 3))


def test_repeat_backward_axis():
    arr = np.random.randn(3, 4).astype(np.float64)
    for axis in [0, 1]:
        x = npg.array(arr, requires_grad=True)
        y = x.repeat(3, axis=axis)
        y.sum().backward()

        xt = torch.from_numpy(arr).requires_grad_(True)
        yt = xt.repeat_interleave(3, dim=axis)
        yt.sum().backward()
        assert x.grad is not None
        check_equality(x.grad, xt.grad.numpy())


def test_repeat_backward_flat():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr, requires_grad=True)
    y = x.repeat(2)
    y.sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    yt = xt.flatten().repeat_interleave(2)
    yt.sum().backward()
    assert x.grad is not None
    check_equality(x.grad, xt.grad.numpy().reshape(arr.shape))


def test_repeat_functional():
    arr = np.random.randn(3, 4).astype(np.float64)
    x = npg.array(arr)
    check_equality(npg.repeat(x, 2, axis=1).data, np.repeat(arr, 2, axis=1))


# --- split ---


@given(split_args())
def test_split_forward(data):
    arr, split_size_or_sections, dim = data
    chunks = npg.split(npg.array(arr), split_size_or_sections, dim=dim)
    torch_chunks = torch.split(torch.from_numpy(arr), split_size_or_sections, dim=dim)
    assert len(chunks) == len(torch_chunks)
    for c, tc in zip(chunks, torch_chunks, strict=True):
        check_equality(c.data, tc.numpy())


@given(split_args(dtypes=FLOAT_DTYPES))
def test_split_backward(data):
    arr, split_size_or_sections, dim = data
    x = npg.array(arr, requires_grad=True)
    chunks = npg.split(x, split_size_or_sections, dim=dim)
    s = chunks[0].sum()
    for c in chunks[1:]:
        s = s + c.sum()
    s.backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    torch_chunks = torch.split(xt, split_size_or_sections, dim=dim)
    st_ = torch.stack([c.sum() for c in torch_chunks]).sum()
    st_.backward()

    check_equality(x.grad, xt.grad.numpy())


def test_split_int_uneven():
    """Last chunk is smaller when size doesn't divide evenly."""
    arr = np.arange(7, dtype=np.float64)
    chunks = npg.split(npg.array(arr), 3, dim=0)
    torch_chunks = torch.split(torch.from_numpy(arr), 3, dim=0)
    assert len(chunks) == len(torch_chunks) == 3
    for c, tc in zip(chunks, torch_chunks, strict=True):
        check_equality(c.data, tc.numpy())


def test_split_list_sections():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    chunks = npg.split(npg.array(arr), [1, 2], dim=1)
    torch_chunks = torch.split(torch.from_numpy(arr), [1, 2], dim=1)
    for c, tc in zip(chunks, torch_chunks, strict=True):
        check_equality(c.data, tc.numpy())


def test_split_method():
    """Array.split method is equivalent to npg.split."""
    arr = np.arange(6, dtype=np.float64)
    x = npg.array(arr)
    pairs = zip(x.split(3), npg.split(x, 3), strict=True)
    assert all(np.array_equal(a.data, b.data) for a, b in pairs)


# --- dual-mode calling conventions ---


def test_view_tuple():
    x = npg.array(np.arange(6, dtype=np.float64))
    check_equality(x.view((2, 3)).data, np.arange(6, dtype=np.float64).reshape(2, 3))


def test_view_unpacked():
    x = npg.array(np.arange(6, dtype=np.float64))
    check_equality(x.view(2, 3).data, np.arange(6, dtype=np.float64).reshape(2, 3))


def test_transpose_tuple():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    check_equality(npg.array(arr).transpose((1, 0)).data, arr.T)


def test_transpose_unpacked():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    check_equality(npg.array(arr).transpose(1, 0).data, arr.T)


def test_npg_reshape_tuple():
    arr = np.arange(6, dtype=np.float64)
    check_equality(npg.reshape(npg.array(arr), (2, 3)).data, arr.reshape(2, 3))


def test_npg_reshape_unpacked():
    arr = np.arange(6, dtype=np.float64)
    check_equality(npg.reshape(npg.array(arr), 2, 3).data, arr.reshape(2, 3))


def test_npg_transpose_tuple():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    check_equality(npg.transpose(npg.array(arr), (1, 0)).data, arr.T)


def test_npg_transpose_unpacked():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    check_equality(npg.transpose(npg.array(arr), 1, 0).data, arr.T)


# --- permute ---


@given(transpose_args(dtypes=FLOAT_DTYPES))
def test_permute_matches_transpose(data):
    arr, axes = data
    x = npg.array(arr)
    check_equality(x.permute(*axes).data, x.transpose(*axes).data)


@given(transpose_args(dtypes=FLOAT_DTYPES))
def test_permute_backward(data):
    arr, axes = data
    x = npg.array(arr, requires_grad=True)
    x.permute(*axes).sum().backward()

    xt = torch.from_numpy(arr).requires_grad_(True)
    xt.permute(*axes).sum().backward()
    check_equality(x.grad, xt.grad.numpy())


def test_npg_permute():
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    check_equality(
        npg.permute(npg.array(arr), 2, 0, 1).data,
        np.transpose(arr, (2, 0, 1)),
    )
