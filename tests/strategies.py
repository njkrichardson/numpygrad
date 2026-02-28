import numpy as np
import numpy.random as npr
from hypothesis import strategies as st

from tests.configuration import (
    DTYPES,
    FLOAT_DISTRIBUTION,
    FLOAT_DTYPES,
    MAX_DIM_SIZE,
    MAX_NUM_DIMS,
    MIN_DIM_SIZE,
    MIN_NUM_DIMS,
    VALUE_RANGE,
)

npr.seed(0)


@st.composite
def shape_nd(draw, min_num_dims: int = MIN_NUM_DIMS, max_num_dims: int = MAX_NUM_DIMS):
    num_dims = draw(st.integers(min_value=min_num_dims, max_value=max_num_dims))
    shape = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)) for _ in range(num_dims)
    )
    if len(shape) == 1:
        return shape[0]

    return shape


@st.composite
def generic_array(draw, shape: tuple[int, ...] | None = None, dtypes=DTYPES):
    if shape is None:
        shape = draw(shape_nd())
    dtype = draw(st.sampled_from(dtypes))
    if np.issubdtype(dtype, np.integer):
        arr = npr.randint(*VALUE_RANGE, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        arr = FLOAT_DISTRIBUTION(shape).astype(dtype)
    else:
        raise ValueError

    return arr


def _arrays(shapes: list[tuple[int, ...]], dtypes: list[np.dtype]):
    arrays = []
    for shape, dtype in zip(shapes, dtypes, strict=False):
        if np.issubdtype(dtype, np.integer):
            arr = npr.randint(*VALUE_RANGE, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = FLOAT_DISTRIBUTION(shape).astype(dtype)
        else:
            raise ValueError
        arrays.append(arr)

    return tuple(arrays)


@st.composite
def _array_pair_same_shape(draw, min_dims=MIN_NUM_DIMS, max_dims=MAX_NUM_DIMS, dtypes=DTYPES):
    ndim = draw(st.integers(min_value=min_dims, max_value=max_dims))
    dtype = draw(st.sampled_from(dtypes))
    shape = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)) for _ in range(ndim)
    )

    if np.issubdtype(dtype, np.integer):
        A = npr.randint(*VALUE_RANGE, size=shape).astype(dtype)
        B = npr.randint(*VALUE_RANGE, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        A = FLOAT_DISTRIBUTION(shape).astype(dtype)
        B = FLOAT_DISTRIBUTION(shape).astype(dtype)
    else:
        raise ValueError

    return A, B


@st.composite
def _array_pair_broadcastable(
    draw,
    min_dims=MIN_NUM_DIMS,
    max_dims=MAX_NUM_DIMS,
    dtypes=DTYPES,
    mm_broadcastable=False,
):
    if mm_broadcastable:
        assert max_dims >= 2
        min_dims = max(min_dims, 2)

    ndim = draw(st.integers(min_value=min_dims, max_value=max_dims))
    dtype = draw(st.sampled_from(dtypes))

    shape_A = []
    shape_B = []

    for d in range(ndim):
        if mm_broadcastable and d >= ndim - 2:
            break

        use_same = draw(st.booleans())  # axis is same in both arrays
        if use_same:
            size = draw(st.integers(MIN_DIM_SIZE, MAX_DIM_SIZE))
            shape_A.append(size)
            shape_B.append(size)
        else:
            shape_A.append(draw(st.integers(MIN_DIM_SIZE, MAX_DIM_SIZE)))
            shape_B.append(1)  # singleton -> broadcastable

    if mm_broadcastable:
        # For batch matmul, the last two dimensions must be compatible
        # for multiplication
        (m, k, n) = draw(shape_nd(min_num_dims=3, max_num_dims=3))
        shape_A.extend([m, k])
        shape_B.extend([k, n])

    return _arrays([tuple(shape_A), tuple(shape_B)], [dtype, dtype])


@st.composite
def array_pair(
    draw,
    min_dims=MIN_NUM_DIMS,
    max_dims=MAX_NUM_DIMS,
    dtypes=DTYPES,
    broadcastable=False,
    mm_broadcastable=False,
    same_shape=False,
):
    assert not (same_shape and broadcastable), "Cannot be both same shape and broadcastable"
    assert not (same_shape and mm_broadcastable), "Cannot be both same shape and mm_broadcastable"
    assert not (broadcastable and mm_broadcastable), (
        "Cannot be both broadcastable and mm_broadcastable"
    )
    if same_shape:
        return draw(_array_pair_same_shape(min_dims, max_dims, dtypes))
    elif broadcastable:
        return draw(_array_pair_broadcastable(min_dims, max_dims, dtypes, mm_broadcastable=False))
    elif mm_broadcastable:
        return draw(_array_pair_broadcastable(min_dims, max_dims, dtypes, mm_broadcastable))
    else:
        dtype = draw(st.sampled_from(dtypes))
        shape_A = draw(shape_nd(min_dims, max_dims))
        shape_B = draw(shape_nd(min_dims, max_dims))
        return _arrays([shape_A, shape_B], [dtype, dtype])


@st.composite
def mat_vec_pair(draw):
    (m, n) = draw(shape_nd(min_num_dims=2, max_num_dims=2))
    dtype = draw(st.sampled_from(DTYPES))
    return _arrays([(m, n), (n,)], [dtype, dtype])


@st.composite
def mat_mat_pair(draw):
    (m, k, n) = draw(shape_nd(min_num_dims=3, max_num_dims=3))
    dtype = draw(st.sampled_from(DTYPES))
    return _arrays([(m, k), (k, n)], [dtype, dtype])


@st.composite
def dot_1d_pair(draw, dtypes=DTYPES):
    """Two 1D arrays of the same length for np.dot(a, b) (inner product)."""
    n = draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE))
    dtype = draw(st.sampled_from(dtypes))
    return _arrays([(n,), (n,)], [dtype, dtype])


@st.composite
def batch_mm(draw, dtypes=DTYPES):
    return draw(
        _array_pair_broadcastable(min_dims=1, max_dims=5, mm_broadcastable=True, dtypes=dtypes)
    )


@st.composite
def positive_array(draw, shape: tuple[int, ...] | None = None, dtypes=FLOAT_DTYPES):
    """Array with strictly positive values (e.g. for log)."""
    if shape is None:
        shape = draw(shape_nd(min_num_dims=1))  # avoid 0-dim (scalar) for torch compat
        if isinstance(shape, int):
            shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    return np.abs(arr) + 1e-6


@st.composite
def prod_safe_array(draw, shape: tuple[int, ...] | None = None, dtypes=FLOAT_DTYPES):
    """Array with values in (0.1, 1.0] so that product over many elements does not
    overflow.
    """
    if shape is None:
        shape = draw(shape_nd(min_num_dims=1))
        if isinstance(shape, int):
            shape = (shape,)
    dtype = draw(st.sampled_from(dtypes))
    return npr.uniform(0.1, 1.0, size=shape).astype(dtype)


def axes_permutation(ndim: int):
    """Strategy that returns a permutation of range(ndim)."""
    return st.permutation(list(range(ndim))).map(tuple)


@st.composite
def reshape_shape(draw, shape: tuple[int, ...]):
    """Strategy that returns a new shape with same total size as shape."""
    size = int(np.prod(shape))
    if size == 0:
        return shape
    ndim = len(shape)
    # Simple options: same, reversed, or flattened
    options = [shape, tuple(reversed(shape)), (size,)]
    if ndim == 2:
        options.append((shape[1], shape[0]))
    return draw(st.sampled_from(options))


@st.composite
def slice_key(draw, shape: tuple[int, ...]):
    """Strategy that returns a valid indexing key for the given shape."""
    if len(shape) == 0:
        return ()
    keys = []
    for s in shape:
        key = draw(
            st.one_of(
                st.integers(0, max(0, s - 1)),
                st.slices(s),
            )
        )
        keys.append(key)
    return tuple(keys)


@st.composite
def slice_key_positive_step(draw, shape: tuple[int, ...]):
    """Indexing key that only uses integer or slice with step=1 (torch-compatible)."""
    if len(shape) == 0:
        return ()
    keys = []
    for s in shape:
        key = draw(
            st.one_of(
                st.integers(0, max(0, s - 1)),
                st.just(slice(None)),
                st.builds(
                    lambda start, stop: slice(start, stop),
                    st.integers(0, max(0, s - 1)),
                    st.integers(1, max(1, s)),
                ).filter(lambda sl: sl.start < sl.stop),
            )
        )
        keys.append(key)
    return tuple(keys)


@st.composite
def stack_arrays(draw, n: int = 2, min_dims=MIN_NUM_DIMS, max_dims=MAX_NUM_DIMS, dtypes=DTYPES):
    """Strategy that returns (arrays_tuple, axis) for npg.stack."""
    shape = draw(shape_nd(min_num_dims=min_dims, max_num_dims=max_dims))
    if isinstance(shape, int):
        shape = (shape,)
    arrays = _arrays([shape] * n, [draw(st.sampled_from(dtypes))] * n)
    axis = draw(st.integers(0, len(shape)))
    return arrays, axis


@st.composite
def cat_arrays(draw, n: int = 2, min_dims=1, max_dims=MAX_NUM_DIMS, dtypes=DTYPES):
    """Strategy that returns (arrays_tuple, axis) for npg.cat."""
    ndim = draw(st.integers(min_value=min_dims, max_value=max_dims))
    axis = draw(st.integers(0, ndim - 1))
    base_shape = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)) for _ in range(ndim)
    )
    dtype = draw(st.sampled_from(dtypes))
    sizes_on_axis = [
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE)) for _ in range(n)
    ]
    shapes = [
        tuple(base_shape[j] if j != axis else sizes_on_axis[i] for j in range(ndim))
        for i in range(n)
    ]
    return _arrays(shapes, [dtype] * n), axis


@st.composite
def reduction_args(draw, dtypes=DTYPES, with_axis=True):
    """Strategy yielding (arr, axis, keepdims) for reduction ops. arr has ndim >= 1."""
    shape = draw(shape_nd(min_num_dims=1))
    if isinstance(shape, int):
        shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    ndim = len(shape)
    if with_axis:
        axis = draw(st.one_of(st.none(), st.integers(0, ndim - 1)))
    else:
        axis = None
    keepdims = draw(st.booleans())
    return arr, axis, keepdims


@st.composite
def transpose_args(draw, dtypes=DTYPES):
    """Strategy yielding (arr, axes) for transpose. arr has ndim >= 1."""
    shape = draw(shape_nd(min_num_dims=1))
    if isinstance(shape, int):
        shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    axes = tuple(draw(st.permutations(list(range(len(shape))))))
    return arr, axes


@st.composite
def reshape_args(draw, dtypes=DTYPES):
    """Strategy yielding (arr, new_shape) for reshape. new_shape has same size as arr."""
    shape = draw(shape_nd(min_num_dims=1))
    if isinstance(shape, int):
        shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    new_shape = draw(reshape_shape(shape))
    return arr, new_shape


@st.composite
def slice_args(draw, dtypes=DTYPES, allow_negative_step=True):
    """Strategy yielding (arr, key) for slice. key is valid for arr.shape."""
    shape = draw(shape_nd(min_num_dims=1))
    if isinstance(shape, int):
        shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    key = draw(slice_key(shape) if allow_negative_step else slice_key_positive_step(shape))
    return arr, key


@st.composite
def unsqueeze_args(draw, dtypes=DTYPES):
    """Strategy yielding (arr, axis) for unsqueeze. axis in 0..ndim inclusive."""
    shape = draw(shape_nd(min_num_dims=1))
    if isinstance(shape, int):
        shape = (shape,)
    arr = draw(generic_array(shape=shape, dtypes=dtypes))
    ndim = len(shape)
    axis = draw(st.integers(0, ndim))
    return arr, axis
