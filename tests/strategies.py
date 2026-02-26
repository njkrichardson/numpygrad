from hypothesis import strategies as st
import numpy as np
import numpy.random as npr

from tests.configuration import (
    VALUE_RANGE,
    FLOAT_DTYPES,
    DTYPES,
    FLOAT_DISTRIBUTION,
    MIN_DIM_SIZE,
    MAX_DIM_SIZE,
    MIN_NUM_DIMS,
    MAX_NUM_DIMS,
)

npr.seed(0)


@st.composite
def shape_nd(draw, min_num_dims: int = MIN_NUM_DIMS, max_num_dims: int = MAX_NUM_DIMS):
    num_dims = draw(st.integers(min_value=min_num_dims, max_value=max_num_dims))
    shape = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE))
        for _ in range(num_dims)
    )
    if len(shape) == 1:
        return shape[0]

    return shape


@st.composite
def generic_array(draw, dtypes=DTYPES):
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
    for shape, dtype in zip(shapes, dtypes):
        if np.issubdtype(dtype, np.integer):
            arr = npr.randint(*VALUE_RANGE, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = FLOAT_DISTRIBUTION(shape).astype(dtype)
        else:
            raise ValueError
        arrays.append(arr)

    return tuple(arrays)


@st.composite
def _array_pair_same_shape(
    draw, min_dims=MIN_NUM_DIMS, max_dims=MAX_NUM_DIMS, dtypes=DTYPES
):
    ndim = draw(st.integers(min_value=min_dims, max_value=max_dims))
    dtype = draw(st.sampled_from(dtypes))
    shape = tuple(
        draw(st.integers(min_value=MIN_DIM_SIZE, max_value=MAX_DIM_SIZE))
        for _ in range(ndim)
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
        assert max_dims >= 3, "Need at least 3 dimensions for batch matmul"
        min_dims = max(min_dims, 3)

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
        # For batch matmul, the last two dimensions must be compatible for multiplication
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
    assert not (same_shape and broadcastable), (
        "Cannot be both same shape and broadcastable"
    )
    assert not (same_shape and mm_broadcastable), (
        "Cannot be both same shape and mm_broadcastable"
    )
    assert not (broadcastable and mm_broadcastable), (
        "Cannot be both broadcastable and mm_broadcastable"
    )
    if same_shape:
        return draw(_array_pair_same_shape(min_dims, max_dims, dtypes))
    elif broadcastable:
        return draw(
            _array_pair_broadcastable(
                min_dims, max_dims, dtypes, mm_broadcastable=False
            )
        )
    elif mm_broadcastable:
        return draw(
            _array_pair_broadcastable(min_dims, max_dims, dtypes, mm_broadcastable)
        )
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
    # if np.issubdtype(dtype, np.integer):
    #     A = npr.randint(*VALUE_RANGE, size=(m, n)).astype(dtype)
    #     x = npr.randint(*VALUE_RANGE, size=(n,)).astype(dtype)
    # elif np.issubdtype(dtype, np.floating):
    #     A = FLOAT_DISTRIBUTION((m, n)).astype(dtype)
    #     x = FLOAT_DISTRIBUTION((n,)).astype(dtype)
    # else:
    #     raise ValueError
    #
    # return (A, x)


@st.composite
def mat_mat_pair(draw):
    (m, k, n) = draw(shape_nd(min_num_dims=3, max_num_dims=3))
    dtype = draw(st.sampled_from(DTYPES))
    return _arrays([(m, k), (k, n)], [dtype, dtype])
    # if np.issubdtype(dtype, np.integer):
    #     A = npr.randint(*VALUE_RANGE, size=(m, k)).astype(dtype)
    #     B = npr.randint(*VALUE_RANGE, size=(k, n)).astype(dtype)
    # elif np.issubdtype(dtype, np.floating):
    #     A = FLOAT_DISTRIBUTION((m, k)).astype(dtype)
    #     B = FLOAT_DISTRIBUTION((k, n)).astype(dtype)
    # else:
    #     raise ValueError
    #
    # return (A, B)


@st.composite
def batch_mm(draw, dtypes=DTYPES):
    return draw(
        _array_pair_broadcastable(
            min_dims=1, max_dims=5, mm_broadcastable=True, dtypes=dtypes
        )
    )
