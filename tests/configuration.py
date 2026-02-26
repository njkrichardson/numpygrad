import numpy as np
import numpy.random as npr

MIN_NUM_DIMS: int = 0
MAX_NUM_DIMS: int = 5

MIN_DIM_SIZE: int = 1
MAX_DIM_SIZE: int = 8

VALUE_DYNAMIC_RANGE = 32
VALUE_RANGE = (-VALUE_DYNAMIC_RANGE, VALUE_DYNAMIC_RANGE)
POW_DYNAMIC_RANGE = 4
POW_RANGE = (-POW_DYNAMIC_RANGE, POW_DYNAMIC_RANGE)

INT_DTYPES = (np.int32, np.int64)
FLOAT_DTYPES = (np.float64,)
DTYPES = FLOAT_DTYPES + INT_DTYPES


def FLOAT_DISTRIBUTION(shape):
    return npr.uniform(*VALUE_RANGE, size=shape)


FP32_EQUAL_TOLERANCE = 1e-5
FP64_EQUAL_TOLERANCE = 1e-12


def check_equality(x: np.ndarray, desired: np.ndarray, *, rtol: float | None = None, atol: float | None = None):
    if np.issubdtype(x.dtype, np.integer):
        np.testing.assert_array_equal(x, desired)
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(desired.dtype)  # handle upcasting
        if x.dtype == np.float32:
            EQUAL_TOLERANCE = FP32_EQUAL_TOLERANCE
        elif x.dtype == np.float64:
            EQUAL_TOLERANCE = FP64_EQUAL_TOLERANCE
        else:
            raise ValueError(f"Unsupported dtype: {x.dtype}")
        tol = rtol if rtol is not None else EQUAL_TOLERANCE
        atol_ = atol if atol is not None else tol
        np.testing.assert_allclose(
            x, desired, rtol=tol, atol=atol_
        )
