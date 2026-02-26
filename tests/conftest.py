import numpy as np
import numpy.random as npr

MIN_NUM_DIMS: int = 0
MAX_NUM_DIMS: int = 5

MIN_DIM_SIZE: int = 1
MAX_DIM_SIZE: int = 8

VALUE_DYNAMIC_RANGE = 32
VALUE_RANGE = (-VALUE_DYNAMIC_RANGE, VALUE_DYNAMIC_RANGE)

DTYPES = (np.float32, np.float64, np.int32, np.int64)


def FLOAT_DISTRIBUTION(shape):
    return npr.uniform(*VALUE_RANGE, size=shape)


EQUAL_TOLERANCE = 1e-5
