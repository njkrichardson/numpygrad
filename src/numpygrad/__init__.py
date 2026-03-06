from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("numpygrad")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from numpy import e, inf, nan, pi

import numpygrad.configuration as configuration
import numpygrad.optim as optim
import numpygrad.random as random
import numpygrad.utils as utils
from numpygrad.core import array, ndarray
from numpygrad.core.array_creation import (
    arange,
    empty_like,
    full,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from numpygrad.core.contexts import no_grad
from numpygrad.core.dtypes import float32, float64, int32, int64
from numpygrad.core.random import manual_seed
from numpygrad.ops import (
    abs,
    add,
    argmax,
    argmin,
    cat,
    ceil,
    clip,
    concatenate,
    conv2d,
    copy,
    cos,
    cumprod,
    cumsum,
    diagonal,
    dot,
    exp,
    expand_dims,
    flatten,
    floor,
    gelu,
    isfinite,
    isinf,
    isnan,
    log,
    log_softmax,
    masked_fill,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mm,
    mul,
    norm,
    permute,
    pow,
    prod,
    relu,
    repeat,
    reshape,
    setitem,
    sigmoid,
    sign,
    sin,
    softmax,
    softplus,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sum,
    tan,
    tanh,
    trace,
    transpose,
    triu,
    unsqueeze,
    var,
    where,
)
from numpygrad.utils import Log, io

newaxis = None

__all__ = [
    "__version__",
    "array",
    "ndarray",
    "configuration",
    "io",
    "Log",
    "utils",
    "manual_seed",
    "random",
    # array creation
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full",
    "linspace",
    "arange",
    # constants
    "pi",
    "e",
    "inf",
    "nan",
    "newaxis",
    # dtypes
    "float32",
    "float64",
    "int32",
    "int64",
    # optim/utils
    "optim",
    "no_grad",
    # elementwise ops
    "exp",
    "pow",
    "log",
    "abs",
    "add",
    "mul",
    "relu",
    "clip",
    "maximum",
    "minimum",
    "sin",
    "cos",
    "tan",
    "floor",
    "ceil",
    "sign",
    "copy",
    "where",
    # predicates
    "isnan",
    "isinf",
    "isfinite",
    # reductions
    "sum",
    "mean",
    "prod",
    "min",
    "max",
    "argmax",
    "argmin",
    "var",
    "std",
    "cumsum",
    "cumprod",
    # linear algebra
    "mm",
    "matmul",
    "dot",
    "norm",
    # transforms
    "flatten",
    "unsqueeze",
    "expand_dims",
    "reshape",
    "transpose",
    "permute",
    "triu",
    "squeeze",
    "repeat",
    "diagonal",
    "trace",
    "sqrt",
    "split",
    "stack",
    "cat",
    "concatenate",
    # activations
    "softmax",
    "log_softmax",
    "sigmoid",
    "tanh",
    "softplus",
    "gelu",
    # convolution
    "conv2d",
    # special
    "masked_fill",
    "setitem",
]
