import enum


class OperatorId(enum.StrEnum):
    # elementwise
    ADD = enum.auto()
    MUL = enum.auto()
    POW = enum.auto()
    EXP = enum.auto()
    LOG = enum.auto()
    ABS = enum.auto()
    CLIP = enum.auto()
    MAXIMUM = enum.auto()
    MINIMUM = enum.auto()
    RELU = enum.auto()

    # reductions
    SUM = enum.auto()
    MEAN = enum.auto()
    MAX = enum.auto()
    MIN = enum.auto()
    PRODUCT = enum.auto()
    ARGMAX = enum.auto()

    # linear algebra
    MATMUL = enum.auto()
    DOT = enum.auto()
    NORM = enum.auto()

    # activations
    SOFTMAX = enum.auto()
    LOG_SOFTMAX = enum.auto()
    SIGMOID = enum.auto()
    TANH = enum.auto()
    SOFTPLUS = enum.auto()

    # transforms
    TRANSPOSE = enum.auto()
    RESHAPE = enum.auto()
    SLICE = enum.auto()
    UNSQUEEZE = enum.auto()
    FLATTEN = enum.auto()
    EXPAND = enum.auto()
    STACK = enum.auto()
    CAT = enum.auto()

    # special methods
    GT = enum.auto()
    LT = enum.auto()
    GE = enum.auto()
    LE = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()
    SETITEM = enum.auto()
