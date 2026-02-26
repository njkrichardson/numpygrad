import enum


class OperatorId(enum.StrEnum):
    ADD = enum.auto()
    MUL = enum.auto()
    POW = enum.auto()
    SUM = enum.auto()
    MEAN = enum.auto()
    MATMUL = enum.auto()
    RELU = enum.auto()
    TRANSPOSE = enum.auto()
    RESHAPE = enum.auto()
