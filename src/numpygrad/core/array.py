import numpy as np

from numpygrad.core.device import DeviceId
from numpygrad.core.dispatch import dispatch
from numpygrad.core.opid import OperatorId

type ArrayCoercible = "np.ndarray | int | float | \
list[int | float] | tuple[int | float] | Array | tuple[int, ...]"


class Array:
    def __init__(
        self,
        data: ArrayCoercible,
        *,
        device: DeviceId | str = "cpu_np",
        requires_grad: bool = False,
        label: str = "",
        dtype: np.dtype | None = None,
    ):
        if isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass
        elif isinstance(data, Array):
            # Unwrap so self.data is always ndarray and zeros_like(grad) gets
            # correct shape
            data = data.data

        self.data: np.ndarray = data  # type: ignore
        if dtype is not None:
            self.data = self.data.astype(dtype)

        self.device: DeviceId = DeviceId(device)

        if self.dtype != np.float32 and self.dtype != np.float64 and requires_grad:
            raise ValueError("Only arrays of floating point dtype can require gradients")
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self.grad_fn = None
        self.ctx = None
        self.parents = ()
        self.label = label

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return self.data.item()

    def __repr__(self) -> str:
        out = f"Array(data={self.data}"
        if self.requires_grad:
            out += ", requires_grad=True"
            if self.grad_fn is not None:
                out += f", grad_fn={self.grad_fn.__name__}"
        if self.label != "":
            out += f", label={self.label}"
        out += ")"
        return out

    def __getitem__(self, key) -> "Array":
        return dispatch(OperatorId.SLICE, self, key=key)

    def __setitem__(
        self, key: "ArrayCoercible | slice | tuple[slice, ...]", value: ArrayCoercible
    ) -> None:
        # In-place mutation on Arrays that participate in autograd is not supported.
        # Use the functional setitem op instead (numpygrad.ops.setitem or
        # Array.setitem).
        from numpygrad.ops.core import (
            ensure_array,
        )  # local import to avoid circular dependency

        if self.requires_grad or ensure_array(value).requires_grad:
            raise RuntimeError(
                "__setitem__ on Arrays that require grad is not supported; "
                "use a functional setitem operation instead."
            )

        result = dispatch(OperatorId.SETITEM, self, key, value)
        self.data = result.data

    def setitem(self, key: "ArrayCoercible | slice | tuple[slice, ...]", value: ArrayCoercible):
        return dispatch(OperatorId.SETITEM, self, key, value)

    def __gt__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.GT, self, other)

    def __lt__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.LT, self, other)

    def __ge__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.GE, self, other)

    def __le__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.LE, self, other)

    def __eq__(self, other: ArrayCoercible) -> "Array":  # type: ignore
        return dispatch(OperatorId.EQ, self, other)

    def __ne__(self, other: ArrayCoercible) -> "Array":  # type: ignore
        return dispatch(OperatorId.NE, self, other)

    def __mul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MUL, self, other)

    def __add__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.ADD, self, other)

    def __rmul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MUL, self, other)

    def __neg__(self) -> "Array":
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.POW, self, other)

    def __truediv__(self, other: "int | float | Array | np.ndarray") -> "Array":
        return self * other**-1

    def __matmul__(self, other: ArrayCoercible) -> "Array":
        return dispatch(OperatorId.MATMUL, self, other)

    def sum(self, axis=None, keepdims=False) -> "Array":
        return dispatch(OperatorId.SUM, self, axis=axis, keepdims=keepdims)

    def exp(self) -> "Array":
        return dispatch(OperatorId.EXP, self)

    def log(self) -> "Array":
        return dispatch(OperatorId.LOG, self)

    def abs(self) -> "Array":
        return dispatch(OperatorId.ABS, self)

    def transpose(self, axes: tuple[int, ...]) -> "Array":
        return dispatch(OperatorId.TRANSPOSE, self, axes=axes)

    def reshape(self, *new_shape) -> "Array":
        shape = (
            new_shape[0]
            if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list))
            else new_shape
        )
        return dispatch(OperatorId.RESHAPE, self, new_shape=shape)

    def view(self, new_shape: tuple[int, ...] | int) -> "Array":
        return self.reshape(new_shape)

    def mean(self, axis: tuple[int, ...] | int | None = None, keepdims: bool = False) -> "Array":
        return dispatch(OperatorId.MEAN, self, axis=axis, keepdims=keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> "Array":
        return dispatch(OperatorId.MAX, self, axis=axis, keepdims=keepdims)

    def min(self, axis: int | None = None, keepdims: bool = False) -> "Array":
        return dispatch(OperatorId.MIN, self, axis=axis, keepdims=keepdims)

    def prod(self, axis: int | None = None, keepdims: bool = False) -> "Array":
        return dispatch(OperatorId.PRODUCT, self, axis=axis, keepdims=keepdims)

    def argmax(self, axis: int | None = None, keepdims: bool = False) -> "Array":
        return dispatch(OperatorId.ARGMAX, self, axis=axis, keepdims=keepdims)

    @property
    def T(self) -> "Array":
        return self.transpose(axes=tuple(reversed(range(self.ndim))))

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)

        # build topo order
        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in getattr(node, "parents", ()):
                    build(parent)
                topo.append(node)

        build(self)

        # seed gradient
        self.grad = grad

        # backward pass
        for node in reversed(topo):
            if getattr(node, "grad_fn", None) is None:
                continue

            grads = node.grad_fn.backward(node.ctx, node.grad)

            for parent, parent_grad in zip(node.parents, grads, strict=False):
                if getattr(parent, "requires_grad", False):
                    parent.grad += parent_grad
