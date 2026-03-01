from typing import Any

import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.contexts import is_autograd_active
from numpygrad.ops.core import ensure_array


class Context:
    def __init__(self):
        self._saved_arrays: tuple[Array, ...] = ()
        self._saved_versions: tuple[int, ...] = ()
        # Optional attributes set by individual ops in forward(), read in backward().
        # Using explicit fields so mypy knows they exist on Context.
        self.axis: int | tuple[int, ...] | None = None
        self.axes: tuple[int, ...] | None = None
        self.keepdims: bool = False
        self.key: Any = None  # numpy index: int | slice | ndarray | tuple thereof
        self.num_arrays: int = 0
        self.original_shapes: tuple[tuple[int, ...], ...] = ()
        self.sizes: tuple[int, ...] = ()
        self.count: int | float = 0
        self.a_shape: tuple[int, ...] = ()
        self.b_shape: tuple[int, ...] = ()
        self.squeeze_a: bool = False
        self.squeeze_b: bool = False
        self.reduction: str = ""
        self.targets: np.ndarray | None = None
        self.log_probs: np.ndarray | None = None
        self.stride: tuple[int, int] = (1, 1)
        self.padding: tuple[int, int] = (0, 0)
        self.input_shape: tuple[int, ...] = ()

    def store(self, *arrays) -> None:
        self._saved_arrays = arrays
        self._saved_versions = tuple(a._version if isinstance(a, Array) else -1 for a in arrays)

    @property
    def saved_arrays(self) -> tuple[Array, ...]:
        for arr, v in zip(self._saved_arrays, self._saved_versions, strict=True):
            if isinstance(arr, Array) and v != -1 and arr._version != v:
                raise RuntimeError(
                    "A tensor saved for backward was mutated by an in-place operation."
                )
        return self._saved_arrays


class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        arrays = tuple(ensure_array(a) for a in args)
        requires_grad = any(a.requires_grad for a in arrays) and is_autograd_active()

        if not requires_grad:
            ctx = Context()
            out = cls.forward(ctx, *args, **kwargs)

            out.requires_grad = False
            out.grad_fn = None
            out.parents = ()
            return out

        ctx = Context()
        out = cls.forward(ctx, *args)
        out.grad_fn = cls
        out.ctx = ctx
        out.parents = args
        out.requires_grad = True

        return out
