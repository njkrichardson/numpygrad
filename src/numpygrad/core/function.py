from typing import Any

from numpygrad.core.array import Array
from numpygrad.core.contexts import is_autograd_active
from numpygrad.ops.core import ensure_array


class Context:
    def __init__(self):
        self.saved_arrays: tuple[Array, ...] = ()
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
        self.a: Array | None = None
        self.b: Array | None = None
        self.a_shape: tuple[int, ...] = ()
        self.b_shape: tuple[int, ...] = ()
        self.squeeze_a: bool = False
        self.squeeze_b: bool = False

    def store(self, *arrays) -> None:
        self.saved_arrays = arrays


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
