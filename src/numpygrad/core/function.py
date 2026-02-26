from numpygrad.core.array import Array


class Context:
    def __init__(self):
        self.saved_arrays: tuple[Array, ...] = ()

    def store(self, *arrays) -> None:
        self.saved_arrays = arrays


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()

        out = cls.forward(ctx, *args)

        out.grad_fn = cls
        out.ctx = ctx
        out.parents = args
        out.requires_grad = True

        return out
