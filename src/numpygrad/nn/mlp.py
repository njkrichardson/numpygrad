from numpygrad.core.array import Array
from numpygrad.nn.activations import ReLU, Sigmoid, Tanh
from numpygrad.nn.linear import Linear
from numpygrad.nn.module import Module, Sequential


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        dims = [input_dim] + hidden_sizes + [output_dim]

        if activation == "relu":
            act = ReLU()
        elif activation == "tanh":
            act = Tanh()
        elif activation == "sigmoid":
            act = Sigmoid()
        else:
            raise ValueError(f"Activation {activation} not supported")

        layers: list[Module] = []
        pairs = list(zip(dims[:-1], dims[1:], strict=False))
        for i, (nin, nout) in enumerate(pairs):
            layers.append(Linear(nin, nout))
            if i < len(pairs) - 1:
                layers.append(act)
        self.layers = Sequential(*layers)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def forward(self, x: Array) -> Array:
        return self.layers(x)
