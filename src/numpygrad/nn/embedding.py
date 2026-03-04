import numpy as np

from numpygrad.core.array import Array
from numpygrad.nn.module import Module, Parameter
from numpygrad.ops import embedding as embedding_op


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            Array(np.random.randn(num_embeddings, embedding_dim), requires_grad=True)
        )

    def forward(self, indices: Array) -> Array:
        return embedding_op(self.weight, indices)

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
