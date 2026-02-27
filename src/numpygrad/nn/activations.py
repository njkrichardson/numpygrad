from numpygrad.nn.module import Module
from numpygrad.core.array import Array
from numpygrad.ops import relu

class ReLU(Module):
    def forward(self, x: Array) -> Array:
        return relu(x)