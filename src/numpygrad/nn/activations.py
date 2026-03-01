from numpygrad.core.array import Array
from numpygrad.nn.module import Module
from numpygrad.ops import relu, sigmoid, softplus
from numpygrad.ops import tanh as tanh_op


class ReLU(Module):
    def forward(self, x: Array) -> Array:
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Array) -> Array:
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x: Array) -> Array:
        return tanh_op(x)


class SoftPlus(Module):
    def forward(self, x: Array) -> Array:
        return softplus(x)
