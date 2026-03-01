import numpy as np

from numpygrad.core.array import Array
from numpygrad.core.array_creation import zeros
from numpygrad.nn.module import Module, Parameter
from numpygrad.ops import conv2d


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: tuple[int, int] = (kH, kW)
        self.stride: tuple[int, int] = (stride, stride) if isinstance(stride, int) else stride
        self.padding: tuple[int, int] = (padding, padding) if isinstance(padding, int) else padding

        fan_in = in_channels * kH * kW
        self.weight = Parameter(
            Array(
                np.random.randn(out_channels, in_channels, kH, kW) * np.sqrt(2 / fan_in),
                requires_grad=True,
            )
        )
        if bias:
            self.bias: Parameter | None = Parameter(zeros(out_channels, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: Array) -> Array:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}, "
            f"dilation=1, groups=1 [not supported])"
        )
