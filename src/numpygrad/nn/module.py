from typing import Any

from numpygrad.core.array import Array, ArrayCoercible


class Parameter(Array):
    def __init__(self, data: ArrayCoercible):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self._buffers: dict[str, Array] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Array):
            self._buffers[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        elif name in self._buffers:
            return self._buffers[name]
        else:
            raise AttributeError(f"Module has no attribute {name}")

    def add_module(self, name: str, module: "Module") -> None:
        self._modules[name] = module

    def state_dict(self):
        """Walk parameters, buffers, and modules recursively (DFS) and return a
        flat dictionary of names and values.
        """
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        for name, buffer in self._buffers.items():
            state[name] = buffer.data
        return state

    def parameters(self):
        """Return all parameters in this module and submodules (recursive)."""
        result = list(self._parameters.values())
        for module in self._modules.values():
            result.extend(module.parameters())
        return result

    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> Array:
        return self.forward(x)


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(f"module_{i}", module)

    def forward(self, x: Array) -> Array:
        for module in self._modules.values():
            x = module(x)
        return x
