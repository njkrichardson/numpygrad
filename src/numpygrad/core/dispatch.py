from numpygrad.core.device import DeviceId
from numpygrad.core.opid import OperatorId
from numpygrad.core.registry import OperatorRequirements, Registry


def dispatch(op_id: OperatorId, *args, **kwargs):
    from numpygrad.core.array import Array

    arrays = [x for x in args if isinstance(x, Array)]
    if not arrays and "arrays" in kwargs:
        arrays_arg = kwargs["arrays"]
        assert isinstance(arrays_arg, (list, tuple))
        arrays = [x for x in arrays_arg if isinstance(x, Array)]
    device: DeviceId = arrays[0].device
    requires_grad = any(array.requires_grad for array in arrays)
    direction = OperatorRequirements.Autograd if requires_grad else OperatorRequirements.ForwardOnly
    _implementation = Registry[op_id][device][direction]
    return _implementation(*args, **kwargs)
