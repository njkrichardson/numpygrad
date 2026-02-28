import dataclasses
import enum
from collections.abc import Callable

from numpygrad.core.device import DeviceId
from numpygrad.core.opid import OperatorId


class OperatorRequirements(enum.StrEnum):
    ForwardOnly = enum.auto()
    Autograd = enum.auto()


@dataclasses.dataclass
class OperatorRegistry:
    registry: dict[OperatorId, dict[DeviceId, dict[OperatorRequirements, Callable]]] = (
        dataclasses.field(default_factory=dict)
    )


_Registry = OperatorRegistry()
Registry = _Registry.registry


def register(
    opid: OperatorId,
    device_id: DeviceId = DeviceId.cpu_np,
    op_requirements: OperatorRequirements = OperatorRequirements.ForwardOnly,
) -> Callable:
    def decorator(f: Callable):
        Registry.setdefault(opid, dict())
        Registry[opid].setdefault(device_id, {})
        Registry[opid][device_id][op_requirements] = f
        return f

    return decorator
