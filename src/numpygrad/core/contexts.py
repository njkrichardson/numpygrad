from contextlib import contextmanager

_autograd_active = True


def is_autograd_active() -> bool:
    return _autograd_active


@contextmanager
def no_grad():
    global _autograd_active
    previous_state = _autograd_active
    _autograd_active = False
    try:
        yield
    finally:
        _autograd_active = previous_state
