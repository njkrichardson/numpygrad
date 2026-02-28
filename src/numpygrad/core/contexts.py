_autograd_active = True


def is_autograd_active() -> bool:
    return _autograd_active


def set_autograd_active(active: bool):
    global _autograd_active
    _autograd_active = active


class NoGradContext:
    def __enter__(self):
        self.previous_state = is_autograd_active()
        set_autograd_active(False)

    def __exit__(self, exc_type, exc_value, traceback):
        set_autograd_active(self.previous_state)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


no_grad = NoGradContext
