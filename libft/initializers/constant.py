import libft.backend.math as M
from libft.initializers.initializer import Initializer


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    Arguments:
        value: float, Default: 0
            The value of the generator tensors.
    """
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return M.constant(self.value, shape=shape, dtype=dtype)
