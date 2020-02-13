import libft.backend.math as M
from libft.initializers.initializer import Initializer


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""
    def __call__(self, shape, dtype=None):
        return M.constant(1, shape=shape, dtype=dtype)
