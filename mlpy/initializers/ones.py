import mlpy.backend.math as M
from mlpy.initializers.initializer import Initializer


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""
    def __call__(self, shape, dtype=None):
        return M.constant(1, shape=shape, dtype=dtype)
