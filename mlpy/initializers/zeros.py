import mlpy.backend.math as M
from mlpy.initializers.initializer import Initializer


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""
    def __call__(self, shape, dtype=None):
        return M.constant(0, shape=shape, dtype=dtype)
