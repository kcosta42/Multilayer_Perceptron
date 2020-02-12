import mlpy.backend.math as M
from mlpy.initializers.initializer import Initializer


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    Args:
        low (float, optional): Low range of random values to generate.
        high (float, optional): Upper range of random values to generate.
        seed (integer, optional): Used to seed the random generator.
    """

    def __init__(self, low=-0.05, high=0.05, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape, dtype=None):
        x = M.random_uniform(shape, self.low, self.high, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x
