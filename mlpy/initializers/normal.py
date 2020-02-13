import mlpy.backend.math as M
from mlpy.initializers.initializer import Initializer


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Arguments:
        mean: float, Default: 0.0
            Mean of the random values to generate.
        stddev: float, Default: 0.05
            Std dev of the random values to generate.
        seed: integer, Default: None
            Used to seed the random generator.
    """
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=None):
        x = M.random_normal(shape, self.mean, self.stddev, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x
