import mlpy.backend.math as M
from mlpy.activations.activation import Activation


class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, x):
        """Hyperbolic tangent activation function.

        Arguments:
            x: array-like
                Input tensor.

        Returns:
            The hyperbolic activation:
            `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
        """
        return (M.exp(x) - M.exp(-x)) / (M.exp(x) + M.exp(-x))

    def gradient(self, x):
        """Gradient of the Tanh activation function."""
        return 1 - self.call(x)**2
