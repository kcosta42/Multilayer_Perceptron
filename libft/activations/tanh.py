import libft.backend.math as M
from libft.activations.activation import Activation


class TanH(Activation):
    """Hyperbolic tangent activation function."""
    def __init__(self):
        super(TanH, self).__init__()

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
