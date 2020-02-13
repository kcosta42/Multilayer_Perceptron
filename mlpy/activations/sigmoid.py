import mlpy.backend.math as M
from mlpy.activations.activation import Activation


class Sigmoid(Activation):
    """Sigmoid activation function."""
    def __init__(self):
        super(Sigmoid, self).__init__()

    def call(self, x):
        """Numerically stable Sigmoid activation function.

        Args:
            x (tensor): Input tensor.

        Returns:
            The sigmoid activation: `1 / (1 + exp(-x))`.
        """
        return M.where(x >= 0, 1 / (1 + M.exp(-x)), M.exp(x) / (1 + M.exp(x)))

    def gradient(self, x):
        """Gradient of the Sigmoid activation function."""
        return self.call(x) * (1 - self.call(x))
