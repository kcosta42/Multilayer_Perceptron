import mlpy.backend.math as M
from mlpy.activations.activation import Activation


class Softmax(Activation):
    """Softmax activation function."""
    def __init__(self):
        super(Softmax, self).__init__()

    def call(self, x):
        """Softmax activation function.

        Args:
            x: Input tensor.

        Returns:
            Tensor, output of softmax transformation.
        """
        exps = M.exp(x - M.max(x, axis=-1, keepdims=True))
        return exps / M.sum(exps, axis=-1, keepdims=True)

    def gradient(self, x):
        """Gradient of Softmax activation function."""
        p = self.call(x)
        return p * (1 - p)
