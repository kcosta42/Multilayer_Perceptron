import libft.backend.math as M
from libft.activations.activation import Activation


class Linear(Activation):
    """Linear Activation."""
    def __init__(self):
        super(Linear, self).__init__()

    def call(self, x):
        """Linear (i.e. identity) activation function.

        Arguments:
            x: array-like
                Input tensor.

        Returns:
            Input tensor, unchanged.
        """
        return x

    def gradient(self, x):
        """Gradient of Linear activation function."""
        return M.ones(x.shape)
