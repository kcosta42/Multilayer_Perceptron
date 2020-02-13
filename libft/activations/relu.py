import libft.backend.math as M
from libft.activations.activation import Activation


class ReLU(Activation):
    """Rectified Linear Unit.

    It returns element-wise `max(alpha * x, x)`.

    Arguments:
        alpha: float, Default: 0.0
            Slope of the negative part.
    """
    def __init__(self, alpha=0.0):
        super(ReLU, self).__init__()
        self.alpha = alpha

    def call(self, x):
        """Rectified Linear Unit.

        Arguments:
            x: array-like
                Input tensor.

        Returns:
            A tensor.
        """
        # return M.max(self.alpha * x, x)
        return M.where(x < 0, self.alpha * x, x)

    def gradient(self, x):
        """Gradient of ReLU activation function."""
        return M.where(x < 0, self.alpha, 1)
