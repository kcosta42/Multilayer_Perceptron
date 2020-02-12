import mlpy.backend.math as M
from mlpy.activations.activation import Activation


class ReLU(Activation):
    """Rectified Linear Unit.

    It returns element-wise `max(alpha * x, x)`.

    Args:
        alpha (float, optional): Slope of the negative part. Defaults to zero.
    """
    def __init__(self, alpha=0.0):
        super(ReLU, self).__init__()
        self.alpha = alpha

    def call(self, x):
        """Rectified Linear Unit.

        Args:
            x (tensor): Input tensor.

        Returns:
            A tensor.
        """
        # return M.max(self.alpha * x, x)
        return M.where(x < 0, self.alpha * x, x)

    def gradient(self, x):
        """Gradient of ReLU activation function."""
        return M.where(x < 0, self.alpha, 1)
