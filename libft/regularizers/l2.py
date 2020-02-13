import libft.backend.math as M
from libft.regularizers.regularizer import Regularizer


class L2(Regularizer):
    """Regularizer for L2 regularization.

    Arguments:
        l2: float, Default: 0.01
            L2 regularization factor.
    """
    def __init__(self, l2=0.01):
        self.l2 = l2

    def call(self, x):
        regularization = self.l2 * M.sum(M.square(x))
        return regularization

    def gradient(self, x):
        return 2 * self.l2 * x
