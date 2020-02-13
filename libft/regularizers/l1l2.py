import libft.backend.math as M
from libft.regularizers.regularizer import Regularizer


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    Arguments:
        l1: float, Default: 0.01
            L1 regularization factor.
        l2: float, Default: 0.01
            L2 regularization factor.
    """
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def call(self, x):
        regularization = 0.0
        regularization += self.l1 * M.sum(M.abs(x))
        regularization += self.l2 * M.sum(M.square(x))
        return regularization

    def gradient(self, x):
        regularization = 0.0
        regularization += self.l1 * M.sign(x)
        regularization += 2 * self.l2 * x
        return regularization
