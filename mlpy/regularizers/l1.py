from mlpy.regularizers.regularizer import Regularizer
import mlpy.backend.math as M


class L1(Regularizer):
    """Regularizer for L1 regularization.

    Arguments:
        l1: float, Default: 0.01
            L1 regularization factor.
    """
    def __init__(self, l1=0.01):
        self.l1 = l1

    def call(self, x):
        regularization = self.l1 * M.sum(M.abs(x))
        return regularization

    def gradient(self, x):
        return self.l1 * M.sign(x)
