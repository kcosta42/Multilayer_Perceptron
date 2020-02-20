import libft.backend.math as M
from libft.layers.layer import Layer


class Dropout(Layer):
    """Applies Dropout to the input.

    Arguments:
        rate: float
            Fraction of the input units to drop.

    References:
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = M.min(1.0, M.max(0.0, rate))
        self.mask = None

    def call(self, inputs):
        mask = (1 - self.rate)
        if self.trainable:
            self.mask = M.random_uniform(shape=inputs.shape) > self.rate
            mask = self.mask
        return inputs * mask

    def backward(self, loss):
        return loss * self.mask
