import libft.backend.math as M
from libft.optimizers.optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSProp optimizer.

    Arguments:
        learning_rate: float, Default: 1e-3
            Learning rate.
        rho: float, Default: 0.9
        decay: float, Default: 0.0
        epsilon: float, Default: 1e-7

    References:
        - [rmsprop: Divide the gradient by a running average of its recent
           magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) # noqa
    """

    def __init__(self, learning_rate=1e-3, rho=0.9, decay=0.0, epsilon=1e-7):
        super(RMSprop, self).__init__()
        self.initial_decay = decay
        self.iterations = 0

        self.learning_rate = learning_rate
        self.rho = rho
        self.decay = self.initial_decay
        self.epsilon = epsilon
        self.accumulators = None

    def update(self, loss, param):
        self.iterations += 1

        if self.accumulators is None:
            self.accumulators = M.zeros(param.shape)

        lr = self.learning_rate
        it = self.iterations
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * it))

        p = param
        g = loss
        a = self.accumulators
        a = self.rho * a + (1. - self.rho) * M.square(g)
        return p - lr * g / (M.sqrt(a) + self.epsilon)
