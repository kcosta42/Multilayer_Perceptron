import mlpy.backend.math as M
from mlpy.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Arguments:
        learning_rate: float, Default: 0.01
            Learning rate >= 0.
        decay: float, Default: 0.0
            Decay in learning rate.
        momentum: float, Default: 0.0
            Parameter >= 0 that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov: boolean, Default: False
            Whether to apply Nesterov momentum.
    """
    def __init__(self,
                 learning_rate=0.01,
                 decay=0.0,
                 momentum=0.0,
                 nesterov=False):
        super(SGD, self).__init__()
        self.initial_decay = decay
        self.iterations = 0

        self.learning_rate = learning_rate
        self.decay = self.initial_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.moments = None

    def update(self, loss, param):
        self.iterations += 1

        lr = self.learning_rate
        it = self.iterations
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * it))

        if self.moments is None:
            self.moments = M.zeros(param.shape)

        p = param
        g = loss
        m = self.moments
        v = self.momentum * m - lr * g
        self.moments = v

        if self.nesterov:
            return p + self.momentum * v - lr * g
        return p + v
