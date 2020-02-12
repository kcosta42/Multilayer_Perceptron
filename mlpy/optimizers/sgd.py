import mlpy.backend.math as M
from mlpy.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Args:
        learning_rate (float, optional): Learning rate >= 0.
        decay (float, optional): Decay in learning rate.
        momentum (float, optional): Parameter >= 0 that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov (boolean, optional). Whether to apply Nesterov momentum.
    """
    def __init__(self,
                 learning_rate=0.01,
                 decay=0.0,
                 momentum=0.0,
                 nesterov=False):
        super(SGD, self).__init__()
        self.initial_decay = decay
        self.iterations = M.zeros(1)

        self.learning_rate = learning_rate
        self.decay = self.initial_decay
        self.momentum = momentum
        self.nesterov = nesterov

    def get_updates(self, grads, params):
        self.updates = [M.update_add(self.iterations, 1)]

        lr = self.learning_rate
        it = self.iterations[0]
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * it))

        shapes = [p.shape for p in params]
        moments = [M.zeros(shape) for shape in shapes]  # TODO Manage this
        # print(moments)
        for i in range(len(params)):
            p = params[i]
            g = grads[i]
            m = moments[i]

            v = self.momentum * m - lr * g
            self.updates.append(M.update(m, v))
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append(M.update(p, new_p))
        return self.updates
