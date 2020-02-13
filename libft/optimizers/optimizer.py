class Optimizer(object):
    """Abstract optimizer base class."""
    def __init__(self):
        self.updates = []

    def get_updates(self, grads, params):
        raise NotImplementedError
