import mlpy.backend.math as M
from mlpy.metrics.metric import MeanMetricWrapper


class Accuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels."""
    def __init__(self):
        super(Accuracy, self).__init__(self.accuracy)

    def accuracy(self, y_true, y_pred):
        return M.equal(y_true, y_pred)
