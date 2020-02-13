import mlpy.backend.math as M
from mlpy.metrics.metric import MeanMetricWrapper


class BinaryAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels.

    Arguments:
        threshold: float, Default: 0.5
            Threshold for deciding whether prediction values are 1 or 0.
    """
    def __init__(self, threshold=0.5):
        super(BinaryAccuracy, self).__init__(
            self.binary_accuracy,
            threshold=threshold,
        )

    def binary_accuracy(self, y_true, y_pred, threshold=0.5):
        if threshold != 0.5:
            y_pred = (y_pred > threshold)
        return M.mean(M.equal(y_true, M.round(y_pred)), axis=-1)
