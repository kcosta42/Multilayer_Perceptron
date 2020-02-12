import mlpy.backend.math as M
from mlpy.losses.loss import Loss


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between labels and predictions."""
    def call(self, y_true, y_pred):
        return M.mean(M.square(y_pred - y_true), axis=-1)

    def gradient(self, y_true, y_pred):
        return -(y_true - y_pred)
