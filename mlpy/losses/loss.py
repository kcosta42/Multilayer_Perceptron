import mlpy.backend.math as M


class Loss(object):
    """Loss base class."""
    def __call__(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): The predicted values.

        Returns:
            Weighted loss float tensor or scalar.
        """
        losses = self.call(y_true, y_pred)
        return M.sum(losses) / losses.size

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): The predicted values.
        """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """Compute gradient for the `Loss` instance.

        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): The predicted values.
        """
        raise NotImplementedError
