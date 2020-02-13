import mlpy.backend.math as M

SUM = 'sum'
SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'
WEIGHTED_MEAN = 'weighted_mean'


class Metric(object):
    """Encapsulates metric logic and state."""
    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value."""
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """Accumulates statistics for the metric."""
        raise NotImplementedError


class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values."""
    def __init__(self, reduction):
        """Creates a `Reduce` instance.

        Arguments:
            reduction: string
                a metrics `Reduction` value.
        """
        super(Reduce, self).__init__()
        self.reduction = reduction

    def call(self, values):
        """Accumulates statistics for computing the reduction metric.

        Arguments:
            values: array-like
                Per-example value.

        Returns:
            A scalar corresponding to the metric value.
        """
        if self.reduction == SUM:
            return M.sum(values)
        if self.reduction in [WEIGHTED_MEAN, SUM_OVER_BATCH_SIZE]:
            return M.sum(values) / values.size

        raise NotImplementedError(f"{self.reduction} not implemented.")


class Sum(Reduce):
    """Computes the (weighted) sum of the given values."""
    def __init__(self):
        super(Sum, self).__init__(reduction=SUM)


class Mean(Reduce):
    """Computes the (weighted) mean of the given values."""
    def __init__(self):
        super(Mean, self).__init__(reduction=WEIGHTED_MEAN)


class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric."""
    def __init__(self, fn, **kwargs):
        """Creates a `MeanMetricWrapper` instance.

        Arguments:
            fn: callable
                The metric function to wrap, with signature
                `fn(y_true, y_pred, **kwargs)`.
            **kwargs: dict
                The keyword arguments that are passed on to `fn`.
        """
        super(MeanMetricWrapper, self).__init__()
        self._fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Accumulates metric statistics.

        `y_true` and `y_pred` must have the same shape.

        Arguments:
            y_true: array-like
                The ground truth values.
            y_pred: array-like
                The predicted values.

        Returns:
            A scalar corresponding to the metric value.
        """
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).call(matches)
