from libft.metrics.accuracy import Accuracy
from libft.metrics.binary_accuracy import BinaryAccuracy
from libft.metrics.metric import Metric

METRICS = {
    'accuracy': Accuracy,
    'binary_accuracy': BinaryAccuracy,
}


def get(identifier, **kwargs):
    """Metric instance getter.

    Arguments:
        identifier: string or Metric
            An Metric instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Metric instance.

    Returns:
        An Metric instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Metric):
        return identifier

    identifier = identifier.lower()
    if identifier not in METRICS:
        raise ValueError(f"Could not interpret Metric instance "
                         f"identifier: {identifier}")

    metric = METRICS[identifier](**kwargs)
    return metric


__all__ = [
    'get',
    'Accuracy',
    'BinaryAccuracy',
]
