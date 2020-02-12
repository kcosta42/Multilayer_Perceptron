from mlpy.metrics.metric import Metric
from mlpy.metrics.accuracy import Accuracy
from mlpy.metrics.binary_accuracy import BinaryAccuracy

METRICS = {
    'accuracy': Accuracy,
    'binary_accuracy': BinaryAccuracy,
}


def get(identifier, **kwargs):
    """Metric instance getter.

    Args:
        identifier (string or Metric):
            An metric instance name or instance.

    Raises:
        ValueError: If identifier does not match with an existing metric
            instance.

    Returns:
        Metric instance.
    """
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
