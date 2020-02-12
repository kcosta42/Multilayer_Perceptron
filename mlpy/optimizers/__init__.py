from mlpy.optimizers.optimizer import Optimizer
from mlpy.optimizers.sgd import SGD

OPTIMIZERS = {
    'sgd': SGD
}


def get(identifier, **kwargs):
    """Optimizer instance getter.

    Args:
        identifier (string or Optimizer):
            An optimizer instance name or instance.

    Raises:
        ValueError: If identifier does not match with an existing optimizer
            instance.

    Returns:
        Optimizer instance.
    """
    if isinstance(identifier, Optimizer):
        return identifier

    identifier = identifier.lower()
    if identifier not in OPTIMIZERS:
        raise ValueError(f"Could not interpret Optimizer instance "
                         f"identifier: {identifier}")

    optimizer = OPTIMIZERS[identifier](**kwargs)
    return optimizer


__all__ = [
    'get',
    'SGD',
]
