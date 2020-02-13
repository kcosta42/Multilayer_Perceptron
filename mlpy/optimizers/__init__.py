from mlpy.optimizers.optimizer import Optimizer
from mlpy.optimizers.sgd import SGD
from mlpy.optimizers.rmsprop import RMSprop

OPTIMIZERS = {
    'rmsprop': RMSprop,
    'sgd': SGD,
}


def get(identifier, **kwargs):
    """Optimizer instance getter.

    Arguments:
        identifier: string or Optimizer
            An Optimizer instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Optimizer instance.

    Returns:
        An Optimizer instance.
    """
    if identifier is None:
        return None

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
    'RMSprop',
    'SGD',
]
