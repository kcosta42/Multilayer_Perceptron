from mlpy.losses.binary_crossentropy import BinaryCrossentropy
from mlpy.losses.loss import Loss
from mlpy.losses.mean_squared_error import MeanSquaredError

LOSSES = {
    'mse': MeanSquaredError,
    'mean_squared_error': MeanSquaredError,
    'bce': BinaryCrossentropy,
    'binary_crossentropy': BinaryCrossentropy,
}


def get(identifier, **kwargs):
    """Loss instance getter.

    Arguments:
        identifier: string or Loss
            An Loss instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Loss instance.

    Returns:
        An Loss instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Loss):
        return identifier

    identifier = identifier.lower()
    if identifier not in LOSSES:
        raise ValueError(f"Could not interpret Loss instance "
                         f"identifier: {identifier}")

    loss = LOSSES[identifier](**kwargs)
    return loss


__all__ = [
    'get',
    'MeanSquaredError',
    'BinaryCrossentropy',
]
