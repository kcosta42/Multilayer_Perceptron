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

    Args:
        identifier (string or Loss):
            An loss instance name or instance.

    Raises:
        ValueError: If identifier does not match with an existing loss instance.

    Returns:
        Loss instance.
    """
    if isinstance(identifier, Loss):
        return identifier

    identifier = identifier.lower()
    if identifier not in LOSSES:
        raise ValueError(f"Could not interpret loss instance "
                         f"identifier: {identifier}")

    loss = LOSSES[identifier](**kwargs)
    return loss


__all__ = [
    'get',
    'MeanSquaredError',
    'BinaryCrossentropy',
]
