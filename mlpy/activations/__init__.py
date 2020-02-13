from mlpy.activations.activation import Activation
from mlpy.activations.elu import eLU, SeLU
from mlpy.activations.linear import Linear
from mlpy.activations.relu import ReLU
from mlpy.activations.sigmoid import Sigmoid
from mlpy.activations.softmax import Softmax
from mlpy.activations.tanh import Tanh

ACTIVATIONS = {
    'elu': eLU,
    'selu': SeLU,
    'linear': Linear,
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': Tanh,
}

def get(identifier, **kwargs):
    """Activation instance getter.

    Args:
        identifier (string): An activation instance name.

    Raises:
        ValueError: If identifier does not match with an existing activation
            instance.

    Returns:
        An Activation instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Activation):
        return identifier

    identifier = identifier.lower()
    if identifier not in ACTIVATIONS:
        raise ValueError(f"Could not interpret activation instance "
                         f"identifier: {identifier}")

    activation = ACTIVATIONS[identifier](**kwargs)

    return activation


__all__ = [
    'get',
    'eLU',
    'SeLU',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Softmax',
    'Tanh',
]
