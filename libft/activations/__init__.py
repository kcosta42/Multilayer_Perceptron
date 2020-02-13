from libft.activations.activation import Activation
from libft.activations.elu import SeLU, eLU
from libft.activations.linear import Linear
from libft.activations.relu import ReLU
from libft.activations.sigmoid import Sigmoid
from libft.activations.softmax import Softmax
from libft.activations.tanh import TanH

ACTIVATIONS = {
    'selu': SeLU,
    'elu': eLU,
    'linear': Linear,
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': TanH,
}

def get(identifier, **kwargs):
    """Activation instance getter.

    Arguments:
        identifier: string or Activation
            An Activation instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Activation instance.

    Returns:
        An Activation instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Activation):
        return identifier

    identifier = identifier.lower()
    if identifier not in ACTIVATIONS:
        raise ValueError(f"Could not interpret Activation instance "
                         f"identifier: {identifier}")

    activation = ACTIVATIONS[identifier](**kwargs)

    return activation


__all__ = [
    'get',
    'SeLU',
    'eLU',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Softmax',
    'TanH',
]
