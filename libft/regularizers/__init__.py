from libft.regularizers.l1 import L1
from libft.regularizers.l1l2 import L1L2
from libft.regularizers.l2 import L2
from libft.regularizers.regularizer import Regularizer

REGULARIZERS = {
    'l1': L1,
    'l2': L2,
    'l1l2': L1L2,
}

def get(identifier, **kwargs):
    """Regularizer instance getter.

    Arguments:
        identifier: string or Regularizer
            An Regularizer instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Regularizer instance.

    Returns:
        An Regularizer instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Regularizer):
        return identifier

    identifier = identifier.lower()
    if identifier not in REGULARIZERS:
        raise ValueError(f"Could not interpret Regularizer instance "
                         f"identifier: {identifier}")

    regularizer = REGULARIZERS[identifier](**kwargs)

    return regularizer


__all__ = [
    'get',
    'L1',
    'L2',
    'L1L2',
]
