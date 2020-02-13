from libft.initializers.constant import Constant
from libft.initializers.glorot import GlorotNormal, GlorotUniform
from libft.initializers.he import HeNormal, HeUniform
from libft.initializers.initializer import Initializer
from libft.initializers.lecun import LecunNormal, LecunUniform
from libft.initializers.normal import RandomNormal
from libft.initializers.ones import Ones
from libft.initializers.uniform import RandomUniform
from libft.initializers.variance import VarianceScaling
from libft.initializers.zeros import Zeros

INITIALIZERS = {
    'constant': Constant,
    'glorot_normal': GlorotNormal,
    'glorot_uniform': GlorotUniform,
    'he_normal': HeNormal,
    'he_uniform': HeUniform,
    'lecun_uniform': LecunNormal,
    'lecun_normal': LecunUniform,
    'normal': RandomNormal,
    'ones': Ones,
    'uniform': RandomUniform,
    'variance_scaling': VarianceScaling,
    'zeros': Zeros,
}


def get(identifier, **kwargs):
    """Initializer instance getter.

    Arguments:
        identifier: string or Initializer
            An Initializer instance or it's name.
        kwargs: dict
            Keywords arguments for instance initialisation.

    Raises:
        ValueError:
            If identifier does not match with an existing Initializer instance.

    Returns:
        Initializer instance.
    """
    if identifier is None:
        return None

    if isinstance(identifier, Initializer):
        return identifier

    identifier = identifier.lower()
    if identifier not in INITIALIZERS:
        raise ValueError(f"Could not interpret Initializer instance "
                            f"identifier: {identifier}")

    initializer = INITIALIZERS[identifier](**kwargs)
    return initializer


__all__ = [
    'get',
    'Constant',
    'GlorotNormal',
    'GlorotUniform',
    'HeNormal',
    'HeUniform',
    'LecunNormal',
    'LecunUniform',
    'RandomNormal',
    'Ones',
    'RandomUniform',
    'VarianceScaling',
    'Zeros',
]
