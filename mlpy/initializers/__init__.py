from mlpy.initializers.constant import Constant
from mlpy.initializers.glorot import GlorotNormal
from mlpy.initializers.glorot import GlorotUniform
from mlpy.initializers.he import HeNormal
from mlpy.initializers.he import HeUniform
from mlpy.initializers.initializer import Initializer
from mlpy.initializers.lecun import LecunNormal
from mlpy.initializers.lecun import LecunUniform
from mlpy.initializers.normal import RandomNormal
from mlpy.initializers.ones import Ones
from mlpy.initializers.uniform import RandomUniform
from mlpy.initializers.variance import VarianceScaling
from mlpy.initializers.zeros import Zeros

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

    Args:
        identifier (string or Initializer):
            An initializer instance name or instance.

    Raises:
        ValueError: If identifier does not match with an existing initializer
            instance.

    Returns:
        Initializer instance.
    """
    if isinstance(identifier, Initializer):
        return identifier

    identifier = identifier.lower()
    if identifier not in INITIALIZERS:
        raise ValueError(f"Could not interpret initializer instance "
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
