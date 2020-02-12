from mlpy.initializers.variance import VarianceScaling


class GlorotNormal(VarianceScaling):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Args:
        seed (integer): Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(scale=1.0,
                                           mode='fan_avg',
                                           distribution='normal',
                                           seed=seed)


class GlorotUniform(VarianceScaling):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Args:
        seed (integer): Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(scale=1.0,
                                            mode='fan_avg',
                                            distribution='uniform',
                                            seed=seed)
