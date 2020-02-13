from libft.initializers.variance import VarianceScaling


class HeNormal(VarianceScaling):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        seed: integer, Default: None
            Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(scale=2.0,
                                       mode='fan_in',
                                       distribution='normal',
                                       seed=seed)


class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        seed: integer, Default: None
            Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(scale=2.0,
                                        mode='fan_in',
                                        distribution='uniform',
                                        seed=seed)
