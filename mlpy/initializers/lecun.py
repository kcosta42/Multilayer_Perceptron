from mlpy.initializers.variance import VarianceScaling


class LecunNormal(VarianceScaling):
    """LeCun normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        seed: integer, Default: None
            Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
    [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(scale=1.0,
                                          mode='fan_in',
                                          distribution='normal',
                                          seed=seed)


class LecunUniform(VarianceScaling):
    """LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        seed: integer, Default: None
            Used to seed the random generator.

    Returns:
        A VarianceScaling initializer.

    References:
    [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    def __init__(self, seed=None):
        super(LecunUniform, self).__init__(scale=1.0,
                                           mode='fan_in',
                                           distribution='uniform',
                                           seed=seed)
