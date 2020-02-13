import libft.backend.math as M
from libft.initializers.initializer import Initializer


class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.

    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    Arguments:
        scale: float, Default: 1.0
            Scaling factor.
        mode: string, Default: "fan_in"
            One of "fan_in", "fan_out", "fan_avg".
        distribution: string, Default: "normal"
            One of "normal", "uniform".
            Random distribution to use.
        seed: integer, Default: None
            Used to seed the random generator.

    Raises:
        ValueError:
            In case of an invalid value for the "scale", mode" or
            "distribution" arguments.

    References:
        https://github.com/keras-team/keras/blob/master/keras/initializers.py
    """
    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError(f"`scale` must be a positive float. Got: {scale}")

        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError("Invalid `mode` argument: "
                             "expected on of {'fan_in', 'fan_out', 'fan_avg'} "
                             f"but got {mode}")

        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError("Invalid `distribution` argument: "
                             "expected one of {'normal', 'uniform'} "
                             f"but got {distribution}")
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = shape[0], shape[1]
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = M.sqrt(scale) / .87962566103423978
            x = M.random_uniform(shape, 0., stddev, seed=self.seed)
        else:
            limit = M.sqrt(3. * scale)
            x = M.random_uniform(shape, -limit, limit, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x
