import mlpy.backend.math as M
from mlpy.activations.activation import Activation


class eLU(Activation):
    """Exponential linear unit.

    Args:
        alpha (float, optional): A scalar, slope of negative section.

    References:
        - [Fast and Accurate Deep Network Learning by Exponential
           Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    """
    def __init__(self, alpha=1.0):
        super(eLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """Exponential linear unit.

        Args:
            x (tensor): Input tensor.

        Returns:
            The exponential linear activation:
            `x` if `x >= 0` and
            `alpha * (exp(x)-1)` if `x < 0`.
        """
        return M.where(x >= 0, x, self.alpha * (M.exp(x) - 1))

    def gradient(self, x):
        """Gradient of eLU activation function."""
        return M.where(x >= 0, 1, self.alpha * M.exp(x))


class SeLU(eLU):
    """Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    Note:
        - To be used together with the initialization "lecun_normal".
        - To be used together with the dropout variant "AlphaDropout".

    References:
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    def __init__(self):
        super(SeLU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def call(self, x):
        """Scaled Exponential Linear Unit (SELU).

        Args:
            x: A tensor or variable to compute the activation function for.

        Returns:
            The scaled exponential unit activation: `scale * elu(x, alpha)`.
        """
        elu = super(SeLU, self).forward(x)
        return self.scale * elu

    def gradient(self, x):
        """Gradient of SeLU activation function."""
        elu = super(SeLU, self).backward(x)
        return self.scale * elu
