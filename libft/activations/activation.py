from libft.layers.layer import Layer


class Activation(Layer):
    """Activation layer, which applies an activation function to an output."""
    def __init__(self, **kwargs):
        super(Activation, self).__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError

    def backward(self, loss):
        return loss * self.gradient(self.inputs)

    def forward(self, x):
        """Forward activation function."""
        raise NotImplementedError

    def gradient(self, x):
        """Backward (ie. Gradient / Derivative) activation function."""
        raise NotImplementedError
