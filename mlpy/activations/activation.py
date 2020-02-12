import mlpy.backend.math as M
from mlpy.layers.layer import Layer


class Activation(Layer):
    """Activation layer, which applies an activation function to an output."""
    def __init__(self):
        super(Activation, self).__init__()

    def call(self, inputs):
        raise NotImplementedError

    def backward(self, loss, params):
        self.gradients = []

        dZ = loss[2]
        kernel = params[0]

        activation_grad = M.dot(kernel, dZ)
        Z_grad = activation_grad * M.transpose(self.gradient(self.inputs))

        self.gradients.append(Z_grad)
        self.gradients.append(activation_grad)
        return self.gradients

    def forward(self, x):
        """Forward activation function."""
        raise NotImplementedError

    def gradient(self, x):
        """Backward (ie. Gradient / Derivative) activation function."""
        raise NotImplementedError
