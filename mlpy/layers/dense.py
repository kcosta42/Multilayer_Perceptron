import mlpy.backend.math as M
import mlpy.initializers as initializers
from mlpy.layers.layer import Layer


class Dense(Layer):
    """Dense Layer, a regular densely-connected NN layer.

    Args:
        units (int): Dimensionality of the output space.
        use_bias (boolean, optionnal): Whether the layer uses a bias vector.
        kernel_initializer (string or Initializer, optional):
            Initializer for kernel.
        bias_initializer (string or Initializer, optional):
            Initializer for bias.
    """
    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(Dense, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def call(self, inputs):
        output = M.dot(inputs, self.kernel)
        if self.use_bias:
            output = output + M.transpose(self.bias)
        return output

    def backward(self, loss, params):
        self.gradients = []

        dZ = loss[0]
        m = dZ.shape[0]
        kernel_grad = (1. / m) * M.transpose(M.dot(dZ, self.inputs))
        self.gradients.append(kernel_grad)

        bias_grad = (1. / m) * M.sum(dZ, axis=1, keepdims=True)
        self.gradients.append(bias_grad)

        self.gradients.append(dZ)
        return self.gradients

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer)

        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units, 1),
                                        initializer=self.bias_initializer)

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        self.shape = tuple(output_shape)
        self.built = True
