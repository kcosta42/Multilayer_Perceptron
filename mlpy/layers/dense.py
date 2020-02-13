import mlpy.backend.math as M
import mlpy.initializers as initializers
from mlpy.layers.layer import Layer
import copy


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
        self.kernel_optimizer = None
        self.bias_optimizer = None

    def call(self, inputs):
        output = M.dot(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias
        return output

    def backward(self, loss):
        _loss = M.dot(loss, M.transpose(self.kernel))

        kernel_loss = M.dot(M.transpose(self.inputs), loss)
        bias_loss = M.sum(loss, axis=0, keepdims=True)

        self.kernel = self.kernel_optimizer.update(kernel_loss, self.kernel)
        self.bias = self.bias_optimizer.update(bias_loss, self.bias)

        # _loss = M.dot(loss, self.kernel)
        return _loss

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer)

        self.bias = None
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer)

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        self.shape = tuple(output_shape)
        self.built = True

    def optimize(self, optimizer):
        self.kernel_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)
