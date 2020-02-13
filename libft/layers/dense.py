import copy

import libft.activations as activations
import libft.backend.math as M
import libft.initializers as initializers
import libft.regularizers as regularizers
from libft.layers.layer import Layer


class Dense(Layer):
    """Dense Layer, a regular densely-connected NN layer.

    Arguments:
        units: integer
            Dimensionality of the output space.
        activation: string or Activation, Default: None
            Activation function to use.
        use_bias: boolean, Default: True
            Whether the layer uses a bias vector.
        kernel_initializer: string or Initializer, Default: "glorot_uniform"
            Initializer for kernel.
        bias_initializer: string or Initializer, Default: "zeros"
            Initializer for bias.
        kernel_regularizer: string or Regularizer, Default: None
            Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: string or Regularizer, Default: None
            Regularizer function applied to the bias vector.
    """
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_optimizer = None
        self.bias_optimizer = None

    def call(self, inputs):
        output = M.dot(inputs, self.kernel)
        if self.kernel_regularizer:
            output += self.kernel_regularizer(output)

        if self.use_bias:
            output += self.bias
            if self.bias_regularizer:
                output += self.bias_regularizer(self.bias)
        return output

    def backward(self, loss):
        _loss = M.dot(loss, M.transpose(self.kernel))
        if not self.trainable:
            return _loss

        kernel_loss = M.dot(M.transpose(self.inputs), loss)
        if self.kernel_regularizer:
            kernel_loss += self.kernel_regularizer.gradient(kernel_loss)

        bias_loss = M.sum(loss, axis=0, keepdims=True)
        if self.bias_regularizer:
            bias_loss += self.bias_regularizer.gradient(bias_loss)

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
