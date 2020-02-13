import mlpy.backend.math as M


class Layer(object):
    """Base Layer class for all layers.

    Arguments:
        dtype: string, Default: None
            Data type for the weights in this layer.
        input_shape: tuple or list, Default: None
            A shape tuple, not including the batch size.

    Attributes:
        shape: tuple
            Output shape of this layer.
        size: integer
            Number of parameters in this layers.
    """
    def __init__(self, dtype=None, input_shape=None):
        if dtype is None:
            dtype = M.floatx()
        self.dtype = dtype

        self.shape = None
        if input_shape is not None:
            self.shape = (None,) + tuple(input_shape)

        self.built = False
        self.weights = []

        self.outputs = None
        self.inputs = None

    def add_weight(self, shape=None, initializer=None, dtype=None):
        """Used to create and add weights to this layer.

        Arguments:
            shape: tuple, Default: None
                Shape of the weigth.
            initializer: Initializer, Default: None
                An Initializer instance (callable).
            dtype: string, Default: None
                Data type for the weight.

        Returns:
            The initialized weights.
        """
        if shape is None:
            shape = ()

        if dtype is None:
            dtype = self.dtype

        weight = initializer(shape, dtype=dtype)
        self.weights.append(weight)
        return weight

    def __call__(self, inputs):
        """A wrapper for the self.call() function."""
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        if isinstance(inputs, Layer):
            if not hasattr(inputs, 'outputs') or inputs.outputs is None:
                return self
            inputs = inputs.outputs
        self.inputs = inputs

        outputs = self.call(inputs)
        self.outputs = outputs
        return outputs

    def call(self, inputs):
        """This is were the layer's logic lives.

        Arguments:
            inputs: array-like
                An input tensor.

        Returns:
            A resulting tensor.
        """
        return inputs

    def backward(self, loss):
        """Compute the gradient of the layer.

        Arguments:
            loss: scalar or array-like
                The accumulated loss from the previous layer.
        """
        raise NotImplementedError

    def build(self, input_shape):
        """Function used to build the Layer.

        It must be implemented in every layer that need to know the shape
        of the previous layer.

        Arguments:
            input_shape: tuple
                The shape of the inputs.
        """
        self.built = True
        self.shape = tuple(input_shape)

    @property
    def size(self):
        size = 0
        for w in self.weights:
            size += w.size
        return size
