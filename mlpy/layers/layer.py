import mlpy.backend.math as M


class Layer(object):
    """Base Layer class for all layers.

    Args:
        dtype (string, optional): Data type for the weights in this layer.

    Attributes:
        shape (tuple): Output shape of the layer.
        built (boolean): Whether the layer has been built or not.
        weights (list): List of all the weights in this layer.

    """
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = M.floatx()
        self.dtype = dtype

        self.shape = None
        self.built = False
        self.weights = []

        self.gradients = []
        self.outputs = None
        self.inputs = None

    def add_weight(self, shape=None, initializer=None, dtype=None):
        """Used to create and add weights to this layer.

        Args:
            shape (tuple, optional): Shape of the weigth.
            initializer (object, optional): An Initializer instance (callable).
            dtype (string, optional): Data type for the weight.

        Returns:
            The initialized weights.
        """
        if shape is None:
            shape = ()

        if dtype is None:
            dtype = self.dtype

        weight = initializer(shape)
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

        Args:
            inputs (tensor): An input tensor.

        Returns:
            A resulting tensor.
        """
        return inputs

    def backward(self, *args):
        """Compute the gradient of the layer.

        Args:
            *args: All the parameters needed to compute the gradient.
        """
        raise NotImplementedError

    def build(self, input_shape):
        """Function used to build the Layer.

        It must be implemented in every layer that need to know the shape
        of the previous layer.

        Args:
            input_shape (tuple): The shape of the inputs
        """
        self.built = True
        self.shape = tuple(input_shape)

    @property
    def size(self):
        size = 0
        for w in self.weights:
            size += w.size
        return size
