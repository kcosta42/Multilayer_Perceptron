from mlpy.layers.layer import Layer


class Input(Layer):
    """Input Layer class, it represent the main entry for a model.

    Args:
        inputs (tensor, optional): Tensor to wrap into the `Input` layer.
        input_shape (tuple or list, optional): A shape tuple, not including
            the batch size.

    Raises:
        ValueError: If both inputs and input_shape are not defined.
    """
    def __init__(self, inputs=None, input_shape=None):
        super(Input, self).__init__()
        if inputs is None and input_shape is None:
            raise ValueError("`inputs` or `input_shape` must be defined.")

        if inputs is None:
            self.shape = (None,) + tuple(input_shape)
            self.outputs = None
        else:
            self.shape = inputs.shape
            self.outputs = inputs

        self.built = True
