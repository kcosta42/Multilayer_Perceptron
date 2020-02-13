from libft.layers.layer import Layer


class Input(Layer):
    """Input Layer class, it represent the main entry for a model.

    Arguments:
        inputs: array-like
            Tensor to wrap into the `Input` layer.

    Raises:
        ValueError: If both inputs and input_shape are not defined.
    """
    def __init__(self, inputs=None, **kwargs):
        super(Input, self).__init__(**kwargs)
        if inputs is None and self.shape is None:
            raise ValueError("`inputs` or `input_shape` must be defined.")

        if inputs is not None:
            self.shape = inputs.shape
            self.outputs = inputs

        self.built = True
