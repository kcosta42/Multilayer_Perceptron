import mlpy.backend.math as M


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    Args:
        y (1d-array): class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes (integer, optional): total number of classes.
        dtype (string, optional): The data type expected by the input
            (`float32`, `float64`, `int32`...)

    Returns:
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = M.array(y, dtype='int')
    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.ravel()

    if not num_classes:
        num_classes = M.max(y) + 1
    n = y.shape[0]

    categorical = M.zeros((n, num_classes), dtype=dtype)
    categorical[M.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)
    categorical = M.reshape(categorical, output_shape)
    return categorical
