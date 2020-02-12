import mlpy.backend.math as M


def shuffle_data(X, y, seed=None):
    """Random shuffle of the samples in X and y.

    Args:
        X (tensor): Input data.
        y (tensor): Target data.
        seed (integer): Random state.

    Returns:
        A tuple (X, y) permuted.
    """
    p = M.permutation(X.shape[0], seed=seed)
    return X[p], y[p]


def batch_iterator(X, y=None, batch_size=32):
    """Simple batch generator.

    Args:
        X (tensor): Input data.
        y (tensor, optional): Target data.
        batch_size (integer, optional): Batch size. Default to 32.

    Yields:
        If `y` in provided then a tuple (`X`, `y`) is returned,
        else only `X` is returned.
    """
    n_samples = X.shape[0]

    for i in M.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def train_test_split(X, y, split=0.3, shuffle=True, seed=None):
    """Split the data into train and test sets.

    Args:
        X (tensor): Input data.
        y (tensor): Target data.
        split (float, optional): Ratio for splitting (between 0.0 and 1.0).
        shuffle (boolean, optional): Shuffle before splitting.
        seed (integer): Random state.

    Returns:
        A tuple (X_train, X_test, y_train, y_test) corresponding to the split.
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    idx = X.shape[0] - int(X.shape[0] // (1 / split))
    X_train, X_test = X[:idx], X[idx:]
    y_train, y_test = y[:idx], y[idx:]

    return X_train, X_test, y_train, y_test
