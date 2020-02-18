import libft.backend.math as M


class StandardScaler(object):
    """Standardize features by removing the mean and scaling to unit variance.

    Arguments:
        mean: array-like
            Mean for each feature of X.
        std: array-like
            Std for each feature of X.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        """Compute the mean and std to be used for later scaling.

        Arguments:
            X: array-like
                The data used to compute the mean and standard deviation used
                for later scaling along the features axis.
        """
        self.mean = M.array([])
        self.std = M.array([])

        for i in range(X.shape[1]):
            self.mean = M.append(self.mean, M.mean(X[:, i]))
            self.std = M.append(self.std, M.std(X[:, i]))
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Arguments:
            X: array-like
                The data that should be scaled.
        """
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Arguments:
            X: array-like
                Training set.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Arguments:
            X: array-like
                The data used to scale along the features axis.
        """
        return X * self.std + self.mean
