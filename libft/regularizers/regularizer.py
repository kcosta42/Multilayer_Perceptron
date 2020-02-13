
class Regularizer(object):
    """Regularizer base class."""
    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        """Invokes the `Regularizer` instance."""
        return 0.0

    def gradient(self, x):
        """Compute gradient for the `Regularizer` instance."""
        return 0.0
