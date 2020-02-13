class Initializer(object):
    """Initializer base class, all initializers inherit from this class."""
    def __call__(self, shape, dtype=None):
        raise NotImplementedError
