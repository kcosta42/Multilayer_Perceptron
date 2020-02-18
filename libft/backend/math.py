import numpy as np

_FLOATX = 'float32'
_EPSILON = 1e-7


def floatx():
    return _FLOATX


def epsilon():
    return _EPSILON


def equal(x1, x2, **kwargs):
    return np.equal(x1, x2, **kwargs)


def sign(x, **kwargs):
    return np.sign(x, **kwargs)


def abs(x, **kwargs):
    return np.abs(x, **kwargs)


def min(a, b=None, **kwargs):
    if b is None:
        return np.min(a, **kwargs)
    return np.minimum(a, b, **kwargs)


def max(a, b=None, **kwargs):
    if b is None:
        return np.max(a, **kwargs)
    return np.maximum(a, b, **kwargs)


def sum(*args, **kwargs):
    return np.sum(*args, **kwargs)


def round(a, **kwargs):
    return np.round(a, **kwargs)


def mean(a, **kwargs):
    return np.mean(a, **kwargs)


def std(a, **kwargs):
    return np.std(a, **kwargs)


def square(x):
    return np.square(x)


def where(condition, x=None, y=None):
    return np.where(condition, x, y)


def transpose(a, axes=None):
    return np.transpose(a, axes=axes)


def diagflat(*args, **kwargs):
    return np.diagflat(*args, **kwargs)


def dot(a, b):
    return np.dot(a, b)


def exp(x):
    return np.exp(x)


def sqrt(x):
    return np.sqrt(x)


def log(x):
    return np.log(x)


def clip(x1, x2, x3):
    return np.clip(x1, x2, x3)


def random_seed(seed):
    return np.random.seed(seed)


def rand(*args):
    return np.random.rand(*args)


def randint(low, high=None, size=None, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.randint(low, high, size)


def random_uniform(shape=None, low=0.0, high=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.uniform(low, high, size=shape)


def random_normal(shape=None, mean=0.0, stddev=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.normal(mean, stddev, size=shape)


def permutation(x, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.permutation(x)


def array(a, **kwargs):
    return np.array(a, **kwargs)


def constant(value, shape, **kwargs):
    return np.full(shape, value, **kwargs)


def ones(shape, **kwargs):
    return np.ones(shape, **kwargs)


def zeros(shape, **kwargs):
    return np.zeros(shape, **kwargs)


def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)


def reshape(a, newshape, **kwargs):
    return np.reshape(a, newshape, **kwargs)


def append(arr, values, axis=None):
    return np.append(arr, values, axis=axis)
