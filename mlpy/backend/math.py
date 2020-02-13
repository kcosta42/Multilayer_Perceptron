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


def round(a, decimals=0, out=None):
    return np.round(a, decimals=decimals, out=out)


def mean(a, axis=None):
    return np.mean(a, axis=axis)


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


def rand(*args):
    return np.random.rand(*args)


def permutation(x, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.permutation(x)


def random_uniform(shape=None, low=0.0, high=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.uniform(low, high, size=shape)


def random_normal(shape=None, mean=0.0, stddev=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    return np.random.normal(mean, stddev, size=shape)


def array(a, dtype=None, copy=True):
    return np.array(a, dtype=dtype, copy=copy)


def constant(value, shape=None, dtype=None):
    return np.full(shape, value, dtype=dtype)


def ones(shape, dtype=None):
    return constant(1, shape, dtype=dtype)


def zeros(shape, dtype=None):
    return constant(0, shape, dtype=dtype)


def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)


def reshape(a, newshape, order='C'):
    return np.reshape(a, newshape, order=order)
