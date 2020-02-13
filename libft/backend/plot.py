import matplotlib.pyplot as plt


def scatter(x, y, **kwargs):
    return plt.scatter(x, y, **kwargs)


def plot(*args, **kwargs):
    return plt.plot(*args, **kwargs)


def show(*args, **kw):
    return plt.show(*args, **kw)


def xlabel(label, **kwargs):
    return plt.xlabel(label, **kwargs)


def ylabel(label, **kwargs):
    return plt.ylabel(label, **kwargs)
