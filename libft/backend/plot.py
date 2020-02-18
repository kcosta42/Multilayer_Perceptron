import matplotlib.pyplot as plt


def figure(*args, **kwargs):
    return plt.figure(*args, **kwargs)


def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs)


def show(*args, **kw):
    return plt.show(*args, **kw)


def scatter(x, y, **kwargs):
    return plt.scatter(x, y, **kwargs)


def plot(*args, **kwargs):
    return plt.plot(*args, **kwargs)


def xlabel(label, **kwargs):
    return plt.xlabel(label, **kwargs)


def ylabel(label, **kwargs):
    return plt.ylabel(label, **kwargs)
