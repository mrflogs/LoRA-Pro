import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot"]


def plot(func, xlim, *, ax=None, plot_points=300, **kwargs):
    x = np.linspace(xlim[0], xlim[1], plot_points)
    if ax is None:
        ax = plt.gca()
    ax = ax.plot(x, func(x), **kwargs)
    return ax
