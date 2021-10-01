
# Date: 2021/09/07
# Author: Oskar Fernlund
# Description: Library of different stochastic process classes.


# Imports & Dependencies
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')


# Function Definitions
# =============================================================================

def static_plot_1D(samples, figsize=(16, 9)):
    """ Plot 1D stochastic process samples statically (still plot).

    Args:
        samples (array) : Stochastic process sample path(s) to plot
        figsize (tuple) : Size of the figure to generate

    Returns:
        N/A
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.linspace(0, len(samples), len(samples)), samples, lw=1.0)
    fig.tight_layout()
    plt.show()


def static_plot_2D(x_samples, y_samples, figsize=(16, 9)):
    """ Plot 2D stochastic process samples statically (still plot).

    Args:
        x_samples (array) : Stochastic process sample path(s) to plot (x-dir)
        y_samples (array) : Stochastic process sample path(s) to plot (y-dir)
        figsize (tuple) : Size of the figure to generate

    Returns:
        N/A
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x_samples, y_samples, lw=1.0)
    fig.tight_layout()
    plt.show()


def static_plot_3D(x_samples, y_samples, z_samples, figsize=(16, 9)):
    """ Plot 3D stochastic process samples statically (still plot).

    Args:
        x_samples (array) : Stochastic process sample path(s) to plot (x-dir)
        y_samples (array) : Stochastic process sample path(s) to plot (y-dir)
        z_samples (array) : Stochastic process sample path(s) to plot (z-dir)
        figsize (tuple) : Size of the figure to generate

    Returns:
        N/A
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for path in range(x_samples.shape[1]):
        ax.plot3D(x_samples[:, path], y_samples[:, path],
                  z_samples[:, path], lw=1.0)

    fig.tight_layout()
    plt.show()
