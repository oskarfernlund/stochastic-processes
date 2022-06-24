#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes and functions pertaining to random walks.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

SEED = 0
N_STEPS = 10000
N_PATHS = 10000


# =============================================================================
#  CLASSES
# =============================================================================

class SimpleRandomWalk1D:
    """ One dimensional simple random walk class. """

    def __init__(self, seed=0):
        """ Set the step options and random number generator.
        
        Args:
            seed (int) : seed for the random number generator 
        """
        self.dim = 1
        self.options = [(+10,), (-10,)]
        self.rng = np.random.default_rng(seed)

    def sample(self, steps, paths=1):
        """ Generate random walk sample paths.
        
        1st dimension = steps,
        2nd dimension = paths,
        3rd dimension = coordinates.

        Args:
            steps (int) : number of steps to take
            paths (int) : number of paths to sample

        Returns:
            samples (numpy.ndarray) : sampled paths
        """
        samples = self.rng.choice(self.options, size=(steps, paths))
        samples = np.vstack((np.zeros((1, paths, self.dim)), samples))
        samples = np.cumsum(samples, axis=0)

        return samples


class SimpleRandomWalk2D:
    """ Two dimensional simple random walk class. """
    
    def __init__(self, seed=0):
        """ Set the step options and random number generator.
        
        Args:
            seed (int) : seed for the random number generator 
        """
        self.dim = 2
        self.options = [(+10, 0), (-10, 0), (0, +10), (0, -10)]
        self.rng = np.random.default_rng(seed)

    def sample(self, steps, paths=1):
        """ Generate random walk sample paths.
        
        1st dimension = steps, 
        2nd dimension = paths, 
        3rd dimension = coordinates.

        Args:
            steps (int) : number of steps to take
            paths (int) : number of paths to sample

        Returns:
            samples (numpy.ndarray) : sampled paths
        """
        samples = self.rng.choice(self.options, size=(steps, paths))
        samples = np.vstack((np.zeros((1, paths, self.dim)), samples))
        samples = np.cumsum(samples, axis=0)

        return samples


# =============================================================================
#  FUNCTIONS
# =============================================================================

def box_check(x, y):
    """ Check if on a box. 
    
    Args:
        x (numpy.ndarray) : x coordinates
        y (numpy.ndarray) : y coordinates

    Returns:
        (np.ndarray) : boolean array indicating when the condition is met
    """
    return np.logical_or(abs(x) == 20, abs(y) == 20)


def line_check(x, y):
    """ Check if on a line defined by y = 10 - x. 
    
    Args:
        x (numpy.ndarray) : x coordinates
        y (numpy.ndarray) : y coordinates

    Returns:
        (np.ndarray) : boolean array indicating when the condition is met
    """
    return y == 10 - x


def circle_check(x, y):
    """ Check if outside a circle. 
    
    Args:
        x (numpy.ndarray) : x coordinates
        y (numpy.ndarray) : y coordinates

    Returns:
        (np.ndarray) : boolean array indicating when the condition is met
    """
    return ((x - 2.5) / 30) ** 2 + ((y - 2.5) / 40) ** 2 >= 1


def compute_hitting_times_2D(samples, boundary_func):
    """ Compute the random walk hitting times for a given boundary. 
    
    Args:
        samples (numpy.ndarray) : samples paths (2D)
        boundary_func (function) : function for checking the boundary

    Returns:
        hitting_times (numpy.ndarray) : hitting times for the given boundary
    """
    x, y = samples[:, :, 0], samples[:, :, 1]
    on_boundary = boundary_func(x, y)
    hitting_times = on_boundary.argmax(axis=0)
    hitting_times[hitting_times == 0] = N_STEPS
    return hitting_times


def plot_1D_samples(samples, figsize=(8, 5)):
    """ Plot sample paths from a 1D random walk (vs. number of steps).

    Args:
        samples (array) : sample path(s) to plot
        figsize (tuple) : size of the figure to generate
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.linspace(0, (samples).shape[0] - 1, samples.shape[0]), 
            samples[:, :, 0], lw=1)
    ax.set_xlabel("steps")


def plot_2D_samples(samples, figsize=(8, 5)):
    """ Plot sample paths from a 2D random walk (x vs. y).

    Args:
        samples (array) : sample path(s) to plot
        figsize (tuple) : size of the figure to generate
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(samples[:, :, 0], samples[:, :, 1], lw=1)


def plot_hitting_times(hitting_times, figsize=(8, 5)):
    """ Plot a histogram of hitting times. 
    
    Args:
        hitting_times (numpy.ndarray) : hitting times to plot
        figsize (tuple) : size of the figure to generate
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(hitting_times, color="C0", alpha=0.5)
    ax.set_xlabel("hitting time")
    ax.set_ylabel("frequency")


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    # 1D random walk
    # randomwalk1D = SimpleRandomWalk1D(seed=SEED)
    # samples1D = randomwalk1D.sample(steps=N_STEPS, paths=N_PATHS)
    # plot_1D_samples(samples1D)

    # 2D random walk
    randomwalk2D = SimpleRandomWalk2D(seed=SEED)
    samples2D = randomwalk2D.sample(steps=N_STEPS, paths=N_PATHS)
    hitting_times = compute_hitting_times_2D(samples2D, circle_check)
    print(f"\nAvg. hitting time: {np.average(hitting_times)}s\n")
    plot_hitting_times(hitting_times)
    # plot_2D_samples(samples2D)
    plt.show()