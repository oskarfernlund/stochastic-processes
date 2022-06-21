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

SEED = 69
N_STEPS = 1000
N_PATHS = 3


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
        self.options = [(+1,), (-1,)]
        self.rng = np.random.default_rng(seed)

    def sample(self, steps, paths=1):
        """ Generate random walk sample paths.
        
        1st dimension = steps, 2nd dimension = paths, 3rd dimension = 
        coordinates.

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
        self.options = [(+1, 0), (-1, 0), (0, +1), (0, -1)]
        self.rng = np.random.default_rng(seed)

    def sample(self, steps, paths=1):
        """ Generate random walk sample paths.
        
        1st dimension = steps, 2nd dimension = paths, 3rd dimension = 
        coordinates.

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
    plt.show()


def plot_2D_samples(samples, figsize=(8, 5)):
    """ Plot sample paths from a 2D random walk (x vs. y).

    Args:
        samples (array) : sample path(s) to plot
        figsize (tuple) : size of the figure to generate
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(samples[:, :, 0], samples[:, :, 1], lw=1)
    plt.show()


def count_hitting_times(samples, area):
    pass


def plot_histogram(data):
    pass


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
    plot_2D_samples(samples2D)