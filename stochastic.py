
# Date: 2021/09/08
# Author: Oskar Fernlund
# Description: Library of different stochastic process classes.


# Imports & Dependencies
# =============================================================================

import numpy as np


# Class Definitions
# =============================================================================

class SimpleRandomWalk:
    """ Simple random walk class.

    Can be used to generate simple random walks in 1 dimension. If multiple
    dimensions are desired, call the sample method multiple times (one call per
    dimension).

    Attributes:
        steps (list) : Possible steps

    Methods:
        __init__ : Create the random walk object, set step choices
        sample : Generate random walk sample paths
    """
    def __init__(self):
        """ Initialize the process object. Set the possible step choices. """
        self.choices = [-1, 1]

    def sample(self, steps, paths=1):
        """ Generate random walk sample paths. Rows = steps, columns = paths.

        Args:
            steps (int) : Number of steps to sample
            paths (int) : Number of paths to sample

        Returns:
            samples (array) : steps x paths array of samples
        """
        samples = np.random.choice(self.choices, size=(steps-1, paths))
        samples = np.vstack((np.zeros((1, paths)), samples))
        samples = np.cumsum(samples, axis=0)

        return samples


class GaussianRandomWalk:
    """ Description.

    Attributes:

    Methods:

    """

    def __init__(self):
        pass

    def sample(self, steps, paths=1):
        pass


class WienerProcess:
    """ Description.

    Attributes:

    Methods:

    """

    def __init__(self):
        pass

    def sample(self, steps, paths=1):
        pass


class PolyaProcess:
    """ Description.

    Attributes:

    Methods:

    """

    def __init__(self):
        pass

    def sample(self, steps, paths=1):
        pass
