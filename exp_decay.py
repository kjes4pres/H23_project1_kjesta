import scipy as sc
import numpy as np

class ExponentialDecay():

    def __init__(self, decay_constant) -> None:
        if decay_constant < 0:
            raise ValueError("The decay constant cannot be negative.")
        else:
            self.decay_constant = decay_constant


    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of u at time t.

        Args:
        t (float): time
        u (array) : value of function

        Returns:
        array, derivative of u at time t.
        '''
        return -self.decay_constant*u
