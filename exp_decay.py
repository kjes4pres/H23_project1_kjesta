import scipy as sc
import numpy as np

from ode import ODEModel

class ExponentialDecay(ODEModel):

    def __init__(self, decay_constant) -> None:
        if decay_constant < 0:
            raise ValueError("The decay constant cannot be negative.")
        else:
            self._decay_constant = decay_constant

    
    @property
    def decay_constant(self):
        return self._decay_constant


    @decay_constant.setter
    def decay_constant(self, new_decay_constant):
        if new_decay_constant < 0:
            raise ValueError(f"The decay constant must be non-negative, not {new_decay_constant}!")
        else:
            self._decay_constant = new_decay_constant


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
