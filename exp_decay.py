import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

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
    

    @property
    def num_states(self) -> int:
        return 1
    

if __name__ == "__main__":
    # instantiating class with parameter = 0.4:
    model = ExponentialDecay(0.4)
    t0 = 0
    T = 10
    u = np.array([5])
    timespan = (t0, T)
    initial_condition = model(0, u)

    n = T/0.01
    time_points = np.arange(0,11,n)

    result = sc.integrate.solve_ivp(model, timespan, y0 = initial_condition, t_eval = time_points)

    plt.figure()
    plt.plot(result.t, result.y[0])
    plt.show()


