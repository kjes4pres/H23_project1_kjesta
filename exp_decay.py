from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

from ode import *


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
            raise ValueError(
                f"The decay constant must be non-negative, not {new_decay_constant}!"
            )
        else:
            self._decay_constant = new_decay_constant

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of u at time t.

        Args:
        t (float): time
        u (array) : value of function

        Returns:
        array, derivative of u at time t.
        """
        return -self.decay_constant * u

    @property
    def num_states(self) -> int:
        return 1


if __name__ == "__main__":
    # instantiating class with parameter = 0.4:
    model = ExponentialDecay(0.4)
    u = np.array([5])  # just an example
    initial_condition = np.array([1])  # semi-randomly chosen
    timespan = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    solved = solve_ivp(model, timespan, initial_condition, t_eval = t_eval)

    plt.figure()
    plt.title("Solved ODE for exponential decay")
    plt.plot(solved.t, solved.y[0], label="$a = 0.4$ \n $u_0 = 1$")
    plt.xlabel("time")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
    # model = ExponentialDecay(0.4)
    # result = model.solve(u0=np.array([4.0]), T=10.0, dt=0.01)
    # plot_ode_solution(results = result, state_labels = ["u"], filename="exponential_decay")

