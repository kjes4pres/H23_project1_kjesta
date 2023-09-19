import numpy as np
from typing import NamedTuple
from scipy.integrate import solve_ivp
import abc
from typing import Optional, List
import matplotlib.pyplot as plt


class InvalidInitialConditionError(RuntimeError):
    pass


class ODEModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        pass

    @property
    def num_states(self) -> int:
        raise NotImplementedError

    def _create_result(self, solution):
        return ODEResult(time=solution.t, solution=solution.y)

    def solve(self, u0: np.ndarray, T: float, dt: float, method: str = "RK45"):
        if len(u0) == self.num_states:
            timespan = (0, T)
            t_eval = np.arange(0, T + dt, dt)
            solution = solve_ivp(self, timespan, u0, t_eval=t_eval)
            return self._create_result(solution)
        else:
            raise InvalidInitialConditionError


class ODEResult(NamedTuple):
    time: np.ndarray
    solution: np.ndarray

    @property
    def num_states(self):
        return self.solution.shape[0]

    @property
    def num_timepoints(self):
        return self.solution.shape[1]


def plot_ode_solution(
    results: ODEResult,
    state_labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> None:
    '''
    Makes plot of solution to an ODE. 

    Args.:
    results: instance of the ODEResult NamedTuple, contains two numpy arrays.
    state_labels: Optional list of the state variables names.
    filename: optional, should be given if you want to save the figure.

    Return: None, only matplotlib pyplot. 
    '''
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("ODE solution")
    plt.grid()
    if state_labels:
        plt.plot(results.time, results.solution[0], label=state_labels)
        plt.legend()
    else:
        plt.plot(results.time, results.solution[0])

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
