import numpy as np
from typing import NamedTuple
from scipy.integrate import solve_ivp
import abc

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
        return ODEResult(time = solution.t, solution = solution.y)

 
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
