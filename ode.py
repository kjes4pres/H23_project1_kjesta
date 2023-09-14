import scipy as sc
import numpy as np
from typing import NamedTuple
from scipy.integrate import solve_ivp

class ODEModel():
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    
    @property
    def num_states(self) -> int:
        raise NotImplementedError
    

    def _create_result(self, solved):
        return ODEResult(time=solved.t, solved=solved.y)
    

    def solve(self, u0: np.ndarray, T: float, dt: float, method: str = "RK45"):
        timespan = (0, T)
        t_eval = np.linspace(0, T + 1, dt)
        solved = solve_ivp(self, timespan, u0, t_eval = t_eval)
        return self._create_result(solved)

class ODEResult(NamedTuple):
    time: np.ndarray
    solution: np.ndarray


