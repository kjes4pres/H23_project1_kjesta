import numpy as np
from dataclasses import dataclass

from ode import *


@dataclass
class PendulumResults:
    time: np.ndarray
    solution: np.ndarray
    L: float
    g: float

    @property
    def theta(self) -> np.ndarray:
        return self.solution[0]

    @property
    def omega(self) -> np.ndarray:
        return self.solution[1] 
    
    @property
    def x(self) -> np.ndarray:
        x = self.L*np.sin(self.theta)
        return x
    
    @property
    def y(self) -> np.ndarray:
        y = -self.L*np.cos(self.theta)
        return y 
    

    @property
    def potential_energy(self) -> np.ndarray:
        P = self.g*(self.y + self.L)
        return P
    
    @property
    def velocity_x(self) -> np.ndarray:
        vx = np.gradient(self.x, self.time)
        return vx
    
    @property
    def velocity_y(self) -> np.ndarray:
        vy = np.gradient(self.y, self.time)
        return vy
    

    @property
    def kinetic_energy(self) -> np.ndarray:
        vx = self.velocity_x
        vy = self.velocity_y
        K = (1/2)*(vx*vx + vy*vy)
        return K
    

    @property
    def total_energy(self) -> np.ndarray:
        T = self.potential_energy + self.kinetic_energy
        return T



class Pendulum(ODEModel):
    def __init__(self, M = 1, L=1, g=9.81) -> None:
        self.g = g  # gravity [m^2/s]
        self.L = L  # length of rod [m]
        self.M = M  # mass of pendulum [kg]

    @property
    def num_states(self) -> int:
        return 2

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Finds time derivative of position and angular velocity of pendulum.

        Input:
        u: numpy array, first index as theta (position), second index as omega (angular velocity)

        Returns:
        du_dt: time derivative of u.
        """
        theta = u[0]
        omega = u[1]

        theta_dt = omega
        omega_dt = -(self.g / self.L) * np.sin(theta)

        du_dt = np.array([theta_dt, omega_dt])
        return du_dt
    

    def _create_result(self, solution) -> PendulumResults:
        return PendulumResults(solution.t, solution.y, self.L, self.g)
    

def exercise_2b():
    '''
    Making an instance of the class Pendulum,
    and solving the ODE. 
    '''
    model = Pendulum(M = 1)
    u0 = np.array([np.pi/6, 0.35])
    T = 10.0
    dt = 0.01

    result = model.solve(u0, T, dt)
    plot_ode_solution(results = result, state_labels = ["theta", "omega"], filename = "exercise_2b.png")



if __name__ == "__main__":
    exercise_2b()
