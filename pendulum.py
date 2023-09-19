import numpy as np

from ode import *


class Pendulum(ODEModel):
    def __init__(self, M, L=1, g=9.81) -> None:
        self.g = g  # gravity [m^2/s]
        self.L = L  # length of rod [m]
        self.M = M  # mass of pendulum [kg]

    @property
    def num_states(self) -> int:
        return 2

    def __call__(self, u: np.ndarray) -> np.ndarray:
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
