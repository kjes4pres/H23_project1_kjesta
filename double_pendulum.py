import numpy as np
from dataclasses import dataclass

from pendulum import *

class DoublePendulum(ODEModel):
    def __init__(self, L1 = 1, L2 = 1, g = 9.81) -> None:
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        theta1, omega1, theta2, omega2 = u
        
        dtheta1_dt = omega1
        dtheta2_dt  = omega2

        dtheta = theta2 - theta1

        numerator1 = (self.L1*(omega1**2)*np.sin(dtheta)*np.cos(dtheta)) +  (self.g*np.sin(theta2)*np.cos(dtheta)) + (self.L2*(omega2**2)*np.sin(dtheta)) - (2*self.g*np.sin(theta1))
        numerator2 = (-self.L2*(omega2**2)*np.sin(dtheta)*np.cos(dtheta)) +  (2*self.g*np.sin(theta1)*np.cos(dtheta)) - (2*self.L1*(omega1**2)*np.sin(dtheta)) - (2*self.g*np.sin(theta2))

        denominator1 = 2*self.L1 - self.L1*(np.cos(dtheta)*np.cos(dtheta))
        denominator2 = 2*self.L2 - self.L2*(np.cos(dtheta)*np.cos(dtheta))

        domega1_dt = numerator1 / denominator1
        domega2_dt = numerator2 / denominator2

        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])
    

    @property
    def num_states(self) -> int:
        return 4
    
    