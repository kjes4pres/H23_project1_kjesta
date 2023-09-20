from pendulum import *


class DoublePendulum(ODEModel):
    def __init__(self, L1=1, L2=1, g=9.81) -> None:
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        theta1, omega1, theta2, omega2 = u

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        dtheta = theta2 - theta1

        numerator1 = (
            (self.L1 * (omega1**2) * np.sin(dtheta) * np.cos(dtheta))
            + (self.g * np.sin(theta2) * np.cos(dtheta))
            + (self.L2 * (omega2**2) * np.sin(dtheta))
            - (2 * self.g * np.sin(theta1))
        )
        numerator2 = (
            (-self.L2 * (omega2**2) * np.sin(dtheta) * np.cos(dtheta))
            + (2 * self.g * np.sin(theta1) * np.cos(dtheta))
            - (2 * self.L1 * (omega1**2) * np.sin(dtheta))
            - (2 * self.g * np.sin(theta2))
        )

        denominator1 = (2 * self.L1) - (self.L1 * (np.cos(dtheta) * np.cos(dtheta)))
        denominator2 = (2 * self.L2) - (self.L2 * (np.cos(dtheta) * np.cos(dtheta)))

        domega1_dt = numerator1 / denominator1
        domega2_dt = numerator2 / denominator2

        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

    @property
    def num_states(self) -> int:
        return 4
    

    def _create_result(self, solution):
        return DoublePendulumResults(solution.t, solution.y, self.L1, self.L2, self.g)

@dataclass
class DoublePendulumResults:
    time: np.ndarray
    solution: np.ndarray
    L1: float
    L2: float
    g: float

    @property
    def theta1(self) -> np.ndarray:
        return self.solution[0]
    
    @property
    def omega1(self) -> np.ndarray:
        return self.solution[1]
    
    @property
    def theta2(self) -> np.ndarray:
        return self.solution[2]
    
    @property
    def omega2(self) -> np.ndarray:
        return self.solution[3]
    

    @property
    def x1(self) -> np.ndarray:
        return self.L1*np.sin(self.theta1)
    
    @property
    def y1(self) -> np.ndarray:
        return -self.L1*np.cos(self.theta1)
    
    @property
    def x2(self) -> np.ndarray:
        return self.x1 + self.L2*np.sin(self.theta2)
    
    @property
    def y2(self) -> np.ndarray:
        return self.y1 - self.L2*np.cos(self.theta2)
    
    
    @property
    def potential_energy(self) -> np.ndarray:
        P1 = self.g*(self.y1 + self.L1)
        P2 = self.g*(self.y2 + self.L1 + self.L2)
        return P1 + P2
    
    @property
    def velocity_x1(self) -> np.ndarray:
        vx1 = np.gradient(self.x1, self.time)
        return vx1
    
    @property
    def velocity_x2(self) -> np.ndarray:
        vx2 = np.gradient(self.x2, self.time)
        return vx2
    
    
    @property
    def velocity_y1(self) -> np.ndarray:
        vy1 = np.gradient(self.y1, self.time)
        return vy1
    
    @property
    def velocity_y2(self) -> np.ndarray:
        vy2 = np.gradient(self.y2, self.time)
        return vy2
    

    @property
    def kinetic_energy(self) -> np.ndarray:
        vx1 = self.velocity_x1
        vx2 = self.velocity_x2
        vy1 = self.velocity_y1
        vy2 = self.velocity_y2

        K1 = (1/2)*(vx1*vx1 + vy1*vy1)
        K2 =(1/2)*(vx2*vx2 + vy2*vy2)
        return K1 + K2
    

    @property
    def total_energy(self) -> np.ndarray:
        T = self.potential_energy + self.kinetic_energy
        return T
    

def exercise_3d():
    u0 = np.array([np.pi/6, 0.35, 0, 0])
    T = 10.0
    dt = 0.01
    model = DoublePendulum()
    solved_model = model.solve(u0, T, dt, method = "Radau")
    plot_energy(solved_model, filename = "energy_double.png")

    
if __name__ == "__main__":
    exercise_3d()
