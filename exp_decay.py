import scipy as sc
import numpy as np

class ExponentialDecay():
    def __init__(self, decay_constant) -> None:
        if decay_constant < 0:
            raise ValueError("The decay constant cannot be negative.")
        else:
            self.decay_constant = decay_constant


    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        ...
