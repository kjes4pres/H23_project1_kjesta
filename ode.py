import scipy as sc
import numpy as np

class ODEModel():
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    
    @property
    def num_states(self) -> int:
        raise NotImplementedError

