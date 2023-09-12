import scipy as sc
import numpy as np
import pytest

from exp_decay import ExponentialDecay

def test_constructor_raises_ValueError_for_negatives():
    with pytest.raises(ValueError):
        instance = ExponentialDecay(-1)
