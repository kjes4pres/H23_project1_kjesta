import scipy as sc
import numpy as np
import pytest

from exp_decay import ExponentialDecay

def test_negative_decay_raises_ValueError_constructor():
    with pytest.raises(ValueError):
        instance = ExponentialDecay(-1)


def test_rhs():
    # Test example:
    a = 0.4
    u = np.array([3.2])
    t = 0.0
    model = ExponentialDecay(0.4)
    computed = model(t, u)

    # Variable expected is what the computed should be close to.
    tol = 1e-6
    expected = -1.28

    diff = abs(computed - expected)
    success = diff < tol
    message = "Computed derivative differs from the expected value with {diff}."
    assert success, message

def test_negative_decay_raises_ValueError():
    with pytest.raises(ValueError):
        model = ExponentialDecay(0.4)
        model.decay_constant = -1