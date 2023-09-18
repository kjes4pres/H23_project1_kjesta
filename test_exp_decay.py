import pytest
import numpy as np

from exp_decay import ExponentialDecay
from ode import ODEModel, ODEResult, InvalidInitialConditionError


def test_negative_decay_raises_ValueError_constructor():
    with pytest.raises(ValueError):
        model = ExponentialDecay(-1)


def test_rhs():
    # Test example:
    a = 0.4
    u = np.array([3.2])
    t = 0.0
    model = ExponentialDecay(a)
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


def test_num_states():
    tol = 1e-6
    expected = 1  # wanted number of states for instance of this class
    model = ExponentialDecay(0.4)
    computed = model.num_states  # actual number of states for instance of this class
    diff = abs(computed - expected)
    success = diff < tol
    message = "Number of states should be 1 for an instance of ExponentialDecay class, not {computed}."
    assert success, message

    with pytest.raises(AttributeError):
        model = ExponentialDecay(0.4)
        model.num_states = 4


def test_solve_with_different_number_of_initial_states():
    model = ExponentialDecay(0.4)
    u0_wrong = np.array([1, 1])
    with pytest.raises(InvalidInitialConditionError):
        model.solve(u0 = u0_wrong, T = 10, dt = 0.01)
