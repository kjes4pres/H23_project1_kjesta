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

@pytest.mark.parametrize(
        "a, u0, T, dt", ([0.4, 5, 10, 0.01], [2, 6, 20, 0.1], [5, 3, 50, 0.02])
)
def test_solve_time(a, u0, T, dt):
    u0 = np.array([u0])
    tol = 1e-6
    model = ExponentialDecay(a)
    computed = model.solve(u0, T, dt)

    # Checking that t0 = 0.
    assert abs(computed.time[0] - 0) < tol ,  f"Start time should be 0, not {computed.time[0]}"

    # Checking that tN = T.
    assert abs(computed.time[-1] - T) < tol ,  f"End time should be {T}, not {computed.time[-1]}"

    # Checking dt
    dt_computed = computed.time[1] - computed.time[0]
    assert abs(dt_computed - dt) < tol ,  f"dt should be {dt}, not {dt_computed}"


@pytest.mark.parametrize(
        "a, u0, T, dt", ([0.4, 5, 10, 0.01], [2, 6, 20, 0.1], [5, 3, 50, 0.02])
)
def test_solve_solution(a, u0, T, dt):
    tol = 0.01
    time = np.arange(0, T + dt, dt)
    exact = u0*np.exp(-a*time)

    u0 = np.array([u0])

    model = ExponentialDecay(a)
    computed = model.solve(u0, T, dt)

    relative_error = np.linalg.norm(computed.solution[0] - exact) / np.linalg.norm(exact)

    assert relative_error < tol, f"Relative error too large."






