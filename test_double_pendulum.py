import numpy as np
import pytest

from double_pendulum import *


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5, 3.386187037),
        (0.5, 0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ],
)
def test_domega1_dt(theta1, theta2, expected):
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = model(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5, -7.704787325),
        (0.5, 0, 6.768494455),
        (0.5, 0.5, 0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = model(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)


def test_derivatives_at_rest_is_zero():
    t = 0.0
    u = np.array([0, 0, 0, 0])
    du_dt_expected = u
    model = DoublePendulum()
    du_dt_computed = model(t, u)

    for (computed, expected) in zip(du_dt_computed, du_dt_expected):
        success = computed == expected
        msg = f"Computed derivative, {computed}, does not match the analytical, {expected}."
        assert success, msg


def test_solve_pendulum_ode_with_zero_ic():
    u0 = np.array([0, 0, 0, 0])
    T = 10
    dt = 0.01

    model = DoublePendulum()
    computed = model.solve(u0, T, dt)

    comp_solution_theta = computed.solution[0]
    comp_solution_omega = computed.solution[1]
    msg = f"For the inital condition {u0}, the solved ODEs should also be only zeros."
    assert np.all(comp_solution_theta == 0), msg
    assert np.all(comp_solution_omega == 0), msg


def test_solve_double_pendulum_function_zero_ic():
    u0 = np.array([0, 0, 0, 0])
    T = 10
    dt = 0.01

    model = DoublePendulum()
    solved_model = model.solve(u0, T, dt)

    properties = ["theta1", "theta2", "omega1", "omega2", "x1", "x2", "y1", "y2"]
    expected = [0, 0, 0, 0, 0, 0, -model.L1, -(model.L1 + model.L2)]

    for property, expected in zip(properties, expected):
        computed = getattr(solved_model, property)
        msg = f"The initial condition {u0} should give {property} with values of {expected}, not {computed}."
        assert np.all(computed == expected), msg
