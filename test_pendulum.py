import numpy as np
import pytest

from pendulum import *

tol = 1e-2


def test_rhs():
    t = 0.0
    u = np.array([np.pi / 6, 0.35])
    du_dt_expected = np.array([0.35, -3.45])
    model = Pendulum(M=1, L=1.42)
    du_dt_computed = model(t,u)

    for i, (comp, exp) in enumerate(zip(du_dt_computed, du_dt_expected)):
        diff = abs(comp - exp)
        success = diff < tol
        msg = f"Computed derivative, {comp}, does not match the analytical, {exp}."
        assert success, msg


def test_pendulum_at_equilibrium_stays_at_rest():
    t = 0.0
    u = np.array([0, 0])
    du_dt_expected = u
    model = Pendulum(M=1)
    du_dt_computed = model(t,u)

    for i, (comp, exp) in enumerate(zip(du_dt_computed, du_dt_expected)):
        success = comp == exp
        msg = f"Computed derivative, {comp}, does not match the analytical, {exp}."
        assert success, msg


def test_solve_pendulum_ode_with_zero_ic():
    u0 = np.array([0, 0])
    T = 10
    dt = 0.01

    model = Pendulum()
    computed = model.solve(u0, T, dt)

    comp_solution_theta = computed.solution[0]
    comp_solution_omega = computed.solution[1]
    msg  = f"For the inital condition {u0}, the solved ODEs should also be only zeros."
    assert np.all(comp_solution_theta == 0), msg
    assert np.all(comp_solution_omega == 0), msg
