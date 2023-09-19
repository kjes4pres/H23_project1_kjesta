import numpy as np
import pytest

from pendulum import *

tol = 1e-2


def test_rhs():
    u = np.array([np.pi / 6, 0.35])
    du_dt_expected = np.array([0.35, -3.45])
    model = Pendulum(M=1, L=1.42)
    du_dt_computed = model(u)

    for i, (comp, exp) in enumerate(zip(du_dt_computed, du_dt_expected)):
        diff = abs(comp - exp)
        success = diff < tol
        msg = f"Computed derivative, {comp}, does not match the analytical, {exp}."
        assert success, msg


def test_pendulum_at_equilibrium_stays_at_rest():
    u = np.array([0, 0])
    du_dt_expected = u
    model = Pendulum(M=1)
    du_dt_computed = model(u)

    for i, (comp, exp) in enumerate(zip(du_dt_computed, du_dt_expected)):
        success = comp == exp
        msg = f"Computed derivative, {comp}, does not match the analytical, {exp}."
        assert success, msg
