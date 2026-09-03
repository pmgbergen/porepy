"""Unit tests for time-step constraints."""

import numpy as np

import porepy as pp
from porepy.time_stepper.time_step_constraint import CourantTimeStepConstraint


class MockSubdomain:
    def cell_diameters(self, cell_wise: bool, func):
        assert not cell_wise
        assert func is np.max
        return np.array([0.5, 1.0])


class MockMixedDimensionalGrid:
    def subdomains(self):
        return [subdomain]


class MockEquationSystem:
    def evaluate(self, operator):
        assert operator == "darcy_flux"
        return np.array([2.0, 1.0])


class MockModel(pp.SolutionStrategy):
    def darcy_flux(self, domains):
        assert domains == [subdomain]
        return "darcy_flux"


subdomain = MockSubdomain()


def test_courant_time_step_constraint():
    model = MockModel()
    model.mdg = MockMixedDimensionalGrid()
    model.equation_system = MockEquationSystem()

    constraint = CourantTimeStepConstraint(target_cfl=0.8)

    # min(cell diameter) / max(abs(Darcy flux)) * target CFL = 0.5 / 2 * 0.8.
    adjusted_dt = constraint.suggest_dt(dt=10.0, context={"model": model})

    assert adjusted_dt == 0.2
