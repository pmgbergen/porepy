"""Utility methods for setting up models for testing."""
from __future__ import annotations

import numpy as np

import porepy as pp
from porepy.models import fluid_mass_balance as fmb
from porepy.models import momentum_balance as fb


class GeometrySingleFracture2d(pp.ModelGeometry):
    def set_fracture_network(self) -> None:
        p = np.array([[0, 1], [0.5, 0.5]])
        e = np.array([[0], [1]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_bound": 0.5}


class MassBalanceCombined(
    GeometrySingleFracture2d,
    fmb.MassBalanceEquations,
    fmb.ConstitutiveEquationsCompressibleFlow,
    fmb.VariablesSinglePhaseFlow,
    fmb.SolutionStrategyCompressibleFlow,
    pp.DataSavingMixin,
):
    ...


class MomentumBalanceCombined(
    GeometrySingleFracture2d,
    fb.MomentumBalanceEquations,
    fb.ConstitutiveEquationsMomentumBalance,
    fb.VariablesMomentumBalance,
    fb.SolutionStrategyMomentumBalance,
    pp.DataSavingMixin,
):
    """Combine components needed for force balance simulation."""

    pass


def model(model_type: str) -> MassBalanceCombined | MomentumBalanceCombined:
    """Setup for tests."""
    # Suppress output for tests
    params = {"suppress_export": True}

    ob: MassBalanceCombined | MomentumBalanceCombined

    # Choose model and create setup object
    if model_type == "mass_balance":
        ob = MassBalanceCombined(params)
    elif model_type == "momentum_balance":
        ob = MomentumBalanceCombined(params)
    else:
        # To add a new model, a an elif clause here, and a new class above.
        raise ValueError(f"Unknown model type {model_type}")

    # Prepare the simulation
    # (create grids, variables, equations, discretize, etc.)
    ob.prepare_simulation()
    return ob
