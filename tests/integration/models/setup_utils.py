"""Utility methods for setting up models for testing."""
from __future__ import annotations

import numpy as np

import porepy as pp
from porepy.models import energy_balance, fluid_mass_balance, momentum_balance


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
    fluid_mass_balance.MassBalanceEquations,
    fluid_mass_balance.ConstitutiveLawsSinglePhaseFlow,
    fluid_mass_balance.VariablesSinglePhaseFlow,
    fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.DataSavingMixin,
):
    ...


class MomentumBalanceCombined(
    GeometrySingleFracture2d,
    momentum_balance.MomentumBalanceEquations,
    momentum_balance.ConstitutiveLawsMomentumBalance,
    momentum_balance.VariablesMomentumBalance,
    momentum_balance.SolutionStrategyMomentumBalance,
    pp.DataSavingMixin,
):
    """Combine components needed for momentum balance simulation."""

    pass


class EnergyBalanceCombined(
    GeometrySingleFracture2d,
    energy_balance.EnergyBalanceEquations,
    energy_balance.ConstitutiveLawsEnergyBalance,
    energy_balance.VariablesEnergyBalance,
    energy_balance.SolutionStrategyEnergyBalance,
    energy_balance.BoundaryConditionsEnergyBalance,
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
    elif model_type == "energy_balance":
        ob = EnergyBalanceCombined(params)
    else:
        # To add a new model, a an elif clause here, and a new class above.
        raise ValueError(f"Unknown model type {model_type}")

    # Prepare the simulation
    # (create grids, variables, equations, discretize, etc.)
    ob.prepare_simulation()
    return ob
