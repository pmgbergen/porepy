"""Utility methods for setting up models for testing."""
from __future__ import annotations

import numpy as np

import porepy as pp


class GeometrySingleFracture2d(pp.ModelGeometry):
    def set_fracture_network(self) -> None:

        num_fracs = self.params.get("num_fracs", 1)
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        if num_fracs == 0:
            p = np.zeros((2, 0), dtype=float)
            e = np.zeros((2, 0), dtype=int)
        elif num_fracs == 1:
            p = np.array([[0, 1], [0.5, 0.5]])
            e = np.array([[0], [1]])
        elif num_fracs == 2:
            p = np.array([[0, 1, 0.5, 0.5], [0.5, 0.5, 0, 1]])
            e = np.array([[0, 2], [1, 3]])
        else:
            raise ValueError("Only 0, 1 or 2 fractures supported")
        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_bound": 0.5}


class MassBalanceCombined(
    GeometrySingleFracture2d,
    pp.fluid_mass_balance.MassBalanceEquations,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    pp.fluid_mass_balance.ConstitutiveLawsSinglePhaseFlow,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.DataSavingMixin,
):
    ...


class MomentumBalanceCombined(
    GeometrySingleFracture2d,
    pp.momentum_balance.MomentumBalanceEquations,
    pp.momentum_balance.ConstitutiveLawsMomentumBalance,
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
    pp.momentum_balance.VariablesMomentumBalance,
    pp.momentum_balance.SolutionStrategyMomentumBalance,
    pp.DataSavingMixin,
):
    """Combine components needed for momentum balance simulation."""

    pass


class EnergyBalanceCombined(
    GeometrySingleFracture2d,
    pp.energy_balance.EnergyBalanceEquations,
    pp.energy_balance.ConstitutiveLawsEnergyBalance,
    pp.energy_balance.VariablesEnergyBalance,
    pp.energy_balance.SolutionStrategyEnergyBalance,
    pp.energy_balance.BoundaryConditionsEnergyBalance,
    pp.DataSavingMixin,
):
    """Combine components needed for force balance simulation."""

    pass


class PoromechanicsCombined(
    GeometrySingleFracture2d,
    pp.poromechanics.ConstitutiveLawsPoromechanics,
    pp.poromechanics.VariablesPoromechanics,
    pp.poromechanics.EquationsPoromechanics,
    pp.poromechanics.SolutionStrategyPoromechanics,
    pp.poromechanics.BoundaryConditionPoromechanics,
    pp.DataSavingMixin,
):
    """Combine components needed for poromechanics simulation."""

    pass


def model(
    model_type: str, num_fracs: int = 1
) -> MassBalanceCombined | MomentumBalanceCombined:
    """Setup for tests."""
    # Suppress output for tests
    params = {"suppress_export": True, num_fracs: num_fracs}

    ob: MassBalanceCombined | MomentumBalanceCombined

    # Choose model and create setup object
    if model_type == "mass_balance":
        ob = MassBalanceCombined(params)
    elif model_type == "momentum_balance":
        ob = MomentumBalanceCombined(params)
    elif model_type == "energy_balance":
        ob = EnergyBalanceCombined(params)
    elif model_type == "poromechanics":
        ob = PoromechanicsCombined(params)
    else:
        # To add a new model, a an elif clause here, and a new class above.
        raise ValueError(f"Unknown model type {model_type}")

    # Prepare the simulation
    # (create grids, variables, equations, discretize, etc.)
    ob.prepare_simulation()
    return ob
