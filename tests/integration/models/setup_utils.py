"""Utility methods for setting up models for testing."""
from __future__ import annotations

import numpy as np

import porepy as pp


class GeometrySingleFracture2d(pp.ModelGeometry):
    def set_fracture_network(self) -> None:

        num_fracs = self.params.get("num_fracs", 1)
        domain = {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 1}
        if num_fracs == 0:
            p = np.zeros((2, 0), dtype=float)
            e = np.zeros((2, 0), dtype=int)
        elif num_fracs == 1:
            p = np.array([[0, 2], [0.5, 0.5]])
            e = np.array([[0], [1]])
        elif num_fracs == 2:
            p = np.array([[0, 2, 0.5, 0.5], [1, 1, 0, 1]])
            e = np.array([[0, 2], [1, 3]])
        else:
            raise ValueError("Only 0, 1 or 2 fractures supported.")
        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_bound": 0.5}


class MassBalance(
    GeometrySingleFracture2d,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    ...


class MomentumBalance(
    GeometrySingleFracture2d,
    pp.momentum_balance.MomentumBalance,
):
    """Combine components needed for momentum balance simulation."""

    pass


class MassAndEnergyBalance(
    GeometrySingleFracture2d,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Combine components needed for force balance simulation."""

    pass


class Poromechanics(
    GeometrySingleFracture2d,
    pp.poromechanics.Poromechanics,
):
    """Combine components needed for poromechanics simulation."""

    pass


class Thermoporomechanics(
    GeometrySingleFracture2d,
    pp.thermoporomechanics.Thermoporomechanics,
):
    """Combine components needed for poromechanics simulation."""

    pass


def model(
    model_type: str, num_fracs: int = 1
) -> MassBalance | MomentumBalance | MassAndEnergyBalance | Poromechanics:
    """Setup for tests."""
    # Suppress output for tests
    params = {"suppress_export": True, num_fracs: num_fracs}

    ob: MassBalance | MomentumBalance | MassAndEnergyBalance | Poromechanics

    # Choose model and create setup object
    if model_type == "mass_balance":
        ob = MassBalance(params)
    elif model_type == "momentum_balance":
        ob = MomentumBalance(params)
    elif model_type == "energy_balance" or model_type == "mass_and_energy_balance":
        ob = MassAndEnergyBalance(params)
    elif model_type == "poromechanics":
        ob = Poromechanics(params)
    elif model_type == "thermoporomechanics":
        ob = Thermoporomechanics(params)
    else:
        # To add a new model, insert an elif clause here, and a new class above.
        raise ValueError(f"Unknown model type {model_type}")

    # Prepare the simulation
    # (create grids, variables, equations, discretize, etc.)
    ob.prepare_simulation()
    return ob
