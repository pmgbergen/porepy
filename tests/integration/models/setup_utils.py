"""Utility methods for setting up models for testing."""
from __future__ import annotations

import inspect

import numpy as np

import porepy as pp


class GeometrySingleFracture2d(pp.ModelGeometry):
    def set_fracture_network(self) -> None:
        # Length scale:
        ls = 1 / self.units.m

        num_fracs = self.params.get("num_fracs", 1)
        domain = {"xmin": 0, "xmax": 2 * ls, "ymin": 0, "ymax": 1 * ls}
        if num_fracs == 0:
            p = np.zeros((2, 0), dtype=float) * ls
            e = np.zeros((2, 0), dtype=int)
        elif num_fracs == 1:
            p = np.array([[0, 2], [0.5, 0.5]]) * ls
            e = np.array([[0], [1]])
        elif num_fracs == 2:
            p = np.array([[0, 2, 0.5, 0.5], [1, 1, 0, 1]]) * ls
            e = np.array([[0, 2], [1, 3]])
        else:
            raise ValueError("Only 0, 1 or 2 fractures supported.")
        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        # Length scale:
        ls = 1 / self.units.m
        return {"mesh_size_frac": 0.5 * ls, "mesh_size_bound": 0.5 * ls}


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


def domains_from_method_name(
    mdg: pp.MixedDimensionalGrid,
    method_name: str,
    domain_dimension: int,
):
    """Return the domains to be tested for a given method.

    The method to be tested is assumed to take as input only its domain of definition,
    that is a list of subdomains or interfaces. The test framework is not compatible with
    methods that take other arguments (and such a method would also break the implicit
    contract of the constitutive laws).

    Parameters:
        mdg: Mixed-dimensional grid.
        method_name: Name of the method to be tested.
        domain_dimension: Only domains of the specified dimension will be tested.

    Returns:
        list of pp.Grid or pp.MortarGrid: The domains to be tested.

    """
    # Fetch the signature of the method.
    signature = inspect.signature(method_name)
    assert len(signature.parameters) == 1

    # The domain is a list of either subdomains or interfaces.
    if "subdomains" in signature.parameters:
        # If relevant, filter out the domains that are not to be tested.
        domains = mdg.subdomains(dim=domain_dimension)
    elif "interfaces" in signature.parameters:
        domains = mdg.interfaces(dim=domain_dimension)

    return domains


# TODO: Move. Check all values.
granite_values = {
    "permeability": 1e-20,
    "density": 2700,
    "porosity": 7e-3,
    "lame_mu": 16.67 * pp.GIGA,
    "lame_lambda": 11.11 * pp.GIGA,
    "specific_heat_capacity": 790,
}
# Cf. fluid.py
water_values = {
    "specific_heat_capacity": 4180,
    "compressibility": 4e-10,
    "viscosity": 1e-3,
    "density": 1000,
    "thermal_conductivity": 0.6,
    "thermal_expansion": 2.1e-4,
}
