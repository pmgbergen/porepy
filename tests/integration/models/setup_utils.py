"""Utility methods for setting up models for testing."""
from __future__ import annotations

import inspect

import numpy as np

import porepy as pp


class RectangularDomainOrthogonalFractures2d(pp.ModelGeometry):
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
            p = np.array([[0, 2, 0.5, 0.5], [0.5, 0.5, 0, 1]]) * ls
            e = np.array([[0, 2], [1, 3]])
        elif num_fracs == 3:
            p = np.array([[0, 2, 0.5, 0.5, 0.3, 0.7], [0.5, 0.5, 0, 1, 0.3, 0.7]]) * ls
            e = np.array([[0, 2, 4], [1, 3, 5]])
        else:
            raise ValueError("Only up to 3 fractures supported.")
        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        # Length scale:
        ls = 1 / self.units.m
        return {"mesh_size_frac": 0.5 * ls, "mesh_size_bound": 0.5 * ls}


class OrthogonalFractures3d(pp.ModelGeometry):
    """A 3d domain with up to three orthogonal fractures.

    The fractures have constant x, y and z coordinates equal to 0.5, respectively,
    and are situated in a unit cube domain. The number of fractures is controlled by
    the parameter num_fracs, which can be 0, 1, 2 or 3.

    """

    def set_fracture_network(self) -> None:
        """Set the fracture network.

        The fractures are stored in self.fracture_network.

        """
        # Length scale:
        ls = 1 / self.units.m

        num_fracs = self.params.get("num_fracs", 1)
        domain = pp.grids.standard_grids.utils.unit_domain(3)
        pts = []
        if num_fracs > 0:
            # The three fractures are defined by pertubations of the coordinate arrays.
            coords_a = [0.5, 0.5, 0.5, 0.5]
            coords_b = [0, 0, 1, 1]
            coords_c = [0, 1, 1, 0]
            pts.append(np.array([coords_a, coords_b, coords_c]) * ls)
        if num_fracs > 1:
            pts.append(np.array([coords_b, coords_a, coords_c]) * ls)
        if num_fracs > 2:
            pts.append(np.array([coords_b, coords_c, coords_a]) * ls)
        fractures = [pp.PlaneFracture(p) for p in pts]
        self.fracture_network = pp.FractureNetwork3d(fractures, domain)

    def mesh_arguments(self) -> dict:
        # Length scale:
        ls = 1 / self.units.m
        mesh_sizes = {
            "mesh_size_frac": 0.5 * ls,
            "mesh_size_bound": 0.5 * ls,
            "mesh_size_min": 0.2 * ls,
        }
        return mesh_sizes


class NoPhysics(pp.ModelGeometry, pp.SolutionStrategy, pp.DataSavingMixin):
    """A model with no physics, for testing purposes.

    The model comes with minimal physical properties, making testing of individual
    components (e.g. constitutive laws) easier.

    """

    def create_variables(self):
        pass

    def set_equations(self):
        pass


class MassBalance(
    RectangularDomainOrthogonalFractures2d,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    ...


class MomentumBalance(
    RectangularDomainOrthogonalFractures2d,
    pp.momentum_balance.MomentumBalance,
):
    """Combine components needed for momentum balance simulation."""

    pass


class MassAndEnergyBalance(
    RectangularDomainOrthogonalFractures2d,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Combine components needed for force balance simulation."""

    pass


class Poromechanics(
    RectangularDomainOrthogonalFractures2d,
    pp.poromechanics.Poromechanics,
):
    """Combine components needed for poromechanics simulation."""

    pass


class Thermoporomechanics(
    RectangularDomainOrthogonalFractures2d,
    pp.thermoporomechanics.Thermoporomechanics,
):
    """Combine components needed for poromechanics simulation."""

    pass


def model(
    model_type: str, dim: int, num_fracs: int = 1
) -> MassBalance | MomentumBalance | MassAndEnergyBalance | Poromechanics:
    """Setup for tests."""
    # Suppress output for tests
    params = {"suppress_export": True, num_fracs: num_fracs}

    # To define the model we comine a geometry with a physics class.
    # First identify the two component classes from the user input, and then combine
    # them in a new class.

    # Identify the geometry class
    if dim == 2:
        geometry = RectangularDomainOrthogonalFractures2d
    elif dim == 3:
        geometry = OrthogonalFractures3d
    else:
        raise ValueError(f"Unknown dimension {dim}")

    # Identify the physics class
    if model_type == "mass_balance":
        model_class = pp.fluid_mass_balance.SinglePhaseFlow
    elif model_type == "momentum_balance":
        model_class = pp.momentum_balance.MomentumBalance
    elif model_type == "energy_balance" or model_type == "mass_and_energy_balance":
        model_class = pp.mass_and_energy_balance.MassAndEnergyBalance
    elif model_type == "poromechanics":
        model_class = pp.poromechanics.Poromechanics
    elif model_type == "thermoporomechanics":
        model_class = pp.thermoporomechanics.Thermoporomechanics
    else:
        # To add a new model, insert an elif clause here, and a new class above.
        raise ValueError(f"Unknown model type {model_type}")

    # Combine the two classes
    class Model(geometry, model_class):
        pass

    # Create an instance of the combined class
    model = Model(params)

    # Prepare the simulation
    # (create grids, variables, equations, discretize, etc.)
    model.prepare_simulation()
    return model


def domains_from_method_name(
    mdg: pp.MixedDimensionalGrid,
    method_name: str,
    domain_dimension: int,
) -> list[pp.Grid] | list[pp.MortarGrid]:
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
