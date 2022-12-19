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


class BoundaryConditionsThermoporomechanicsDirNorthSouth(
    pp.thermoporomechanics.BoundaryConditionsThermoporomechanics
):
    """Boundary conditions for the thermoporomechanics problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    Implementation of mechanical values facilitates time-dependent boundary conditions
    with use of :class:`pp.time.TimeDependentArray` for :math:`\nabla \cdot u` term.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, north + south, "dir")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value of 0 unless fluid's reference pressure is changed.

        Parameters:
            subdomains: List of subdomains for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        vals = []
        if len(subdomains) == 0:
            return pp.ad.Array(np.zeros(0), name="bc_values_darcy")
        for sd in subdomains:
            _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
            vals_loc = np.zeros(sd.num_faces)
            vals_loc[north + south] = self.fluid.pressure()
            vals.append(vals_loc)
        return pp.wrap_as_ad_array(np.hstack(vals), name="bc_values_darcy")

    def bc_type_mechanics(self, sd):
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, north + south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_type_mobrho(self, sd):
        """Boundary condition type for the density-mobility product.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, north + south, "dir")

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, north + south, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, north + south, "dir")

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary condition values for the mobility.

        Nonzero values are only defined on the north and south boundaries corresponding
        to the reference value of the density-mobility product.

        Parameters:
            subdomains: List of subdomains for which to define boundary conditions.

        Returns:
            bc_values: Array of boundary condition values.

        """
        bc_values = []
        for sd in subdomains:
            # Get density and viscosity values on boundary faces applying trace to
            # interior values.
            _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
            # Append to list of boundary values
            vals = np.zeros(sd.num_faces)
            vals[north + south] = self.fluid.density() / self.fluid.viscosity()
            bc_values.append(vals)

        # Concatenate to single array and wrap as ad.Array
        bc_values = pp.wrap_as_ad_array(np.hstack(bc_values), name="bc_values_mobility")
        return bc_values

    def bc_values_mechanics_np(self, sd: pp.Grid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Extracted from below method to facilitate time dependent boundary conditions.
        Values for north and south faces are set to zero unless otherwise specified
        through attributes ux_north, uy_north, ux_south, uy_south.

        Parameters:
            sd: Subdomain for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        _, _, _, north, south, _, _ = self.domain_boundary_sides(sd)
        values = np.zeros((sd.dim, sd.num_faces))
        values[0, south] = self.params.get("ux_south", 0)
        values[1, south] = self.params.get("uy_south", 0)
        values[0, north] = self.params.get("ux_north", 0)
        values[1, north] = self.params.get("uy_north", 0)
        return values.ravel("F")

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the mechanics problem.

        Parameters:
            subdomains: List of subdomains for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # Set the boundary values
        bc_values = []
        if len(subdomains) == 0:
            return pp.ad.Array(np.zeros(0), name="bc_values_mechanics")
        for sd in subdomains:
            bc_values.append(self.bc_values_mechanics_np(sd))
        ad_values = pp.wrap_as_ad_array(
            np.hstack(bc_values), name="bc_values_mechanics"
        )
        return ad_values


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
    "biot_coefficient": 0.8,
    "permeability": 1e-20,
    "density": 2700,
    "porosity": 7e-3,
    "shear_modulus": 16.67 * pp.GIGA,
    "lame_lambda": 11.11 * pp.GIGA,
    "specific_heat_capacity": 790,
    "thermal_conductivity": 2.5,
    "thermal_expansion": 1e-5,
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
