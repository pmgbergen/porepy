"""Utility methods for setting up models for testing."""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np

import porepy as pp


class RectangularDomainThreeFractures(pp.ModelGeometry):
    """A rectangular domain with up to three fractures.

    The first two fractures are orthogonal, with `x` and `y` coordinates equal to
    0.5, respectively. The third fracture is tilted. The number of fractures is
    controlled by the parameter ``num_fracs``, which can be 0, 1, 2, or 3.

    """

    params: dict
    """Parameters for the model."""

    def set_fracture_network(self) -> None:
        # Length scale:
        ls = 1 / self.units.m

        num_fracs = self.params.get("num_fracs", 1)
        domain = pp.Domain({"xmin": 0, "xmax": 2 * ls, "ymin": 0, "ymax": 1 * ls})
        fractures = [
            pp.LineFracture(np.array([[0, 2], [0.5, 0.5]]) * ls),
            pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]) * ls),
            pp.LineFracture(np.array([[0.3, 0.7], [0.3, 0.7]]) * ls),
        ]
        self.fracture_network = pp.FractureNetwork2d(fractures[:num_fracs], domain)

    def mesh_arguments(self) -> dict:
        # Divide by length scale:
        h = 0.5 / self.units.m
        return {"mesh_size_frac": h, "mesh_size_bound": h}

    def set_md_grid(self) -> None:
        if not self.params.get("cartesian", False):
            return super().set_md_grid()

        # Not implemented for 3d. Assert for safety and mypy.
        assert isinstance(self.fracture_network, pp.FractureNetwork2d)

        # Length scale:
        ls = 1 / self.units.m
        # Mono-dimensional grid by default
        phys_dims = np.array([2, 1]) * ls
        n_cells = np.array([8, 2])
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self.domain = pp.Domain(box)
        # Translate fracture network to cart_grid format
        fracs = []
        for f in self.fracture_network._edges.T:
            fracs.append(self.fracture_network._pts[:, f])
        self.mdg = pp.fracs.meshing.cart_grid(fracs, n_cells, physdims=phys_dims)


class OrthogonalFractures3d(pp.ModelGeometry):
    """A 3d domain with up to three orthogonal fractures.

    The fractures have constant `x`, `y` and `z` coordinates equal to 0.5, respectively,
    and are situated in a unit cube domain. The number of fractures is controlled by
    the parameter ``num_fracs``, which can be 0, 1, 2 or 3.

    """

    params: dict
    """Model parameters."""

    def set_fracture_network(self) -> None:
        """Set the fracture network.

        The fractures are stored in self.fracture_network.

        """
        # Length scale:
        ls = 1 / self.units.m

        num_fracs = self.params.get("num_fracs", 1)
        domain: pp.Domain = pp.grids.standard_grids.utils.unit_domain(3)
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


class WellGeometryMixin:
    """Mixin class for models with wells."""

    nd: int
    """Number of dimensions."""
    params: dict
    """Model parameters."""

    def set_well_network(self) -> None:
        """Assign well network class."""
        num_wells = self.params.get("num_wells", 1)
        if self.nd == 2:
            # Comments are the intersection with fractures in
            # RectangularDomainThreeFractures
            wells = [
                pp.Well(np.array([0.5], [0.1], [0])),  # Intersects one fracture
                pp.Well(np.array([0.5], [0.5], [0])),  # Intersects two fractures
                pp.Well(np.array([0.25], [0.9], [0])),  # Intersects no fractures
            ]
            self.well_network = pp.WellNetwork2d(wells[:num_wells])
        else:
            wells = [
                # Intersects one (horizontal) fracture of OrthogonalFractures3d and
                # extends to the top of the domain
                pp.Well(np.array([[0.2, 0.2], [0.1, 0.1], [0.2, 1]])),
                # Intersects two (horizontal and vertical) fractures of
                # OrthogonalFractures3d. Extends between two domain boundaries.
                pp.Well(np.array([[0.0, 0.6], [0.5, 0.5], [0.4, 1]])),
                # Intersects no fractures. Internal well.
                pp.Well(np.array([[0.3, 0.3], [0.3, 0.3], [0.3, 0.4]])),
            ]
            self.well_network = pp.WellNetwork3d(wells[:num_wells])


class BoundaryConditionsMassAndEnergyDirNorthSouth(
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy
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
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

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
            domain_sides = self.domain_boundary_sides(sd)
            vals_loc = np.zeros(sd.num_faces)
            vals_loc[domain_sides.north + domain_sides.south] = self.fluid.pressure()
            vals.append(vals_loc)
        return pp.wrap_as_ad_array(np.hstack(vals), name="bc_values_darcy")

    def bc_type_mobrho(self, sd):
        """Boundary condition type for the density-mobility product.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary condition values for the mobility.

        Nonzero values are only defined on the north and south boundaries corresponding
        to the reference value of the density-mobility product.

        Parameters:
            subdomains: List of subdomains for which to define boundary conditions.

        Returns:
            bc_values: Array of boundary condition values.

        """
        values = []
        for sd in subdomains:
            # Get density and viscosity values on boundary faces applying trace to
            # interior values.
            domain_sides = self.domain_boundary_sides(sd)
            # Append to list of boundary values
            vals = np.zeros(sd.num_faces)
            vals[domain_sides.north + domain_sides.south] = (
                self.fluid.density() / self.fluid.viscosity()
            )
            values.append(vals)

        # Concatenate to single array and wrap as ad.Array
        bc_values = pp.wrap_as_ad_array(np.hstack(values), name="bc_values_mobility")
        return bc_values


class BoundaryConditionsMechanicsDirNorthSouth(
    pp.momentum_balance.BoundaryConditionsMomentumBalance
):
    """Boundary conditions for the mechanics with Dirichlet conditions on north and
    south boundaries.

    """

    params: dict[str, Any]
    """Model parameters."""
    solid: pp.SolidConstants
    """Solid parameters."""
    fluid: pp.FluidConstants
    """Fluid parameters."""

    def bc_type_mechanics(self, sd):
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, domain_sides.north + domain_sides.south, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc

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
        domain_sides = self.domain_boundary_sides(sd)
        values = np.zeros((sd.dim, sd.num_faces))
        values[1, domain_sides.north] = self.solid.convert_units(
            self.params.get("uy_north", 0), "m"
        )
        values[1, domain_sides.south] = self.solid.convert_units(
            self.params.get("uy_south", 0), "m"
        )
        values[0, domain_sides.north] = self.solid.convert_units(
            self.params.get("ux_north", 0), "m"
        )
        values[0, domain_sides.south] = self.solid.convert_units(
            self.params.get("ux_south", 0), "m"
        )
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


class TimeDependentMechanicalBCsDirNorthSouth:
    """Time dependent displacement boundary conditions.

    For use in (thermo)poremechanics.
    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return BoundaryConditionsMechanicsDirNorthSouth.bc_type_mechanics(self, sd)

    def time_dependent_bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        assert len(subdomains) == 1
        sd = subdomains[0]

        domain_sides = self.domain_boundary_sides(sd)
        values = np.zeros((self.nd, sd.num_faces))
        # Add fracture width on top if there is a fracture.
        if len(self.mdg.subdomains()) > 1:
            frac_val = self.solid.convert_units(0.042, "m")
        else:
            frac_val = 0
        values[1, domain_sides.north] = frac_val
        if self.time_manager.time > 1e-5:
            values[1, domain_sides.north] += self.solid.convert_units(
                self.params.get("uy_north", 0), "m"
            )
            values[1, domain_sides.south] += self.solid.convert_units(
                self.params.get("uy_south", 0), "m"
            )
        return values.ravel("F")


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
    RectangularDomainThreeFractures,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    ...


class MomentumBalance(
    RectangularDomainThreeFractures,
    pp.momentum_balance.MomentumBalance,
):
    """Combine components needed for momentum balance simulation."""

    pass


class MassAndEnergyBalance(
    RectangularDomainThreeFractures,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Combine components needed for force balance simulation."""

    pass


class Poromechanics(
    RectangularDomainThreeFractures,
    pp.poromechanics.Poromechanics,
):
    """Combine components needed for poromechanics simulation."""

    pass


class Thermoporomechanics(
    RectangularDomainThreeFractures,
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
        geometry = RectangularDomainThreeFractures
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


def compare_scaled_primary_variables(
    setup_0: pp.SolutionStrategy,
    setup_1: pp.SolutionStrategy,
    variable_names: list[str],
    variable_units: list[str],
    cell_wise: bool = True,
):
    """Compare the solution of two simulations.

    The two simulations are assumed to be identical, except for the scaling of the
    variables. The method compares the values of the variables in SI units.

    Parameters:
        setup_0: First simulation.
        setup_1: Second simulation.
        variable_names: Names of the variables to be compared.
        variable_units: Units of the variables to be compared.

    """
    for var_name, var_unit in zip(variable_names, variable_units):
        # Obtain scaled values.
        scaled_values_0 = setup_0.equation_system.get_variable_values([var_name])
        scaled_values_1 = setup_1.equation_system.get_variable_values([var_name])
        # Convert back to SI units.
        values_0 = setup_0.fluid.convert_units(scaled_values_0, var_unit, to_si=True)
        values_1 = setup_1.fluid.convert_units(scaled_values_1, var_unit, to_si=True)
        compare_values(values_0, values_1, cell_wise=cell_wise)


def compare_scaled_model_quantities(
    setup_0: pp.SolutionStrategy,
    setup_1: pp.SolutionStrategy,
    method_names: list[str],
    method_units: list[str],
    domain_dimensions: list[int | None],
    cell_wise: bool = True,
):
    """Compare the solution of two simulations.

    The two simulations are assumed to be identical, except for the scaling of the
    variables. The method compares the values of the variables in SI units.

    Parameters:
        setup_0: First simulation.
        setup_1: Second simulation.
        method_names: Names of the methods to be compared.
        method_units: Units of the methods to be compared.
        domain_dimensions: Dimensions of the domains to be tested. If None, the method
            will be tested for all dimensions.
        cell_wise: If True, the values are compared cell-wise. If False, the values are
            compared globally.

    """
    for method_name, method_unit, dim in zip(
        method_names, method_units, domain_dimensions
    ):
        values = []
        for setup in [setup_0, setup_1]:
            # Obtain scaled values.
            method = getattr(setup, method_name)
            domains = domains_from_method_name(setup.mdg, method, domain_dimension=dim)
            # Convert back to SI units.
            value = method(domains).evaluate(setup.equation_system)
            if isinstance(value, pp.ad.Ad_array):
                value = value.val
            values.append(setup.fluid.convert_units(value, method_unit, to_si=True))
        compare_values(values[0], values[1], cell_wise=cell_wise)


def compare_values(
    values_0: np.ndarray,
    values_1: np.ndarray,
    cell_wise: bool = True,
):
    """Compare two arrays of values.

    Parameters:
        values_0: First array of values.
        values_1: Second array of values.
        cell_wise: If True, compare cell-wise values. If False, compare sums of values.

    """
    # Compare values.
    if cell_wise:
        # Compare cell-wise values.
        assert np.allclose(values_0, values_1)
    else:
        # Compare sums instead of individual values, to avoid
        # errors due to different grids generated by gmsh (particularly for different
        # length scales).
        # Tolerance relative to the sum of the absolute values, not the differences.
        # Add a small absolute tolerance to avoid problems with zero values.
        rtol = 1e-5 * np.sum(np.abs(values_0))
        assert np.isclose(np.sum(values_0 - values_1), 0, atol=1e-10 + rtol)
