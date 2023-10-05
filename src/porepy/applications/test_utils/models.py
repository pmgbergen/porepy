"""Utility methods for setting up models for testing."""
from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures


class RectangularDomainThreeFractures(pp.ModelGeometry):
    """A rectangular domain with up to three fractures.

    The first two fractures are orthogonal, with `x` and `y` coordinates equal to
    0.5, respectively. The third fracture is tilted. The number of fractures is
    controlled by the parameter ``fracture_indices``, which can be any subset of
    [0, 1, 2].

    """

    params: dict
    """Parameters for the model."""

    def set_fractures(self) -> None:
        # Length scale:
        ls = self.solid.convert_units(1, "m")

        fracture_indices = self.params.get("fracture_indices", [0])
        fractures = [
            pp.LineFracture(np.array([[0, 2], [0.5, 0.5]]) * ls),
            pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]) * ls),
            pp.LineFracture(np.array([[0.3, 0.7], [0.3, 0.7]]) * ls),
        ]
        self._fractures = [fractures[i] for i in fracture_indices]

    def meshing_arguments(self) -> dict:
        # Divide by length scale:
        ls = self.solid.convert_units(1, "m")

        mesh_sizes = {
            # Cartesian: 2 by 8 cells.
            "cell_size_x": 0.25 * ls,
            "cell_size_y": 0.5 * ls,
            # Simplex. Whatever gmsh decides.
            "cell_size_fracture": 0.5 * ls,
            "cell_size_boundary": 0.5 * ls,
            "cell_size_min": 0.2 * ls,
        }
        return mesh_sizes

    def set_domain(self) -> None:
        if not self.params.get("cartesian", False):
            self.params["grid_type"] = "simplex"
        else:
            self.params["grid_type"] = "cartesian"

        # Length scale:
        ls = self.solid.convert_units(1, "m")

        # Mono-dimensional grid by default
        phys_dims = np.array([2, 1]) * ls
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self._domain = pp.Domain(box)


class OrthogonalFractures3d(CubeDomainOrthogonalFractures):
    """A 3d domain with up to three orthogonal fractures.

    The fractures have constant `x`, `y` and `z` coordinates equal to 0.5, respectively,
    and are situated in a unit cube domain. The number of fractures is controlled by
    the parameter ``num_fracs``, which can be 0, 1, 2 or 3.

    """

    params: dict
    """Model parameters."""

    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")

        mesh_sizes = {
            "cell_size": 0.5 * ls,
            "cell_size_fracture": 0.5 * ls,
            "cell_size_boundary": 0.5 * ls,
            "cell_size_min": 0.2 * ls,
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
        parameters = {"mesh_size": 0.2}
        self.well_network = pp.WellNetwork3d(
            self.domain, wells[:num_wells], parameters=parameters
        )


class BoundaryConditionsMassDirNorthSouth:
    """Boundary conditions for the flow problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
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

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value of 0 unless fluid's reference pressure is changed.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        vals_loc = np.zeros(boundary_grid.num_cells)
        vals_loc[domain_sides.north + domain_sides.south] = self.fluid.pressure()
        return vals_loc

    def bc_type_fluid_flux(self, sd):
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


class BoundaryConditionsEnergyDirNorthSouth(pp.BoundaryConditionMixin):
    """Boundary conditions for the thermal problem.

    Dirichlet boundary conditions are defined on the north and south boundaries. Some
    of the default values may be changed directly through attributes of the class.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
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

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
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


class BoundaryConditionsMechanicsDirNorthSouth(pp.BoundaryConditionMixin):
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

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Extracted from below method to facilitate time dependent boundary conditions.
        Values for north and south faces are set to zero unless otherwise specified
        through attributes ux_north, uy_north, ux_south, uy_south.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
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


class TimeDependentMechanicalBCsDirNorthSouth(BoundaryConditionsMechanicsDirNorthSouth):
    """Time dependent displacement boundary conditions.

    For use in (thermo)poremechanics.
    """

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        domain_sides = self.domain_boundary_sides(boundary_grid)
        values = np.zeros((self.nd, boundary_grid.num_cells))
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


class NoPhysics(
    pp.ModelGeometry, pp.SolutionStrategy, pp.DataSavingMixin, pp.BoundaryConditionMixin
):
    """A model with no physics, for testing purposes.

    The model comes with minimal physical properties, making testing of individual
    components (e.g. constitutive laws) easier.

    """

    def create_variables(self):
        pass

    def set_equations(self):
        pass

    def update_all_boundary_conditions(self):
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


class MassAndEnergyBalance(
    RectangularDomainThreeFractures,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Combine components needed for force balance simulation."""


class Poromechanics(
    RectangularDomainThreeFractures,
    pp.poromechanics.Poromechanics,
):
    """Combine components needed for poromechanics simulation."""


class Thermoporomechanics(
    RectangularDomainThreeFractures,
    pp.thermoporomechanics.Thermoporomechanics,
):
    """Combine components needed for poromechanics simulation."""


def model(
    model_type: str, dim: int, num_fracs: int = 1
) -> MassBalance | MomentumBalance | MassAndEnergyBalance | Poromechanics:
    """Setup for tests."""
    # Suppress output for tests
    fracture_indices = [i for i in range(num_fracs)]
    params = {"suppress_export": True, "fracture_indices": fracture_indices}

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


def subdomains_or_interfaces_from_method_name(
    mdg: pp.MixedDimensionalGrid,
    method_name: Callable,
    domain_dimension: int,
) -> list[pp.Grid] | list[pp.MortarGrid]:
    """Return the domains to be tested for a given method.

    The method to be tested is assumed to take as input only its domain of definition,
    that is a list of subdomains or interfaces. The test framework is not compatible with
    methods that take other arguments (and such a method would also break the implicit
    contract of the constitutive laws). Note that for the ambiguity related to methods
    defined on either subdomains or boundaries, only subdomains are considered.

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
    if "subdomains" in signature.parameters or "domains" in signature.parameters:
        # If relevant, filter out the domains that are not to be tested.
        domains = mdg.subdomains(dim=domain_dimension)
    elif "interfaces" in signature.parameters:
        domains = mdg.interfaces(dim=domain_dimension)

    return domains


def _add_mixin(mixin, parent):
    """Helper method to dynamically construct a class by adding a mixin.

    Multiple mixins can be added by nested calls to this method.

    Reference:
        https://www.geeksforgeeks.org/create-classes-dynamically-in-python/

    """
    parent_name = parent.__name__
    mixin_name = mixin.__name__
    name = f"Combined_{mixin_name}_{parent_name}"
    # IMPLEMENTATION NOTE: The last curly bracket can be used to add code to the created
    # class (empty brackets is equivalent to a ``pass``). In principle, we could add
    # this as an extra parameter to this function, but at the moment it is unclear why
    # such an addition could not be made in the mixin class instead.
    cls = type(name, (mixin, parent), {})
    return cls


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
    "fracture_normal_stiffness": 1529,
    "maximum_fracture_closure": 1e-4,
    "fracture_gap": 1e-4,
    "residual_aperture": 0.01,
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
        scaled_values_0 = setup_0.equation_system.get_variable_values(
            variables=[var_name], time_step_index=0
        )
        scaled_values_1 = setup_1.equation_system.get_variable_values(
            variables=[var_name], time_step_index=0
        )
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
            domains = subdomains_or_interfaces_from_method_name(
                setup.mdg, method, domain_dimension=dim
            )
            # Convert back to SI units.
            value = method(domains).evaluate(setup.equation_system)
            if isinstance(value, pp.ad.AdArray):
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


def get_model_methods_returning_ad_operator(model_setup) -> list[str]:
    """Get all possible testable methods to be used in the test_ad_methods_xxx.py files.

    A testable method is one that:

        (1) Has a single input parameter,
        (2) The name of the parameter is either 'subdomains' or 'interfaces', and
        (3) Returns either a 'pp.ad.Operator' or a 'pp.ad.DenseArray'.

    Parameters:
        model_setup: Model setup after `prepare_simulation()` has been called.

    Returns:
        List of all possible testable method names for the given model.

    """

    # Get all public methods
    all_methods = [method for method in dir(model_setup) if not method.startswith("_")]

    # Get all testable methods
    testable_methods: list[str] = []
    for method in all_methods:
        # Get method in callable form
        callable_method = getattr(model_setup, method)

        # Retrieve method signature via inspect
        try:
            signature = inspect.signature(callable_method)
        except TypeError:
            continue

        # Append method to the `testable_methods` list if the conditions are met
        if (
            len(signature.parameters) == 1
            and (
                "subdomains" in signature.parameters
                or "interfaces" in signature.parameters
                or "domains" in signature.parameters
            )
            and (
                "pp.ad.Operator" in signature.return_annotation
                or "pp.ad.DenseArray" in signature.return_annotation
            )
        ):
            testable_methods.append(method)

    return testable_methods
