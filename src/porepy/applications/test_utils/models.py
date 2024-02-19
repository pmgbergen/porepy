"""Utility methods for setting up models for testing."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    OrthogonalFractures3d,
    RectangularDomainThreeFractures,
)


class NoPhysics(  # type: ignore[misc]
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


class MassBalance(  # type: ignore[misc]
    RectangularDomainThreeFractures,
    pp.fluid_mass_balance.SinglePhaseFlow,
): ...


class MomentumBalance(  # type: ignore[misc]
    RectangularDomainThreeFractures,
    pp.momentum_balance.MomentumBalance,
):
    """Combine components needed for momentum balance simulation."""


class MassAndEnergyBalance(  # type: ignore[misc]
    RectangularDomainThreeFractures,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Combine components needed for force balance simulation."""


class Poromechanics(  # type: ignore[misc]
    RectangularDomainThreeFractures,
    pp.poromechanics.Poromechanics,
):
    """Combine components needed for poromechanics simulation."""


class Thermoporomechanics(  # type: ignore[misc]
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
    params = {"times_to_export": [], "fracture_indices": fracture_indices}

    # To define the model we comine a geometry with a physics class.
    # First identify the two component classes from the user input, and then combine
    # them in a new class.

    # Identify the geometry class
    geometry: Any = None
    if dim == 2:
        geometry = RectangularDomainThreeFractures
    elif dim == 3:
        geometry = OrthogonalFractures3d
    else:
        raise ValueError(f"Unknown dimension {dim}")

    # Identify the physics class
    model_class: Any = None
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
    domain_dimension: int | None,
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
        domains = mdg.interfaces(dim=domain_dimension)  # type: ignore[assignment]

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


# TODO: Purge in favour of pp.solid_values.granite
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
            value = method(domains).value(setup.equation_system)
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
