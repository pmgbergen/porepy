"""Provides tests for AD methods used in the single phase flow model.

We provide two tests:

    (1) `test_tested_vs_testable_methods_single_phase_flow`: This test checks that
      all the testable methods (see docstring of
      `models.get_model_methods_returning_ad_operator()` to see what constitutes
      a testable method) of the single-phase flow model are included in the
      parametrization of `test_ad_operator_methods_single_phase_flow`, and

    (2) `test_ad_operator_methods_single_phase_flow`: A parametrized test where each
      tested method is evaluated and their output is compared to expected values.

"""

from __future__ import annotations

import copy
from typing import Literal, Optional

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils import models, well_models
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.applications.material_values.solid_values import (
    extended_granite_values_for_testing as granite_values,
)
from porepy.applications.material_values.fluid_values import (
    extended_water_values_for_testing as water_values,
)

@pytest.fixture(scope="function")
def model_setup():
    """Minimal compressible single-phase flow setup with two intersecting fractures.

    The model is set up with realistic physical parameters using water and granite.
    A 2x2 Cartesian grid on a unit domain is employed. One horizontal and one vertical
    fractures are included in the domain. This results in the following `mdg`:

        Maximum dimension present: 2
        Minimum dimension present: 0
        1 grids of dimension 2 with in total 4 cells
        2 grids of dimension 1 with in total 4 cells
        1 grids of dimension 0 with in total 1 cells
        2 interfaces between grids of dimension 2 and 1 with in total 8 mortar cells.
        2 interfaces between grids of dimension 1 and 0 with in total 4 mortar cells.

    Non-zero (but constant) states for the primary variables (pressure and interface
    fluxes) are assigned to the `iterate` states. Note that the model is not
    effectively run, but it is discretized via `prepare_simulation()`.

    Returns:
        Prepared-for-simulation model with non-trivial primary variables.

    """

    class Model(SquareDomainOrthogonalFractures, SinglePhaseFlow):
        """Single phase flow model in a domain with two intersecting fractures."""

    # Material constants
    solid = pp.SolidConstants(**granite_values)
    fluid = pp.FluidComponent(**water_values)
    # Declare model parameters
    params = {
        "material_constants": {"solid": solid, "fluid": fluid},
        "fracture_indices": [0, 1],
        "grid_type": "cartesian",
        "meshing_arguments": {"cell_size_x": 0.5, "cell_size_y": 0.5},
    }

    # Instantiate the model setup
    setup = Model(params)

    # Prepare to simulate
    setup.prepare_simulation()

    # Set constant but non-zero values for the primary variables
    setup.equation_system.set_variable_values(
        200 * pp.BAR * np.ones(setup.mdg.num_subdomain_cells()),
        [setup.pressure_variable],
        iterate_index=0,
    )
    setup.equation_system.set_variable_values(
        1e-12 * np.ones(setup.mdg.num_interface_cells()),
        [setup.interface_darcy_flux_variable],
        iterate_index=0,
    )

    return setup


@pytest.fixture(scope="function")
def all_tested_methods(request) -> list[str]:
    """Get all tested methods.

    Parameters:
        request: PyTest request object.

    Returns:
        List of all tested methods in `test_ad_operator_methods_single_phase_flow`.

    """
    # Declare name of the test.
    test_name = "test_ad_operator_methods_single_phase_flow"

    # Retrieve all tests that are part of the parametrization.
    tests = [
        item.name for item in request.session.items if item.name.startswith(test_name)
    ]

    # Filter the name of the first parameter from the pytest.mark.parametrize
    tested_methods = [test[test.find("[") + 1 : test.find("-")] for test in tests]

    return tested_methods


@pytest.fixture(scope="function")
def all_testable_methods(model_setup) -> list[str]:
    """Get all testable methods.

    Parameters:
        model_setup: Single-phase flow model setup after `prepare_simulation()`
            has been called.

    Returns:
        List of all possible testable methods for the model.

    """
    return models.get_model_methods_returning_ad_operator(model_setup)


def test_tested_vs_testable_methods_single_phase_flow(
    all_tested_methods: list[str],
    all_testable_methods: list[str],
) -> None:
    """Test whether all the tested methods coincide with all the testable methods.

    Note:
        This test will fail if it is run in isolation since it relies on the
        collection of the tests from `test_ad_operator_methods_single_phase_flow`.
        As long as such tests are part of the pytest session, this should work fine.

    Parameters:
        all_tested_methods: List of all tested methods.
        all_testable_methods: List of all testable methods.

    """
    # Failure here could mean two things:
    #
    #   (1) The `all_tested_methods` fixture is an empty list due to the tests
    #       from `test_ad_operator_methods_single_phase_flow` not being collected (see
    #       the Note in the docstring), or
    #
    #   (2) There is at least one method that was added/renamed in the core, but it was
    #       not included/updated in the parametrization of
    #       `test_ad_operator_methods_single_phase_flow`. To fix the test, please add
    #       the missing method with the expected value as part of the parametrization of
    #       `test_ad_operator_methods_single_phase_flow`. If you don't know which method
    #       is missing, add a break point before the "assert" and compare the list of
    #       `testable_methods` with the list of `tested_methods`. Remember to debug
    #       while running all the tests from the module so that you don't incur in (1).
    #
    assert all_tested_methods == all_testable_methods


# NOTE: The tests for darcy_flux, fluid_flux, fluid_flux, fluid_source,
# interface_darcy_flux_equation, interface_fluid_flux, interface_flux_equation,
# mass_balance_equation, normal_permeability, skin_factor, and pressure_trace were based
# on different water fluid values in PorePy 1.10.0 and prior. The new expected values
# for each test are copied from the test's output. We rely on the correctness of the
# previous tests for this choice.
@pytest.mark.parametrize(
    "method_name, expected_value, dimension_restriction",
    [
        # Combination of mobility and fluid density = rho/mu
        # = rho_ref * exp(c_f * (p - p_ref)) / mu = 1000 * exp(4e-10 * 2e7) /  0.001
        (
            "advection_weight_mass_balance",
            water_values["density"]
            * np.exp(water_values["compressibility"] * 200 * pp.BAR)
            / water_values["viscosity"],
            None,
        ),
        ("aperture", np.array([1, 1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]), None),
        (
            'boundary_fluid_flux',
            np.array(
                [
                    996207.58483034,
                    0.,
                    996207.58483034,
                    996207.58483034,
                    0.,
                    996207.58483034,
                    996207.58483034,
                    996207.58483034,
                    0.,
                    0.,
                    996207.58483034,
                    996207.58483034,
                    0.,
                    0.,
                    0.,
                    0.,
                    996207.58483034,
                    0.,
                    996207.58483034,
                    0.,
                    996207.58483034,
                    0.,
                    996207.58483034,
                    0.
                ]
            ),
            None
        ),
        ("combine_boundary_operators_darcy_flux", np.zeros(24), None),
        # Darcy flux.
        (
            "darcy_flux",
            np.array(
                [
                    -2.0e-10,
                    -1.0e-12,
                    2.0e-10,
                    -2.0e-10,
                    -1.0e-12,
                    2.0e-10,
                    -2.0e-10,
                    -2.0e-10,
                    -1.0e-12,
                    -1.0e-12,
                    2.0e-10,
                    2.0e-10,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    -4.0e-13,
                    -1.0e-12,
                    4.0e-13,
                    1.0e-12,
                    -4.0e-13,
                    -1.0e-12,
                    4.0e-13,
                    1.0e-12,
                ]
            ),
            None,
        ),
        (
            "equivalent_well_radius",
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]),
            None,
        ),
        ("fluid_compressibility", water_values["compressibility"], None),
        # Values of the mass fluxes (i.e., scaling with rho/mu is incorporated).
        (
            "fluid_flux",
            np.array(
                [
                    -1.99241517e-04,
                    -1.00533254e-06,
                    2.01066509e-04,
                    -1.99241517e-04,
                    -1.00533254e-06,
                    2.01066509e-04,
                    -1.99241517e-04,
                    -1.99241517e-04,
                    -1.00533254e-06,
                    -1.00533254e-06,
                    2.01066509e-04,
                    2.01066509e-04,
                    1.00533254e-06,
                    1.00533254e-06,
                    1.00533254e-06,
                    1.00533254e-06,
                    -3.98483034e-07,
                    -1.00533254e-06,
                    4.02133017e-07,
                    1.00533254e-06,
                    -3.98483034e-07,
                    -1.00533254e-06,
                    4.02133017e-07,
                    1.00533254e-06,
                ]
            ),
            None,
        ),
        # fluid_mass = rho * phi * specific_volume
        (
            "fluid_mass",
            np.array(
                [
                    3.27386543e00,
                    3.27386543e00,
                    3.27386543e00,
                    3.27386543e00,
                    6.54773085e-03,
                    6.54773085e-03,
                    6.54773085e-03,
                    6.54773085e-03,
                    1.30954617e-05,
                ]
            ),
            None,
        ),
        # Only sources from interface fluxes are non-zero. External sources are zero.
        (
            "fluid_source",
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.01066509e-06,
                        2.01066509e-06,
                        2.01066509e-06,
                        2.01066509e-06,
                        4.02133017e-06,
                    ]
                ]
            ),
            None,
        ),
        ("interface_darcy_flux_equation", 5.00001e-07, None),
        ("interface_fluid_flux", 1.00533254e-06, None),
        ("interface_flux_equation", 5.00001e-07, None),
        ("interface_vector_source_darcy_flux", 0, None),
        (
            "mass_balance_equation",
            np.array(
                [
                    3.01159202e-02,
                    3.01177451e-02,
                    3.01177451e-02,
                    3.01195701e-02,
                    5.88240034e-05,
                    5.88276534e-05,
                    5.88240034e-05,
                    5.88276534e-05,
                    -3.90246847e-06,
                ]
            ),
            None,
        ),
        # # Mobility = 1 / mu
        # ("mobility", 1.0 / (water_values["viscosity"]), None),
        # # Combination of mobility and fluid density = rho / mu
        # # Mobility_rho = rho_ref * exp(c_f * (p - p_ref)) / mu
        # (
        #     "mobility_rho",
        #     water_values["density"]
        #     * np.exp(water_values["compressibility"] * 200 * pp.BAR)
        #     / water_values["viscosity"],
        #     None,
        # ),
        ("normal_permeability", granite_values["normal_permeability"], None),
        ("permeability", granite_values["permeability"], None),
        ("porosity", granite_values["porosity"], None),
        ("pressure", 200 * pp.BAR, None),
        # pressure_exponential = exp(c_f * (p - p_ref))
        (
            "pressure_exponential",
            np.exp(water_values["compressibility"] * 200 * pp.BAR),
            None,
        ),
        (
            "pressure_trace",
            np.array(
                [
                    0.00e00,
                    19900000,
                    0.00e00,
                    0.00e00,
                    19900000,
                    0.00e00,
                    0.00e00,
                    0.00e00,
                    19900000,
                    19900000,
                    0.00e00,
                    0.00e00,
                    19900000,
                    19900000,
                    19900000,
                    19900000,
                    0.00e00,
                    -29999999.99999999,
                    0.00e00,
                    -29999999.99999999,
                    0.00e00,
                    -29999999.99999999,
                    0.00e00,
                    -29999999.99999999,
                ]
            ),
            None,
        ),
        # ("reference_pressure", 0, None),
        ("skin_factor", granite_values["skin_factor"], None),
        ("tangential_component", np.array([[1.0, 0.0]]), 0),  # Check only for 0d.
        # Same as advection weight in 1-phase, 1-component cast
        (
            "total_mobility",
            water_values["density"]
            * np.exp(water_values["compressibility"] * 200 * pp.BAR)
            / water_values["viscosity"],
            None,
        ),
        ("well_fluid_flux", 0, 2),  # Use dim_restriction=2 to ignore well flux.
        ("well_flux_equation", 0, 2),  # Use dim_restriction=2 to ignore well equation.
        ("well_radius", granite_values["well_radius"], None),
        # Testing of methods with deeper namespace.
        # rho = rho_ref * exp(c_f * (p - p_ref))
        (
            "fluid.density",
            water_values["density"]
            * np.exp(water_values["compressibility"] * 200 * pp.BAR),
            None,
        ),
        # specific_volume = 1 / (rho_ref * exp(c_f * (p - p_ref)))
        (
            "fluid.specific_volume",
            1.0
            / (
                water_values["density"]
                * np.exp(water_values["compressibility"] * 200 * pp.BAR)
            ),
            None,
        ),
        (
            "fluid.reference_phase.density",
            water_values["density"]
            * np.exp(water_values["compressibility"] * 200 * pp.BAR),
            None,
        ),
        (
            "fluid.reference_phase.specific_volume",
            1.0
            / (
                water_values["density"]
                * np.exp(water_values["compressibility"] * 200 * pp.BAR)
            ),
            None,
        ),
        ("fluid.reference_phase.viscosity", water_values["viscosity"], None),
    ],
)
def test_ad_operator_methods_single_phase_flow(
    model_setup,
    method_name: str,
    expected_value: float | np.ndarray,
    dimension_restriction: int | None,
) -> None:
    """Test that Ad operator methods return expected values.

    Parameters:
        model_setup: Prepared-for-simulation single phase flow model setup.
        method_name: Name of the method to be tested.
        expected_value: The expected value from the evaluation.
        dimension_restriction: Dimension in which the method is restricted. If None,
            all valid dimensions for the method will be considered. Can also be used
            to test a method that is valid in all dimensions, but for the sake of
            compactness, only tested in one dimension.

    """
    # Processing name space.
    method_namespace = method_name.split(".")
    owner = model_setup
    # loop top to bottom through namespace to get to the actual method defined on some
    # grids and returning an operator.
    for name in method_namespace:
        method = getattr(owner, name)
        owner = method

    # Obtain list of subdomain or interface grids where the method is defined.
    domains = models.subdomains_or_interfaces_from_method_name(
        model_setup.mdg,
        method,
        dimension_restriction,
    )

    # Obtain operator in AD form.
    operator = method(domains)

    # Discretize (if necessary), evaluate, and retrieve numerical values.
    operator.discretize(model_setup.mdg)
    val = operator.value(model_setup.equation_system)

    if isinstance(val, sps.bsr_matrix):  # needed for `tangential_component`
        val = val.toarray()

    # Compare the actual and expected values.
    assert np.allclose(val, expected_value, rtol=1e-8, atol=1e-15)

@pytest.mark.parametrize(
    "method_name, p_or_c, expected_value",
    [
        ('phase_mobility', 'phase', 1 / water_values['viscosity']),
        ('fractional_phase_mobility', 'phase', 1),
        (
            'component_mobility',
            'component',
            water_values["density"]
            * np.exp(water_values["compressibility"] * 200 * pp.BAR)
            / water_values["viscosity"],
        ),
        ('fractional_component_mobility', 'component', 1),
    ]
)
def test_mobility_single_phase_flow(
    model_setup: pp.PorePyModel,
    method_name: str,
    p_or_c: Literal['phase', 'component'],
    expected_value: float,
) -> None:
    """Tests the evaluation of various mobility methods in the single-phase,
    single-component model.

    Fractional mobilities must be 1.
    The phase and component mobility must be equal to the total mobility.

    """
    # mobilities are only defined on subdomains
    domains = model_setup.mdg.subdomains()

    assert model_setup.fluid.num_components == 1
    assert model_setup.fluid.num_phases == 1

    if p_or_c == 'phase':
        instance = model_setup.fluid.reference_phase
    elif p_or_c == 'component':
        instance = model_setup.fluid.reference_component
    else:
        assert False, 'Unclear test input'

    # Fetching method and calling it with the right instance
    op: pp.ad.Operator = getattr(model_setup, method_name)(instance, domains)
    val = op.value(model_setup.equation_system)
    # Compare the actual and expected values.
    assert np.allclose(val, expected_value, rtol=1e-8, atol=1e-15)

@pytest.mark.parametrize(
    "units",
    [
        {"m": 2, "kg": 3, "s": 1, "K": 1},
    ],
)
def test_unit_conversion(units):
    """Test that solution is independent of units.

    Parameters:
        units (dict): Dictionary with keys as those in
            :class:`~pp.compositional.materials.Constants`.

    """

    class Model(SquareDomainOrthogonalFractures, SinglePhaseFlow):
        """Single phase flow model in a domain with two intersecting fractures."""

        def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """Ensure nontrivial solution."""
            vals = self.reference_variable_values.pressure * np.ones(
                boundary_grid.num_cells
            )
            faces = self.domain_boundary_sides(boundary_grid).east
            vals[faces] += self.units.convert_units(1e5, "Pa")
            return vals

    solid_vals = pp.solid_values.extended_granite_values_for_testing
    fluid_vals = water_values
    numerical_vals = pp.numerical_values.extended_numerical_values_for_testing
    ref_vals = pp.reference_values.extended_reference_values_for_testing
    solid = pp.SolidConstants(**solid_vals)
    fluid = pp.FluidComponent(**fluid_vals)
    numerical = pp.NumericalConstants(**numerical_vals)
    reference_values = pp.ReferenceVariableValues(**ref_vals)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "fracture_indices": [0, 1],
        "cartesian": True,
        "material_constants": {"solid": solid, "fluid": fluid, "numerical": numerical},
        "reference_variable_values": reference_values,
    }
    solver_params = {
        "nl_convergence_tol_res": 1e-12,
        "nl_convergence_tol": 1,
    }
    reference_params = copy.deepcopy(params)
    reference_params["file_name"] = "unit_conversion_reference"
    reference_solver_params = copy.deepcopy(solver_params)

    # Create model and run simulation
    setup_0 = Model(reference_params)
    pp.run_time_dependent_model(setup_0, reference_solver_params)

    params["units"] = pp.Units(**units)
    setup_1 = Model(params)

    pp.run_time_dependent_model(setup_1, solver_params)
    variables = [setup_1.pressure_variable, setup_1.interface_darcy_flux_variable]
    variable_units = ["Pa", "Pa * m^2 * s^-1"]
    models.compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    flux_names = ["darcy_flux", "fluid_flux"]
    flux_units = ["Pa * m^2 * s^-1", "kg * m^-1 * s^-1"]
    # No domain restrictions.
    domain_dimensions = [None, None]
    models.compare_scaled_model_quantities(
        setup_0, setup_1, flux_names, flux_units, domain_dimensions
    )


class WellModel(
    well_models.OneVerticalWell,
    models.OrthogonalFractures3d,
    well_models.BoundaryConditionsWellSetup,
    well_models.WellPermeability,
    pp.SinglePhaseFlow,
):
    pass


def test_well_incompressible_pressure_values():
    """One central vertical well, one horizontal fracture.

    The well is placed such that the pressure at the fracture is log distributed, up to
    the effect of the low-permeable matrix.

    """
    params = {
        # Set impermeable matrix
        "material_constants": {
            "solid": pp.SolidConstants(permeability=1e-6 / 4, well_radius=0.01)
        },
        # Use only the horizontal fracture of OrthogonalFractures3d
        "fracture_indices": [2],
    }

    setup = WellModel(params)
    pp.run_time_dependent_model(setup)
    # Check that the matrix pressure is close to linear in z
    matrix = setup.mdg.subdomains(dim=3)[0]
    matrix_pressure = setup.pressure([matrix]).value(setup.equation_system)
    dist = np.absolute(matrix.cell_centers[2, :] - 0.5)
    p_range = np.max(matrix_pressure) - np.min(matrix_pressure)
    expected_p = p_range * (0.5 - dist) / 0.5
    diff = expected_p - matrix_pressure
    # This is a very loose tolerance, but the pressure is not exactly linear due to the
    # the fracture pressure distribution and grid effects.
    assert np.max(np.abs(diff / p_range)) < 1e-1
    # With a matrix permeability of 1/4e-6, the pressure drop for a unit flow rate is
    # (1/2**2) / 1/4e-6 = 1e6: 1/2 for splitting flow in two directions (top and
    # bottom), and 1/2 for the distance from the fracture to the boundary.
    assert np.isclose(np.max(matrix_pressure), 1e6, rtol=1e-1)
    # In the fracture, check that the pressure is log distributed
    fracs = setup.mdg.subdomains(dim=2)
    fracture_pressure = setup.pressure(fracs).value(setup.equation_system)
    sd = fracs[0]
    injection_cell = sd.closest_cell(np.atleast_2d([0.5, 0.5, 0.5]).T)
    # Check that the injection cell is the one with the highest pressure
    assert np.argmax(fracture_pressure) == injection_cell
    pt = sd.cell_centers[:, injection_cell]
    dist = pp.distances.point_pointset(pt, sd.cell_centers)
    min_p = 1e6
    scale_dist = np.exp(-np.max(dist))
    perm = 1e-3
    expected_p = min_p + 1 / (4 * np.pi * perm) * np.log(dist / scale_dist)
    assert np.isclose(np.min(fracture_pressure), min_p, rtol=1e-2)
    wells = setup.mdg.subdomains(dim=0)
    well_pressure = setup.pressure(wells).value(setup.equation_system)

    # Check that the pressure drop from the well to the fracture is as expected The
    # Peacmann well model is: u = 2 * pi * k * h * (p_fracture - p_well) / ( ln(r_e /
    # r_well) + S) We assume r_e = 0.20 * meas(injection_cell). h is the aperture, and
    # the well radius is 0.01 and skin factor 0. With a permeability of 0.1^2/12
    # and unit flow rate, the pressure drop is p_fracture - p_well = 1 / (2 * pi * 1e-3
    # * 0.1) * ln(0.2 / 0.1)
    k = 1e-2 / 12
    r_e = np.sqrt(sd.cell_volumes[injection_cell]) * 0.2
    dp_expected = 1 / (2 * np.pi * k * 0.1) * np.log(r_e / 0.01)
    dp = well_pressure[0] - fracture_pressure[injection_cell]
    assert np.isclose(dp, dp_expected, rtol=1e-4)


# Definition of parameter space for gravity tests
# For most tests, we want to combine the following parameters:
#     - discretization method
#     - number of nodes in the mortar grid
#     - number of nodes in the 1d fracture grid
#     - grid type (cartesian or simplex)
# Exceptions (if statements below):
#     - Simplex grids only give accurate results for mpfa discretization.
#     - The construction of non-matching grids does not support simplex grids.
# Some tests use tailored parameter combinations defined in the test function.
discretizations = ["mpfa", "tpfa"]

grid_types = ["cartesian", "simplex"]

mortar_nodes = [3, 4]

fracture_nodes = [2, 3, 4]

gravity_parameter_combinations = [
    (discretization_method, grid_type, num_nodes_mortar, num_nodes_1d)
    for discretization_method in discretizations
    for grid_type in grid_types
    for num_nodes_mortar in mortar_nodes
    for num_nodes_1d in fracture_nodes
    if not (
        grid_type == "simplex"
        and (discretization_method == "tpfa" or num_nodes_mortar != num_nodes_1d)
    )
]  # Skip some of these in quick tests, @EK?


def model_setup_gravity(
    dimension: int,
    model_params: dict,
    num_nodes_mortar: Optional[int] = None,
    num_nodes_1d: int = 3,
    neu_val_top: Optional[float] = None,
    dir_val_top: Optional[float] = None,
    discretization_method: Literal["mpfa", "tpfa"] = "mpfa",
    kn: float = 1e0,
    aperture: float = 1e-1,
    gravity_angle: float = 0,
) -> SinglePhaseFlow:
    """Create a single-phase flow model with gravity.

    Parameters:
        dimension: Dimension of the domain.
        model_params: Model parameters as passed to the model constructor.
        num_nodes_mortar: Number of nodes in the mortar grid. If None, no changes are
            made to the grid (default behavior with matching grids).
        num_nodes_1d: Number of nodes in the 1d fracture grid. Only used if
            num_nodes_mortar is not None.
        neu_val_top: Default None implies Dirichlet on top. If not None,
            the prescribed value will be applied as the Neumann bc value.
        dir_val_top: If not None, the prescribed value will be
            applied as the Dirichlet bc value. Note that neu_val_top takes precedent
            over dir_val_top.
        discretization_method: Name of discretization method.
        kn: Normal permeability of the fracture. Will be multiplied by aperture/2
            to yield the normal diffusivity.
        aperture: Fracture aperture.
        gravity_angle: Angle by which to rotate the applied vector source field.

    Returns:
        Single-phase flow model with gravity.

    """
    params = {
        "grid_type": "cartesian",
        "fracture_indices": [-1],  # Constant y and z coordinates in 2d and 3d, resp.
        "meshing_arguments": {"cell_size": 0.5},
        "darcy_flux_discretization": discretization_method,
    }
    params.update(model_params)
    params["material_constants"] = {
        "solid": pp.SolidConstants(normal_permeability=kn, residual_aperture=aperture)
    }
    if dimension == 2:
        Geometry = SquareDomainOrthogonalFractures
    else:
        Geometry = CubeDomainOrthogonalFractures

    class Model(
        Geometry, pp.constitutive_laws.GravityForce, SinglePhaseFlow, FluxDiscretization
    ):
        def set_geometry(self) -> None:
            super().set_geometry()
            if num_nodes_mortar is None:
                return

            if dimension != 2:
                raise NotImplementedError(
                    "Non-matching grids are only implemented for 2d."
                )

            for intf in self.mdg.interfaces():
                new_side_grids = {
                    s: pp.refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
                    for s, g in intf.side_grids.items()
                }

                intf.update_mortar(new_side_grids, tol=1e-4)
                # refine the 1d-physical grid
                _, old_sd_1d = self.mdg.interface_to_subdomain_pair(intf)
                new_sd_1d = pp.refinement.remesh_1d(old_sd_1d, num_nodes=num_nodes_1d)
                new_sd_1d.compute_geometry()

                self.mdg.replace_subdomains_and_interfaces({old_sd_1d: new_sd_1d})
                intf.update_secondary(new_sd_1d, tol=1e-4)
            self.mdg.compute_geometry()

        def gravity_force(
            self, grids: list[pp.Grid] | list[pp.MortarGrid], material: str
        ) -> pp.ad.Operator:
            """Vector source term. Represents gravity effects.

            Parameters:
                grids: List of subdomain or interface grids where the vector source is
                    defined.
                material: Name of the material. Could be either "fluid" or "solid".

            Returns:
                Cell-wise nd-vector source term operator.

            """
            if np.isclose(gravity_angle, 0):
                # Normalize by the GravityForce class' default value.
                default = (
                    self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
                    * self.fluid.reference_component.density
                )
                return super().gravity_force(grids, material) / default
            num_cells = int(np.sum([g.num_cells for g in grids]))
            values = np.zeros((self.nd, num_cells))
            # Angle of zero means force vector of [0, -1]
            values[1] = self.units.convert_units(-np.cos(gravity_angle), "m*s^-2")
            values[0] = self.units.convert_units(np.sin(gravity_angle), "m*s^-2")
            source = pp.wrap_as_dense_ad_array(values.ravel("F"), name="gravity force")
            return source

        def _bound_sides(
            self,
            grid: pp.Grid | pp.BoundaryGrid,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return the sides of the grid where tailored BCs are applied.

            Parameters:
                grid: Grid to provide boundary sides for.

            Returns:
                Tuple of arrays with the indices of the sides where the tailored BCs
                are applied. First one will always have Dirichlet BCs, second one will
                have Neumann BCs if neu_val_top is not None. Nonzero values are assigned
                to the top boundary.

            """
            sides = self.domain_boundary_sides(grid)
            if self.nd == 2:
                return sides.south, sides.north
            else:
                return sides.bottom, sides.top

        def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """Boundary pressure values.

            Parameters:
                grids: List of subdomain or interface grids where the boundary pressure
                    values are defined.

            Returns:
                Cell-wise nd-vector source term operator.

            """
            b_val = np.zeros(boundary_grid.num_cells)
            if boundary_grid.dim == (self.nd - 1) and dir_val_top is not None:
                b_val[self._bound_sides(boundary_grid)[1]] = dir_val_top
            return b_val

        def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """Darcy flux values for the Neumann boundary condition.

            Parameters:
                boundary_grid: Boundary grid to provide values for.

            Returns:
                An array with ``shape=(boundary_grid.num_cells,)`` containing the
                volumetric Darcy flux values on the provided boundary grid. Zero unless
                ``neu_val_top`` is not None.

            """
            vals = np.zeros(boundary_grid.num_cells)
            if boundary_grid.parent.dim == self.nd and neu_val_top is not None:
                cells = self._bound_sides(boundary_grid)[1]
                vals[cells] = neu_val_top * boundary_grid.cell_volumes[cells]
            return vals

        def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
            """Boundary conditions on all external boundaries.

            Parameters:
                sd: Subdomain grid on which to define boundary conditions.

            Returns:
                Boundary condition object. Dirichlet-type BC is assigned at the bottom.
                If neu_val_top is None, Dirichlet also on top.

            """
            # Define boundary faces.
            sides = self._bound_sides(sd)[0]
            if neu_val_top is None:
                sides += self._bound_sides(sd)[1]
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, sides, "dir")

    return Model(params)


class TestMixedDimGravity:
    """Test gravity effects in a mixed-dimensional flow model."""

    def solve(self):
        pp.run_time_dependent_model(self.model, self.model.params)
        pressure = self.model.equation_system.get_variable_values(
            [self.model.pressure_variable], time_step_index=0
        )
        return pressure

    def verify_pressure(self, p_known: float = 0):
        """Verify that the pressure of all subdomains equals p_known."""
        pressure = self.model.equation_system.get_variable_values(
            [self.model.pressure_variable], time_step_index=0
        )
        assert np.allclose(pressure, p_known, rtol=1e-3, atol=1e-3)

    def verify_mortar_flux(self, u_known: float):
        """Verify that the mortar flux of all interfaces equals u_known."""
        flux = self.model.equation_system.get_variable_values(
            [self.model.interface_darcy_flux_variable], time_step_index=0
        )
        assert np.allclose(np.abs(flux), u_known, rtol=1e-3, atol=1e-3)

    def verify_hydrostatic(self, angle=0, a=1e-1):
        """Check that the pressure profile is hydrostatic, with the adjustment
        for the fracture.
        Without the fracture, the profile is expected to be linear within each
        subdomain, with a small additional jump of aperture at the fracture.
        The full range is
        0 (bottom) to -1- aperture (top).
        """
        mdg = self.model.mdg
        sd_primary = mdg.subdomains(dim=mdg.dim_max())[0]
        data_primary = mdg.subdomain_data(sd_primary)
        p_primary = pp.get_solution_values(
            name="pressure", data=data_primary, time_step_index=0
        )

        # The cells above the fracture
        vertical_dim = self.model.nd - 1
        h = sd_primary.cell_centers[vertical_dim]
        ind = h > 0.5
        p_known = -(a * ind + h) * np.cos(angle)
        assert np.allclose(p_primary, p_known, rtol=1e-3, atol=1e-3)
        sd_secondary = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        data_secondary = mdg.subdomain_data(sd_secondary)
        p_secondary = pp.get_solution_values(
            name="pressure", data=data_secondary, time_step_index=0
        )

        # Half the additional jump is added to the fracture pressure
        h = sd_secondary.cell_centers[vertical_dim]
        p_known = -(a / 2 + h) * np.cos(angle)

        assert np.allclose(p_secondary, p_known, rtol=1e-3, atol=1e-3)
        flux = self.model.equation_system.get_variable_values(
            [self.model.interface_darcy_flux_variable], time_step_index=0
        )
        assert np.allclose(flux, 0, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "discretization,grid_type,num_nodes_mortar,num_nodes_1d",
        gravity_parameter_combinations,
    )
    def test_no_flow_neumann(
        self, discretization, grid_type, num_nodes_mortar, num_nodes_1d
    ):
        """Use homogeneous Neumann boundary conditions on top Dirichlet
        on bottom.

        The pressure distribution should be hydrostatic.

        """

        params = {
            "meshing_arguments": {"cell_size": 1 / 3, "cell_size_y": 1 / 2},
            "grid_type": grid_type,
        }
        self.model = model_setup_gravity(
            dimension=2,
            model_params=params,
            num_nodes_mortar=num_nodes_mortar,
            num_nodes_1d=num_nodes_1d,
            neu_val_top=0,
            discretization_method=discretization,
        )
        self.solve()
        self.verify_hydrostatic()
        self.verify_mortar_flux(0)

    @pytest.mark.parametrize(
        "discretization,num_nodes_mortar",
        [
            (discretization, num_nodes_mortar)
            for discretization in discretizations
            for num_nodes_mortar in mortar_nodes
        ],
    )
    @pytest.mark.parametrize(
        "angle",
        [0, np.pi / 2, np.pi],
    )
    def test_no_flow_rotate_gravity(self, discretization, num_nodes_mortar, angle):
        """Rotate the angle of gravity. Neumann boundaries except Dirichlet on bottom.

        There should be no flow.

        This test is only run for cartesian grids, as the simplex grids violate
        assumptions on number of cells in the 2d grid.

        """
        # The angle pi/2 requires nx = 1 for there not to be flow
        params = {
            "meshing_arguments": {"cell_size": 1.0, "cell_size_y": 1 / 2},
            "grid_type": "cartesian",
        }
        num_nodes_1d = 2
        self.model = model_setup_gravity(
            2,
            params,
            num_nodes_mortar=num_nodes_mortar,
            num_nodes_1d=num_nodes_1d,
            neu_val_top=0,
            gravity_angle=angle,
            discretization_method=discretization,
        )
        self.solve()
        if np.isclose(angle, np.pi / 2):
            self.verify_pressure()
        else:
            self.verify_hydrostatic(angle)
        self.verify_mortar_flux(0)

    @pytest.mark.parametrize(
        "discretization,grid_type,num_nodes_mortar,num_nodes_1d",
        gravity_parameter_combinations,
    )
    def test_no_flow_dirichlet(
        self, discretization, grid_type, num_nodes_mortar, num_nodes_1d
    ):
        """
        Dirichlet boundary conditions, but set so that the pressure is hydrostatic,
        and there is no flow.

        """

        params = {
            "meshing_arguments": {"cell_size": 1 / 3, "cell_size_y": 1 / 2},
            "grid_type": grid_type,
        }

        self.model = model_setup_gravity(
            2,
            params,
            num_nodes_mortar=num_nodes_mortar,
            num_nodes_1d=num_nodes_1d,
            dir_val_top=-1.1,
            discretization_method=discretization,
        )
        self.solve()
        self.verify_hydrostatic()
        self.verify_mortar_flux(0)

    @pytest.mark.parametrize(
        "discretization,grid_type,num_nodes_mortar,num_nodes_1d",
        gravity_parameter_combinations,
    )
    def test_inflow_top(
        self, discretization, grid_type, num_nodes_mortar, num_nodes_1d
    ):
        """
        Prescribed inflow at the top. Strength of the flow counteracts gravity, so that
        the pressure is uniform.
        """
        params = {
            "meshing_arguments": {"cell_size": 1 / 2},
            "grid_type": grid_type,
        }
        a = 1e-2
        self.model = model_setup_gravity(
            2,
            params,
            num_nodes_mortar=num_nodes_mortar,
            num_nodes_1d=num_nodes_1d,
            neu_val_top=-1,
            aperture=a,
            discretization_method=discretization,
        )
        self.solve()
        self.verify_pressure()
        self.verify_mortar_flux(1 / (num_nodes_mortar - 1))

    @pytest.mark.parametrize(
        "discretization,grid_type,num_nodes_mortar,num_nodes_1d",
        gravity_parameter_combinations,
    )
    def test_uniform_pressure(
        self, discretization, grid_type, num_nodes_mortar, num_nodes_1d
    ):
        """
        Prescribed pressure at the top. Strength of the flow counteracts gravity, so
        that the pressure is uniform.

        The total flow should equal the gravity force (=1).

        """
        self.model = model_setup_gravity(
            2,
            {"grid_type": grid_type},
            num_nodes_mortar=num_nodes_mortar,
            num_nodes_1d=num_nodes_1d,
            discretization_method=discretization,
            dir_val_top=0,
            aperture=3e-3,
        )
        self.solve()
        self.verify_pressure()

        self.verify_mortar_flux(1 / (num_nodes_mortar - 1))

    # --3d section. See analogous methods/tests above for documentation --#

    @pytest.mark.parametrize("discretization", discretizations)
    def test_one_fracture_3d_no_flow_neumann(self, discretization):
        self.model = model_setup_gravity(
            3,
            {"grid_type": "cartesian"},
            discretization_method=discretization,
            neu_val_top=0,
        )
        self.solve()
        self.verify_hydrostatic()
        self.verify_mortar_flux(0)

    @pytest.mark.parametrize("discretization", discretizations)
    def test_one_fracture_3d_no_flow_dirichlet(self, discretization):
        self.model = model_setup_gravity(
            3,
            {"grid_type": "cartesian"},
            discretization_method=discretization,
            dir_val_top=-1.1,
        )
        self.solve()
        self.verify_hydrostatic()
        self.verify_mortar_flux(0)
