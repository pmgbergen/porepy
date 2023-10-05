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

from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils import models, well_models
from porepy.models.fluid_mass_balance import SinglePhaseFlow


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
    solid = pp.SolidConstants(models.granite_values)
    fluid = pp.FluidConstants(models.water_values)

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
    # Failure here could be mean two things:
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


@pytest.mark.parametrize(
    "method_name, expected_value, dimension_restriction",
    [
        ("aperture", np.array([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01]), None),
        # Darcy fluxes (with unitary values for the viscosity).
        (
            "darcy_flux",
            np.array(
                [
                    -4.0e-13,
                    -1.0e-12,
                    4.0e-13,
                    -4.0e-13,
                    -1.0e-12,
                    4.0e-13,
                    -4.0e-13,
                    -4.0e-13,
                    -1.0e-12,
                    -1.0e-12,
                    4.0e-13,
                    4.0e-13,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    1.0e-12,
                    -8.0e-15,
                    -1.0e-12,
                    8.0e-15,
                    1.0e-12,
                    -8.0e-15,
                    -1.0e-12,
                    8.0e-15,
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
        ("fluid_compressibility", 4e-10, None),
        # rho = rho_ref * exp(c_f * (p - p_ref))
        ("fluid_density", 1000 * np.exp(4e-10 * 200 * pp.BAR), None),
        # Values of the mass fluxes (i.e., scaling with rho/mu is incorporated).
        (
            "fluid_flux",
            np.array(
                [
                    -4.00000000e-07,
                    -1.00803209e-06,
                    4.03212834e-07,
                    -4.00000000e-07,
                    -1.00803209e-06,
                    4.03212834e-07,
                    -4.00000000e-07,
                    -4.00000000e-07,
                    -1.00803209e-06,
                    -1.00803209e-06,
                    4.03212834e-07,
                    4.03212834e-07,
                    1.00803209e-06,
                    1.00803209e-06,
                    1.00803209e-06,
                    1.00803209e-06,
                    -8.00000000e-09,
                    -1.00803209e-06,
                    8.06425668e-09,
                    1.00803209e-06,
                    -8.00000000e-09,
                    -1.00803209e-06,
                    8.06425668e-09,
                    1.00803209e-06,
                ]
            ),
            None,
        ),
        # fluid_mass = rho * phi * specific_volume
        (
            "fluid_mass",
            np.array(
                [
                    1.76405615e00,
                    1.76405615e00,
                    1.76405615e00,
                    1.76405615e00,
                    3.52811230e-02,
                    3.52811230e-02,
                    3.52811230e-02,
                    3.52811230e-02,
                    7.05622460e-04,
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
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.01606417e-06,
                        2.01606417e-06,
                        2.01606417e-06,
                        2.01606417e-06,
                        4.03212834e-06,
                    ]
                ]
            ),
            None,
        ),
        ("fluid_viscosity", 0.001, None),
        ("interface_darcy_flux_equation", 5e9, None),
        ("interface_fluid_flux", 1.00803209e-06, None),
        ("interface_flux_equation", 5e9, None),
        (
            "mass_balance_equation",
            np.array(
                [
                    1.40589657e-02,
                    1.40589689e-02,
                    1.40589689e-02,
                    1.40589721e-02,
                    2.80122961e-04,
                    2.80123025e-04,
                    2.80122961e-04,
                    2.80123025e-04,
                    1.59033151e-06,
                ]
            ),
            None,
        ),
        ("mobility", 1 / 0.001, None),
        # Combination of mobility and fluid density = rho/mu
        # = rho_ref * exp(c_f * (p - p_ref)) / mu = 1000 * exp(4e-10 * 2e7) /  0.001
        ("mobility_rho", 1008032.0855042734, None),
        ("normal_permeability", 1.0, None),
        ("permeability", 1e-20, None),
        ("porosity", 7e-3, None),
        ("pressure", 200 * pp.BAR, None),
        # pressure_exponential = exp(c_f * (p - p_ref))
        ("pressure_exponential", np.exp(4e-10 * 200 * pp.BAR), None),
        (
            "pressure_trace",
            np.array(
                [
                    0.00e00,
                    -3.00e07,
                    0.00e00,
                    0.00e00,
                    -3.00e07,
                    0.00e00,
                    0.00e00,
                    0.00e00,
                    -3.00e07,
                    -3.00e07,
                    0.00e00,
                    0.00e00,
                    -3.00e07,
                    -3.00e07,
                    -3.00e07,
                    -3.00e07,
                    0.00e00,
                    -2.48e09,
                    0.00e00,
                    -2.48e09,
                    0.00e00,
                    -2.48e09,
                    0.00e00,
                    -2.48e09,
                ]
            ),
            None,
        ),
        ("reference_pressure", 0, None),
        ("skin_factor", 0, None),
        ("tangential_component", np.array([[1.0, 0.0]]), 0),  # check only for 0d
        ("well_fluid_flux", 0, 2),  # use dim_restriction=2 to ignore well flux
        ("well_flux_equation", 0, 2),  # use dim_restriction=2 to ignore well equation
        ("well_radius", 0.1, None),
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
    # Get the method to be tested in callable form..
    method: Callable = getattr(model_setup, method_name)

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
    val = operator.evaluate(model_setup.equation_system)
    if isinstance(val, pp.ad.AdArray):
        val = val.val
    elif isinstance(val, sps.bsr_matrix):  # needed for `tangential_component`
        val = val.A

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
            :class:`~pp.models.material_constants.MaterialConstants`.

    """

    class Model(SquareDomainOrthogonalFractures, SinglePhaseFlow):
        """Single phase flow model in a domain with two intersecting fractures."""

    params = {
        "suppress_export": True,  # Suppress output for tests
        "fracture_indices": [0, 1],
        "cartesian": True,
    }
    reference_params = copy.deepcopy(params)
    reference_params["file_name"] = "unit_conversion_reference"

    # Create model and run simulation
    setup_0 = Model(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = Model(params)

    pp.run_time_dependent_model(setup_1, params)
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
    pp.fluid_mass_balance.SinglePhaseFlow,
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
            "solid": pp.SolidConstants({"permeability": 1e-6 / 4, "well_radius": 0.01})
        },
        # Use only the horizontal fracture of OrthogonalFractures3d
        "fracture_indices": [2],
    }

    setup = WellModel(params)
    pp.run_time_dependent_model(setup, params)
    # Check that the matrix pressure is close to linear in z
    matrix = setup.mdg.subdomains(dim=3)[0]
    matrix_pressure = setup.pressure([matrix]).evaluate(setup.equation_system).val
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
    fracture_pressure = setup.pressure(fracs).evaluate(setup.equation_system).val
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
    well_pressure = setup.pressure(wells).evaluate(setup.equation_system).val

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
