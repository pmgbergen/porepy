"""Provides quantitative tests for AD methods used in the single phase flow model."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from tests.integration.models import setup_utils


@pytest.fixture(scope="function")
def model_setup():
    """Minimal compressible single-phase flow setup with a single horizontal fracture.

    The model is set up with realistic physical parameters using water and granite.
    A 2x2 Cartesian grid on a unit domain is employed. One horizontal and one vertical
    fracture are included in the domain. This results in the following mdg:

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
    solid = pp.SolidConstants(setup_utils.granite_values)
    fluid = pp.FluidConstants(setup_utils.water_values)

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
    setup.set_well_network()

    # Set constant but non-zero values for the primary variables
    if hasattr(setup, "pressure_variable"):
        setup.equation_system.set_variable_values(
            200 * pp.BAR * np.ones(setup.mdg.num_subdomain_cells()),
            [setup.pressure_variable],
            iterate_index=0,
        )
    if hasattr(setup, "interface_darcy_flux_variable"):
        setup.equation_system.set_variable_values(
            1e-12 * np.ones(setup.mdg.num_interface_cells()),
            [setup.interface_darcy_flux_variable],
            iterate_index=0,
        )

    return setup


@pytest.mark.parametrize(
    "method_name, expected_value, dimension_restriction",
    [
        ("aperture", np.array([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01]), None),
        ("bc_values_darcy", np.zeros(24), None),
        (
            "bc_values_mobrho",
            np.array(
                [
                    1e6,
                    0,
                    1e6,
                    1e6,
                    0,
                    1e6,
                    1e6,
                    1e6,
                    0,
                    0,
                    1e6,
                    1e6,
                    0,
                    0,
                    0,
                    0,
                    1e6,
                    0,
                    1e6,
                    0,
                    1e6,
                    0,
                    1e6,
                    0,
                ]
            ),
            None,
        ),
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
        ("fluid_density", 1000 * np.exp(4e-10 * 200 * pp.BAR), None),
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
        ("normal_permeability", 1.0, None),
        ("permeability", 1e-20, None),
        ("porosity", 7e-3, None),
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
    request,
) -> None:
    """Test that Ad operator methods return expected values.

    Parameters:
        model_setup: Prepared-for-simulation single phase flow model setup.
        method_name: Name of the method to be tested.
        expected_value: The expected value from the evaluation.
        dimension_restriction: Dimension in which the method is restricted. If None,
            all valid dimensions for the method will be considered.
        request: request pytest object.

    """
    # Check that the parametrization contains all possible testable methods.
    # Failure here means that there is at least one method that was added/renamed
    # in the core, but it was not included in this parametrization. To fix the test,
    # please add the missing method with the expected value as part of the
    # parametrization. If you don't know which method is missing, add a break point
    # before the first assert and compare the list of `testable_methods` with the
    # list of `tested_methods`.
    # JV: I've considered testing this only once (e.g., for the first triplet of the
    # parametrization). However, if the assertion is False, only the first test will
    # fail. This might be extremely confusing for someone trying to debug the test. If
    # you are worried about this being slow, is really not (it takes only ~1.5 ms to
    # run the next three lines).
    testable_methods: list[str] = setup_utils.get_testable_methods_names(model_setup)
    tested_methods: list[str] = setup_utils.methods_from_pytest_session(request)
    assert testable_methods == tested_methods

    # Get the method to be tested.
    method: Callable = getattr(model_setup, method_name)

    # Obtain list of subdomain or interface grids where the method is defined.
    domains = setup_utils.domains_from_method_name(
        model_setup.mdg,
        method,
        dimension_restriction,
    )

    # Obtain operator in AD form.
    operator = method(domains)

    # Discretize (if necessary), evaluate, and retrieve numerical values.
    assert isinstance(operator, pp.ad.Operator)
    operator.discretize(model_setup.mdg)
    val = operator.evaluate(model_setup.equation_system)
    if isinstance(val, pp.ad.AdArray):
        val = val.val
    elif isinstance(val, sps.bsr_matrix):  # needed for `tangential_component`
        val = val.A

    # Compare the actual and expected values.
    assert np.allclose(val, expected_value, rtol=1e-8, atol=1e-15)

