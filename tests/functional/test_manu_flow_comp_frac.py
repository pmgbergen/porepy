"""
This module contains functional tests for approximations to the set of equations
modeling the 2d and 3d, *compressible* flow with a single, fully embedded vertical
fracture.

The manufactured solution for the compressible flow verification is obtained as a
natural extension of the incompressible case, see [1]. The non-linearity is included
via the dependency of the fluid density with the fluid pressure:

.. math::

    \\rho(p) = \\rho_0 \\exp{(c_f (p - p_0))},

where, for the tests included here, :math:`\\rho_0 = 1` [kg * m^-3], :math:`c_f = 0.2`
[Pa^-1], and :math:`p_0 = 0` [Pa]. The rest of the physical parameter are given unitary
values, except for the reference porosity :math:`\\phi_0 = 0.1` and normal permeability
:math:`\\kappa = 0.5`.

Tests:

    [TEST_1] Relative L2-error on Cartesian grids for primary and secondary variables
      for three different times for 2d and 3d.

    [TEST_2] Observed order of convergence (using four levels of refinement for 2d and
      three levels of refinement for 3d) for primary and secondary variables. Order
      of convergence using Cartesian grids are tested for 2d and 3d, whereas
      simplicial grids are limited to 2d.


References:

    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu,
      F. A. (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.manu_flow_comp_2d_frac import (
    ManuCompFlowSetup2d,
    manu_comp_fluid,
    manu_comp_solid,
)
from tests.functional.setups.manu_flow_comp_3d_frac import ManuCompFlowSetup3d


# --> Declaration of module-wide fixtures that are re-used throughout the tests
@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Set material constants.
    Use default values provided in the module where the setup class is included.

    Returns:
        Dictionary containing the material constants with the `solid` and `fluid`
        constant classes.

    """
    solid_constants = pp.SolidConstants(manu_comp_solid)
    fluid_constants = pp.FluidConstants(manu_comp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


# --> [TEST_1] Relative L2-errors on Cartesian grid for three different times


# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants: dict) -> list[list[dict[str, float]]]:
    """Run verification setups and retrieve results for the scheduled times.

    Parameters:
        material_constants: Dictionary containing the material constant classes.

    Returns:
        List of lists of dictionaries of actual relative errors. The outer list contains
        two items, the first contains the results for 2d and the second contains the
        results for 3d. Both inner lists contain two items each, each of which is a
        dictionary of results for the scheduled times, i.e., 0.5 [s], and 1.0 [s].

    """

    # Define model parameters (same for 2d and 3d).
    model_params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
        "time_manager": pp.TimeManager([0, 0.5, 1.0], 0.5, True),
    }

    # Retrieve actual L2-relative errors.
    errors: list[list[dict[str, float]]] = []
    # Loop through models, i.e., 2d and 3d.
    for model in [ManuCompFlowSetup2d, ManuCompFlowSetup3d]:
        setup = model(deepcopy(model_params))  # Make deep copy of params to avoid nasty bugs.
        pp.run_time_dependent_model(setup, {})
        errors_setup: list[dict[str, float]] = []
        # Loop through results, i.e., results for each scheduled time.
        for result in setup.results:
            errors_setup.append(
                {
                    "error_matrix_pressure": getattr(result, "error_matrix_pressure"),
                    "error_matrix_flux": getattr(result, "error_matrix_flux"),
                    "error_frac_pressure": getattr(result, "error_frac_pressure"),
                    "error_frac_flux": getattr(result, "error_frac_flux"),
                    "error_intf_flux": getattr(result, "error_intf_flux"),
                }
            )
        errors.append(errors_setup)

    return errors


# ----> Set desired L2-errors
@pytest.fixture(scope="module")
def desired_l2_errors() -> list[list[dict[str, float]]]:
    """Set desired L2-relative errors.

    Returns:
        List of lists of dictionaries containing the desired relative L2-errors.

    """

    # Desired errors for 2d
    desired_errors_2d = [
        {  # t = 0.5 [s]
            "error_matrix_pressure": 0.05860315482644138,
            "error_matrix_flux": 0.01728816711273373,
            "error_frac_pressure": 4.761115466428997,
            "error_frac_flux": 0.0027528176884234297,
            "error_intf_flux": 3.0521278709541946,
        },
        {  # t = 1.0 [s]
            "error_matrix_pressure": 0.056952568619002386,
            "error_matrix_flux": 0.017206997517806834,
            "error_frac_pressure": 4.7258340277590865,
            "error_frac_flux": 0.0036119330001357737,
            "error_intf_flux": 3.1023316529076546,
        },
    ]
    # Desired errors for 3d
    desired_errors_3d = [
        {  # t = 0.5 [s]
            "error_matrix_pressure": 0.044142110025893674,
            "error_matrix_flux": 0.020240531408035483,
            "error_frac_pressure": 7.345638542028673,
            "error_frac_flux": 0.04968518024390149,
            "error_intf_flux": 5.150695781155413,
        },
        {  # t = 1.0 [s]
            "error_matrix_pressure": 0.043341944057014324,
            "error_matrix_flux": 0.02031093722149098,
            "error_frac_pressure": 7.139915887008252,
            "error_frac_flux": 0.049748152094622,
            "error_intf_flux": 5.228345273854552,
        },
    ]

    return [desired_errors_2d, desired_errors_3d]


@pytest.mark.parametrize("dim_idx", [0, 1])
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
@pytest.mark.parametrize("time_idx", [0, 1])
def test_relative_l2_errors_cartesian_grid(
    dim_idx: int,
    var: str,
    time_idx: int,
    actual_l2_errors: list[list[dict[str, float]]],
    desired_l2_errors: list[list[dict[str, float]]],
) -> None:
    """Check L2-relative errors for primary and secondary variables.

    Note:
        Tests should pass as long as the `desired_error` matches the `actual_error`,
        up to absolute (1e-8) and relative (1e-5) tolerances. The values for such
        tolerances aim at keeping the test meaningful while minimizing the chances of
        failure due to floating-point arithmetic.

        For this functional test, we are comparing errors for the pressure (in the
        matrix and in the fracture) and fluxes (in the matrix, in the fracture,
        and on the interfaces). The errors are measured using the discrete relative
        L2-error norm. The desired errors were obtained by running the model using the
        physical constants from :meth:`~material_constants` on a Cartesian grid with
        64 cells in 2d and 512 in 3d. We test the errors for two different times,
        namely: 0.5 [s], and 1.0 [s].

    Parameters:
        dim_idx: Dimension index acting on the outer list of `actual_l2_errors` and
            `desired_l2_errors`. `0` refers to 2d and `1` to 3d.
        var: Name of the variable to be tested.
        time_idx: Time index acting on the inner lists of 'actual_l2_errors' and
            'desired_l2_errors'. `0` refers to 0.5 [s], and `1` to 1.0 [s].
        actual_l2_errors: List of lists of dictionaries containing the actual
            L2-relative errors.
        desired_l2_errors: List of lists of dictionaries containing the desired
            L2-relative errors.

    """
    np.testing.assert_allclose(
        actual_l2_errors[dim_idx][time_idx]["error_" + var],
        desired_l2_errors[dim_idx][time_idx]["error_" + var],
        atol=1e-8,
        rtol=1e-5,
    )


# --> [TEST_2] Observed order of convergence


# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(material_constants: dict) -> list[list[dict[str, float]]]:
    """Retrieve actual order of convergence.

    Cartesian and simplices for 2d. Cartesian only for 3d.

    Note:
        This is a spatio-temporal analysis, where the spatial step size is decreased
        by a factor of `2` and the temporal step size is decreased by a factor of `4`
        between each run. We have to do this so that the temporal error does not become
        dominant when the grid is refined, i.e., since MPFA is quadratically convergent,
        whereas Backward Euler is linearly convergent. We consider `4` levels of
        refinement for 2d and `3` levels of refinement for 3d.

    Parameters:
        material_constants: Dictionary containing the material constants.

    Returns:
        List of lists of dictionaries containing the actual observed order of
        convergence. The outer list contains two items, the first contains the results
        for 2d and the second for 3d. Each inner list contains the dictionaries with
        the observed order of convergence obtained with Cartesian and simplicial grids
        (only for 2d).

    """
    ooc: list[list[dict[str, float]]] = []
    # Loop through the models
    for model_idx, model in enumerate([ManuCompFlowSetup2d, ManuCompFlowSetup3d]):
        ooc_setup: list[dict[str, float]] = []
        # Loop through grid type
        for grid_type in ["cartesian", "simplex"]:
            # We do not perform a convergence analysis with simplices in 3d
            if model_idx == 1 and grid_type == "simplex":
                continue
            else:
                # Use same parameters for both 2d and 3d
                params = {
                    "grid_type": grid_type,
                    "material_constants": material_constants,
                    "meshing_arguments": {"cell_size": 0.125},
                }
                # Use 4 levels of refinement for 2d and 3 levels for 3d
                if model_idx == 0:
                    conv_analysis = ConvergenceAnalysis(
                        model_class=model,
                        model_params=deepcopy(params),
                        levels=4,
                        spatial_refinement_rate=2,
                        temporal_refinement_rate=4,
                    )
                else:
                    conv_analysis = ConvergenceAnalysis(
                        model_class=model,
                        model_params=deepcopy(params),
                        levels=3,
                        spatial_refinement_rate=2,
                        temporal_refinement_rate=4,
                    )
                results = conv_analysis.run_analysis()
                ooc_setup.append(conv_analysis.order_of_convergence(results))
        ooc.append(ooc_setup)

    return ooc


# ----> Set desired order of convergence
@pytest.fixture(scope="module")
def desired_ooc() -> list[list[dict[str, float]]]:
    """Set desired order of convergence.

    Returns:
        List of lists of dictionaries, containing the desired order of convergence.

    """
    desired_ooc_2d = [
        {  # Cartesian
            "ooc_frac_flux": 1.9207078517903355,
            "ooc_frac_pressure": 2.007469314704246,
            "ooc_intf_flux": 1.9975718577542623,
            "ooc_matrix_flux": 1.5071850357496581,
            "ooc_matrix_pressure": 2.2739632526704496,
        },
        {  # simplex
            "ooc_frac_flux": 1.8757689483196147,
            "ooc_frac_pressure": 2.076308534452869,
            "ooc_intf_flux": 2.0755337518063057,
            "ooc_matrix_flux": 1.6660162630966544,
            "ooc_matrix_pressure": 2.367319458006446,
        },
    ]

    desired_ooc_3d = [
        {  # Cartesian
            "ooc_frac_flux": 2.011343043274247,
            "ooc_frac_pressure": 1.985302288174025,
            "ooc_intf_flux": 1.9998583923263855,
            "ooc_matrix_flux": 1.6009304954707668,
            "ooc_matrix_pressure": 2.1529911615181723,
        }
    ]

    return [desired_ooc_2d, desired_ooc_3d]


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
@pytest.mark.parametrize("grid_type_idx", [0, 1])
@pytest.mark.parametrize("dim_idx", [0, 1])
def test_order_of_convergence(
    var: str,
    dim_idx: int,
    grid_type_idx: int,
    actual_ooc: list[list[dict[str, float]]],
    desired_ooc: list[list[dict[str, float]]],
) -> None:
    """Test observed order of convergence.

    Note:
        We set more flexible tolerances for simplicial grids compared to Cartesian
        grids. This is because we would like to allow for slight changes in the
        order of convergence if the meshes change, i.e., in newer versions of Gmsh.

    Parameters:
        var: Name of the variable to be tested.
        grid_type_idx: Index to identify the grid type; `0` for Cartesian, and `1`
            for simplices.
        dim_idx: Index to identify the dimensionality of the problem; `0` for 2d, and
            `1` for 3d.
        actual_ooc: List of lists of dictionaries containing the actual observed
            order of convergence.
        desired_ooc: List of lists of dictionaries containing the desired observed
            order of convergence.

    """
    # We require the order of convergence to always be larger than 1.0
    if not (dim_idx == 1 and grid_type_idx == 1):  # no analysis for 3d and simplices
        assert 1.0 < actual_ooc[dim_idx][grid_type_idx]["ooc_" + var]

    if grid_type_idx == 0:  # Cartesian
        assert np.isclose(
            desired_ooc[dim_idx][grid_type_idx]["ooc_" + var],
            actual_ooc[dim_idx][grid_type_idx]["ooc_" + var],
            atol=1e-3,  # allow for an absolute difference of 0.001 in OOC
            rtol=1e-3,  # allow for 0.1% of relative difference in OOC
        )
    else:  # Simplex
        if dim_idx == 0:  # no analysis for 3d and simplices
            assert np.isclose(
                desired_ooc[dim_idx][grid_type_idx]["ooc_" + var],
                actual_ooc[dim_idx][grid_type_idx]["ooc_" + var],
                atol=1e-1,  # allow for an absolute difference of 0.1 in OOC
                rtol=5e-1,  # allow for 5% of relative difference in OOC
            )
