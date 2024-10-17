"""
This module contains functional tests for approximations to the set of equations
modeling the 2d and 3d, incompressible flow with a single, fully embedded vertical
fracture.

The 2d manufactured solution is given in Appendix D.1 from [1]. The 3d manufactured
solution is based on a slightly modified version of the solution given in Appendix D.2
from [1] (i.e., the bubble function is scaled to obtain a better conditioned system).

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
from tests.functional.setups.manu_flow_incomp_frac_2d import (
    ManuIncompFlowSetup2d,
    manu_incomp_fluid,
    manu_incomp_solid,
)
from tests.functional.setups.manu_flow_incomp_frac_3d import ManuIncompFlowSetup3d


# --> Declaration of module-wide fixtures that are re-used throughout the tests
@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Set material constants.

    Use default values provided in the module where the setup class is included.

    Returns:
        Dictionary containing the material constants with the `solid` and `fluid`
        constant classes.

    """
    solid_constants = pp.SolidConstants(manu_incomp_solid)
    fluid_constants = pp.FluidConstants(manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


# --> [TEST_1] Relative L2-errors on Cartesian grid


# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants: dict) -> list[dict[str, float]]:
    """Run verification setups and retrieve results.

    Parameters:
        material_constants: Dictionary containing the material constant classes.

    Returns:
        List of dictionaries containing the actual relative L2-errors. The first item of
        the list corresponds to the results for 2d, whereas the second for 3d.

    """

    # Define model parameters (same for 2d and 3d).
    model_params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
    }

    # Retrieve actual L2-relative errors
    errors: list[dict[str, float]] = []
    # Loop through models, i.e., 2d and 3d
    for model in [ManuIncompFlowSetup2d, ManuIncompFlowSetup3d]:
        setup = model(deepcopy(model_params))  # make deep copy of params to avoid nasty bugs
        pp.run_time_dependent_model(setup)
        errors.append(
            {
                "error_matrix_pressure": setup.results[0].error_matrix_pressure,
                "error_matrix_flux": setup.results[0].error_matrix_flux,
                "error_frac_pressure": setup.results[0].error_frac_pressure,
                "error_frac_flux": setup.results[0].error_frac_flux,
                "error_intf_flux": setup.results[0].error_intf_flux,
            }
        )

    return errors


# ----> Set desired L2-errors
@pytest.fixture(scope="module")
def desired_l2_errors() -> list[dict[str, float]]:
    """Set desired L2-relative errors.

    Returns:
        List dictionaries containing the desired relative L2-errors.

    """
    # Desired errors for 2d
    desired_errors_2d = {
        "error_matrix_pressure": 0.060732124330406576,
        "error_matrix_flux": 0.01828457897868048,
        "error_frac_pressure": 4.984308951373194,
        "error_frac_flux": 0.0019904878330327946,
        "error_intf_flux": 3.1453166913070185,
    }

    # Desired error for 3d
    desired_errors_3d = {
        "error_matrix_pressure": 1.3822466693314728,
        "error_matrix_flux": 1.2603123149160123,
        "error_frac_pressure": 6.272401337799361,
        "error_frac_flux": 0.044759629637959035,
        "error_intf_flux": 5.291360607983224,
    }

    return [desired_errors_2d, desired_errors_3d]


@pytest.mark.parametrize("dim_idx", [0, 1])
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
def test_relative_l2_errors_cartesian_grid(
    dim_idx: int,
    var: str,
    actual_l2_errors: list[dict[str, float]],
    desired_l2_errors: list[dict[str, float]],
) -> None:
    """Check L2-relative errors for primary and secondary variables.

    Note:
        Tests should pass as long as the `desired_error` matches the `actual_error`,
        up to absolute (1e-8) and relative (1e-5) tolerances. The values of such
        tolerances aim at keeping the test meaningful while minimizing the chances of
        failure due to floating-point arithmetic close to machine precision.

        For this functional test, we are comparing errors for the pressure (for the
        matrix and the fracture) and fluxes (for the matrix, the fracture, and on the
        interface). The errors are measured in a discrete relative L2-error norm. The
        desired errors were obtained by running the model using the physical constants
        from :meth:`~material_constants` on a Cartesian grid with 64 cells.

    Parameters:
        dim_idx: Dimension index acting on `actual_l2_errors` and `desired_l2_errors`.
            `0` refers to 2d and `1` to 3d.
        var: Name of the variable to be tested.
        actual_l2_errors: List of dictionaries containing the actual L2-relative errors.
        desired_l2_errors: List of dictionaries containing the desired L2-relative
            errors.

    """
    np.testing.assert_allclose(
        actual_l2_errors[dim_idx]["error_" + var],
        desired_l2_errors[dim_idx]["error_" + var],
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
        This is a spatial analysis, where the spatial step size is decreased by a
        factor of `2`. We consider `4` levels of refinement for 2d and `3` levels of
        refinement for 3d.

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
    for model_idx, model in enumerate([ManuIncompFlowSetup2d, ManuIncompFlowSetup3d]):
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
                    )
                else:
                    conv_analysis = ConvergenceAnalysis(
                        model_class=model,
                        model_params=deepcopy(params),
                        levels=3,
                        spatial_refinement_rate=2,
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
        List of dictionaries, containing the desired order of convergence. The first
        entry corresponds to Cartesian grids and second index correspond to simplices.

    """
    desired_ooc_2d = [
        {  # Cartesian
            "ooc_frac_flux": 1.3967607529647372,
            "ooc_frac_pressure": 1.8534965991529369,
            "ooc_intf_flux": 1.9986496319323037,
            "ooc_matrix_flux": 1.660155297595291,
            "ooc_matrix_pressure": 1.900941278698522,
        },
        {  # simplex
            "ooc_frac_flux": 1.300742223520386,
            "ooc_frac_pressure": 1.9739463002335342,
            "ooc_intf_flux": 2.0761838366094403,
            "ooc_matrix_flux": 1.7348556672914186,
            "ooc_matrix_pressure": 1.904889457326223,
        },
    ]

    desired_ooc_3d = [
        {  # Cartesian
            "ooc_frac_flux": 2.0540239290134323,
            "ooc_frac_pressure": 2.01831767379812,
            "ooc_intf_flux": 2.005622051446942,
            "ooc_matrix_flux": 2.1319834447112367,
            "ooc_matrix_pressure": 2.007165614273335,
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
        actual_ooc: List of dictionaries containing the actual observed order of
            convergence.
        desired_ooc: List of dictionaries containing the desired observed order of
            convergence.

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
