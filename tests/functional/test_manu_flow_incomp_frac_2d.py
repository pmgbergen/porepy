"""
This module contains functional tests for approximations to the set of equations
modeling the 2d, incompressible flow with a single, fully embedded vertical fracture.

The manufactured solution is given in Appendix D1 from [1].

Tests:

    [TST-1] Relative L2-error on Cartesian grids for primary and secondary variables.

    [TST-2] Observed order of convergence (using four levels of refinement) for primary
      and secondary variables, both on Cartesian and simplicial grids.


References:

    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu,
      F. A. (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

import numpy as np
import porepy as pp
import pytest

from tests.functional.utils.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.manu_flow_incomp_frac_2d import (
    ManuIncompFlowSetup,
    manu_incomp_fluid,
    manu_incomp_solid,
)


# --> Declaration of module-wide fixtures that are re-used throughout the tests
@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Set material constants. Use default values provided in the module where the
    setup class is included.

    Returns:
        Dictionary containing the material constants with the `solid` and `fluid`
        constant classes.

    """
    solid_constants = pp.SolidConstants(manu_incomp_solid)
    fluid_constants = pp.FluidConstants(manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


# --> [TST-1] Relative L2-errors on Cartesian grid for three different times

# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants: dict) -> dict[str, float]:
    """Run verification setup.

    Parameters:
        material_constants: Dictionary containing the material constant classes.

    Returns:
        Dictionary of actual relative L2-errors.

    """

    # Define model parameters
    params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "mesh_arguments": {"cell_size": 0.125},
    }

    # Run simulation
    setup = ManuIncompFlowSetup(params)
    pp.run_stationary_model(setup, params)

    # Collect errors to facilitate comparison afterwards
    errors: dict[str, float] = {
        "error_matrix_pressure": setup.results[0].error_matrix_pressure,
        "error_matrix_flux": setup.results[0].error_matrix_flux,
        "error_frac_pressure": setup.results[0].error_frac_pressure,
        "error_frac_flux": setup.results[0].error_frac_flux,
        "error_intf_flux": setup.results[0].error_intf_flux,
    }
    return errors


# ----> Set desired L2-errors
@pytest.fixture(scope="module")
def desired_l2_errors() -> dict[str, float]:
    """Set desired L2-relative errors.

    Returns:
        Dictionary of desired relative L2-errors.

    """
    return {
        'error_matrix_pressure': 0.060732124330406576,
        'error_matrix_flux': 0.01828457897868048,
        'error_frac_pressure': 4.984308951373194,
        'error_frac_flux': 0.0019904878330327946,
        'error_intf_flux': 3.1453166913070185,
    }


# ----> Now, we write the actual test
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
def test_relative_l2_errors_cartesian_grid(
        var: str,
        actual_l2_errors: dict[str, float],
        desired_l2_errors: dict[str, float],
) -> None:
    """Check L2-relative errors for primary and secondary variables.

    Note:
        Tests should pass as long as the `desired_error` matches the `actual_error`,
        up to absolute (1e-6) and relative (1e-5) tolerances. The values of such
        tolerances aim at keeping the test meaningful while minimizing the chances of
        failure due to floating-point arithmetic close to machine precision.

        For this functional test, we are comparing errors for the pressure (for the
        matrix and the fracture) and fluxes (for the matrix, the fracture, and on the
        interface). The errors are measured in a discrete relative L2-error norm. The
        desired errors were obtained by running the model using the physical constants
        from :meth:`˜material_constants` on a Cartesian grid with 64 cells.

    Parameters:
        var: Name of the variable to be tested.
        actual_l2_errors: Dictionary containing the actual L2-relative errors.
        desired_l2_errors: Dictionary containing the desired L2-relative errors.

    """
    np.testing.assert_allclose(
        actual_l2_errors["error_" + var],
        desired_l2_errors["error_" + var],
        atol=1e-6,
        rtol=1e-5,
    )


# --> [TST-2] Observed order of convergence on Cartesian and simplices

# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(material_constants: dict) -> list[dict[str, float]]:
    """Retrieve actual order of convergence.

    Note:
        This is a spatial analysis, where the spatial step size is decreased by a
        factor of `2`. We consider `4` levels of successive refinements.

    Parameters:
        material_constants: Dictionary containing the material constants.

    Returns:
        List of dictionaries containing the actual observed order of convergence. The
        first item corresponds to the results obtained with Cartesian grids, and the
        second with simplices.

    """
    ooc: list[dict[str, float]] = []
    for grid_type in ["cartesian", "simplex"]:
        params = {
            "grid_type": grid_type,
            "material_constants": material_constants,
            "mesh_arguments": {"cell_size": 0.125},
        }
        conv_analysis = ConvergenceAnalysis(
            model_class=ManuIncompFlowSetup,
            model_params=params,
            levels=4,
            spatial_rate=2,
        )
        results = conv_analysis.run_analysis()
        ooc.append(conv_analysis.order_of_convergence(results))

    return ooc


# ----> Set desired order of convergence
@pytest.fixture(scope="module")
def desired_ooc() -> list[dict[str, float]]:
    """Set desired order of convergence.

    Returns:
        List of dictionaries, containing the desired order of convergence. The first
        entry corresponds to Cartesian grids and second index correspond to simplices.

    """
    desired_cartesian = {
        'ooc_frac_flux': 1.3967607529647372,
        'ooc_frac_pressure': 1.8534965991529369,
        'ooc_intf_flux': 1.9986496319323037,
        'ooc_matrix_flux': 1.660155297595291,
        'ooc_matrix_pressure': 1.900941278698522
    }
    desired_simplex = {
        'ooc_frac_flux': 1.300742223520386,
        'ooc_frac_pressure': 1.9739463002335342,
        'ooc_intf_flux': 2.0761838366094403,
        'ooc_matrix_flux': 1.7348556672914186,
        'ooc_matrix_pressure': 1.904889457326223,
    }

    return [desired_cartesian, desired_simplex]


# ----> Now, we write the actual test
@pytest.mark.skip(reason="slow")
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
@pytest.mark.parametrize("grid_type_idx", [0, 1])
def test_order_of_convergence(
        var: str,
        grid_type_idx: int,
        actual_ooc: list[dict[str, float]],
        desired_ooc: list[dict[str, float]],
) -> None:
    """Test observed order of convergence.

    Note:
        We loosen the tolerances in the case of simplicial grids relative to
        Cartesian grids. The reason behind this design choice is to allow for minor
        changes in the observed order of convergence due to changes in mesh generation
        (i.e., due to the use of different Gmsh versions).

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
    assert 1.0 < actual_ooc[grid_type_idx]["ooc_" + var]

    if grid_type_idx == 0:  # Cartesian
        assert np.isclose(
            desired_ooc[grid_type_idx]["ooc_" + var],
            actual_ooc[grid_type_idx]["ooc_" + var],
            atol=1e-3,  # allow for an absolute difference of 0.001 in OOC
            rtol=1e-3,  # allow for 0.1% of relative difference in OOC
        )
    else:  # Simplex
        assert np.isclose(
            desired_ooc[grid_type_idx]["ooc_" + var],
            actual_ooc[grid_type_idx]["ooc_" + var],
            atol=1e-1,  # allow for an absolute difference of 0.1 in OOC
            rtol=5e-1,  # allow for 5% of relative difference in OOC
        )

