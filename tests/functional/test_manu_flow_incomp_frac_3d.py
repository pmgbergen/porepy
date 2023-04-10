"""
This module contains functional tests for approximations to the set of equations
modeling the 2d, incompressible flow with a single, fully embedded vertical fracture.

The manufactured solution is given in Appendix D2 from [1].

Tests:

    [TST-1] Relative L2-error on Cartesian grids for primary and secondary variables.

    [TST-2] Observed order of convergence (using three levels of refinement) for primary
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
    manu_incomp_fluid,
    manu_incomp_solid,
)
from tests.functional.setups.manu_flow_incomp_frac_3d import ManuIncompFlow3d


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
    setup = ManuIncompFlow3d(params)
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
        'error_matrix_pressure': 1.3822466693314728,
        'error_matrix_flux': 1.2603123149160123,
        'error_frac_pressure': 6.272401337799361,
        'error_frac_flux': 0.044759629637959035,
        'error_intf_flux': 5.291360607983224
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
        from :meth:`˜material_constants` on a Cartesian grid with 512 cells.

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


# --> [TST-2] Observed order of convergence on Cartesian grids

# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(material_constants: dict) -> dict[str, float]:
    """Retrieve actual order of convergence.

    Note:
        This is a spatial analysis, where the spatial step size is decreased by a
        factor of `2`. We consider `4` levels of successive refinements.

    Parameters:
        material_constants: Dictionary containing the material constants.

    Returns:
        Dictionary containing the actual observed order of convergence.

    """
    params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "mesh_arguments": {"cell_size": 0.125},
    }
    conv_analysis = ConvergenceAnalysis(
        model_class=ManuIncompFlow3d,
        model_params=params,
        levels=4,
        spatial_rate=2,
    )
    results = conv_analysis.run_analysis()

    return conv_analysis.order_of_convergence(results)


# ----> Set desired order of convergence
@pytest.fixture(scope="module")
def desired_ooc() -> dict[str, float]:
    """Set desired order of convergence.

    Returns:
        Dictionary containing the desired order of convergence.

    """
    return {
        'ooc_frac_flux': 2.0318156925119903,
        'ooc_frac_pressure': 2.012288879279475,
        'ooc_intf_flux': 2.0034099549330073,
        'ooc_matrix_flux': 2.0939332795555488,
        'ooc_matrix_pressure': 2.0042937595376835
    }


# ----> Now, we write the actual test
@pytest.mark.skip(reason="slow")
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
def test_order_of_convergence(
        var: str,
        actual_ooc: dict[str, float],
        desired_ooc: dict[str, float],
) -> None:
    """Test observed order of convergence.

    Parameters:
        var: Name of the variable to be tested.
        actual_ooc: Dictionary containing the actual observed order of convergence.
        desired_ooc: Dictionary containing the desired observed order of convergence.

    """

    # We require the order of convergence to always be larger than 1.0
    assert 1.0 < actual_ooc["ooc_" + var]

    # We now assert
    assert np.isclose(
        desired_ooc["ooc_" + var],
        actual_ooc["ooc_" + var],
        atol=1e-3,  # allow for an absolute difference of 0.001 in OOC
        rtol=1e-3,  # allow for 0.1% of relative difference in OOC
    )

