"""
This module contains functional tests for approximations to the set of equations
modeling the 3d, *compressible* flow with a single, fully embedded vertical fracture.

The manufactured solution for the compressible flow verification is obtained as a
natural extension of the incompressible case, see [1]. The non-linearity is included
via the dependency of the fluid density with the fluid pressure:

.. math:

    \\rho(p) = \\rho_0 \\exp{(c_f (p - p_0))},

where, for the tests included here, :math:`\\rho_0 = 1` [kg * m^-3], :math:`c_f = 0.2`
[Pa^-1], and :math:`p_0 = 0` [Pa]. The rest of the physical parameter are given unitary
values, except for the reference porosity :math:`\\phi_0 = 0.1` and normal permeability
:math:`\\kappa = 0.5`.

Tests:

    [TST-1] Relative L2-error on Cartesian grids for primary and secondary variables for
      three different times.

    [TST-2] Observed order of convergence (using three levels of refinement) for primary
      and secondary variables on Cartesian grids.

References:

    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu,
    F. A. (2022). A posteriori error estimates for hierarchical mixed-dimensional
    elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
from tests.functional.utils.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.manu_flow_comp_3d_frac import ManuCompFlowSetup3d
from tests.functional.setups.manu_flow_comp_2d_frac import (
    manu_comp_fluid,
    manu_comp_solid,
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
    solid_constants = pp.SolidConstants(manu_comp_solid)
    fluid_constants = pp.FluidConstants(manu_comp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


# --> [TST-1] Relative L2-errors on Cartesian grid for three different times

# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants: dict) -> list[dict[str, float]]:
    """Run verification setup and retrieve results for the scheduled times.

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
        "time_manager": pp.TimeManager([0, 0.2, 0.8, 1.0], 0.2, True),
    }

    # Run simulation
    setup = ManuCompFlowSetup3d(params)
    pp.run_time_dependent_model(setup, params)

    # Collect errors to facilitate comparison afterwards
    errors: list[dict[str, float]] = []
    for result in setup.results:
        errors.append(
            {
                "error_matrix_pressure": getattr(result, "error_matrix_pressure"),
                "error_matrix_flux": getattr(result, "error_matrix_flux"),
                "error_frac_pressure": getattr(result, "error_frac_pressure"),
                "error_frac_flux": getattr(result, "error_frac_flux"),
                "error_intf_flux": getattr(result, "error_intf_flux"),
            }
        )

    return errors


# ----> Set desired L2-errors
@pytest.fixture(scope="module")
def desired_l2_errors() -> list[dict[str, float]]:
    """Set desired L2-relative errors.

    Returns:
        Dictionary of desired relative L2-errors.

    """
    desired_error_0 = {  # t = 0.2 [s]
        "error_matrix_pressure": 0.05925007221301212,
        "error_matrix_flux": 0.017427251422474147,
        "error_frac_pressure": 4.639395967257667,
        "error_frac_flux": 0.002124677195582403,
        "error_intf_flux": 2.9141633825921764,
    }
    desired_error_1 = {  # t = 0.8 [s]
        "error_matrix_pressure": 0.05763683424696444,
        "error_matrix_flux": 0.017256638140139675,
        "error_frac_pressure": 4.749914708440596,
        "error_frac_flux": 0.003280737213089598,
        "error_intf_flux": 3.0896181780350087,
    }
    desired_error_2 = {  # t = 1.0 [s]
        "error_matrix_pressure": 0.056955711514823516,
        "error_matrix_flux": 0.01720817118615666,
        "error_frac_pressure": 4.726676447078315,
        "error_frac_flux": 0.003612830093695507,
        "error_intf_flux": 3.1029228202482684,
    }
    return [desired_error_0, desired_error_1, desired_error_2]


# ----> Now, we write the actual test
@pytest.mark.parametrize(
    "var",
    ["matrix_pressure", "matrix_flux", "frac_pressure", "frac_flux", "intf_flux"],
)
@pytest.mark.parametrize("time_idx", [0, 1, 2])
def test_relative_l2_errors_cartesian_grid(
        time_idx: int,
        var: str,
        actual_l2_errors: list[dict[str, float]],
        desired_l2_errors: list[dict[str, float]],
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
        from :meth:`˜material_constants` on a Cartesian grid with 512 cells. We test
        the errors for three different times, namely: 0.2 [s], 0.8 [s], and 1.0 [s].

    Parameters:
        time_idx: Time index acting on the lists 'actual_l2_errors' and
            'desired_l2_errors'.
        var: Name of the variable to be tested.
        actual_l2_errors: List of dictionaries containing the actual L2-relative errors.
        desired_l2_errors: List of dictionaries containing the desired L2-relative
            errors.

    """
    np.testing.assert_allclose(
        actual_l2_errors[time_idx]["error_" + var],
        desired_l2_errors[time_idx]["error_" + var],
        atol=1e-6,
        rtol=1e-5,
    )


# --> [TST-2] Observed order of convergence on Cartesian and simplices

# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(material_constants: dict) -> dict[str, float]:
    """Retrieve actual order of convergence.

    Note:
        This is a spatio-temporal analysis, where the spatial step size is decreased
        by a factor of `2` and the temporal step size is decreased by a factor of `4`
        between each run. This is because the MPFA is quadratically convergent,
        whereas Backward Euler is only linearly convergent.

    Parameters:
        material_constants: Dictionary containing the material constants.

    Returns:
        Dictionaries containing the actual observed order of convergence.

    """
    params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "mesh_arguments": {"cell_size": 0.125},
    }
    conv_analysis = ConvergenceAnalysis(
        model_class=ManuCompFlowSetup3d,
        model_params=params,
        levels=3,
        in_space=True,
        spatial_rate=2,
        in_time=True,
        temporal_rate=4,
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
        'ooc_frac_flux': 2.011343043274247,
        'ooc_frac_pressure': 1.985302288174025,
        'ooc_intf_flux': 1.9998583923263855,
        'ooc_matrix_flux': 1.6009304954707668,
        'ooc_matrix_pressure': 2.1529911615181723
    }


# ----> Now, we write the actual test
#@pytest.mark.skip(reason="slow")
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

    assert np.isclose(
        desired_ooc["ooc_" + var],
        actual_ooc["ooc_" + var],
        atol=1e-3,  # allow for an absolute difference of 0.001 in OOC
        rtol=1e-3,  # allow for 0.1% of relative difference in OOC
    )