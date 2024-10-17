"""
This module contains functional tests for approximations to the set of equations
modeling the 2d and 3d flow in non-isothermal deformable porous media.

Tests:

    [TEST_1] Relative L2-error on Cartesian grids for primary and secondary variables
      for three different times for 2d and 3d.

    [TEST_2] Observed order of convergence (using four levels of refinement for 2d and
      three levels of refinement for 3d) for primary and secondary variables. Order of
      convergence using Cartesian grids are tested for 2d and 3d, whereas simplicial
      grids are limited to 2d.

References:

    [1] Coussy, O. (2004). Poromechanics. John Wiley & Sons. ISO 690

    [2] Stefansson, I., Varela, J., Keilegavlen, E. & Berre, I. (2024). Flexible and
    rigorous numerical modelling of multiphysics processes in fractured porous media
    using PorePy. Results in applied Mathematics. 21, 100448.

"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.manu_thermoporomech_nofrac_2d import (
    ManuThermoPoroMechSetup2d,
)
from tests.functional.setups.manu_thermoporomech_nofrac_3d import (
    ManuThermoPoroMechSetup3d,
)


# --> Declaration of module-wide fixtures that are re-used throughout the tests
@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Set material constants.

    Use default values provided in the module where the setup class is included.

    Returns:
        Dictionary containing the material constants with the `solid` and `fluid`
        constant classes.

    """
    fluid_constants = pp.FluidConstants({"compressibility": 0.02})
    solid_constants = pp.SolidConstants({"biot_coefficient": 0.5})
    return {"solid": solid_constants, "fluid": fluid_constants}


# --> [TEST_1] Relative L2-errors on Cartesian grid


# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants) -> list[list[dict[str, float]]]:
    """Run verification setups and retrieve results for the scheduled times.

    Parameters:
        material_constants: Dictionary containing the material constant classes.

    Returns:
        List of lists of dictionaries of actual relative errors. The outer list contains
        two items, the first contains the results for 2d and the second contains the
        results for 3d. Both inner lists contain three items each, each of which is a
        dictionary of results for the scheduled times, i.e., 0.5 [s] and 1.0 [s].

    """

    # Define model parameters (same for 2d and 3d).
    model_params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "time_manager": pp.TimeManager([0, 0.5, 1.0], 0.5, True),
        "heterogeneity": 10.0,
    }

    # Retrieve actual L2-relative errors.
    errors: list[list[dict[str, float]]] = []
    # Loop through models, i.e., 2d and 3d.
    for model in [ManuThermoPoroMechSetup2d, ManuThermoPoroMechSetup3d]:
        setup = model(deepcopy(model_params))  # Make deep copy of params to avoid nasty bugs.
        pp.run_time_dependent_model(setup, {})
        errors_setup: list[dict[str, float]] = []
        # Loop through results, i.e., results for each scheduled time.
        for result in setup.results:
            errors_setup.append(
                {
                    "error_pressure": getattr(result, "error_pressure"),
                    "error_darcy_flux": getattr(result, "error_darcy_flux"),
                    "error_displacement": getattr(result, "error_displacement"),
                    "error_force": getattr(result, "error_force"),
                    "error_temperature": getattr(result, "error_temperature"),
                    "error_energy_flux": getattr(result, "error_energy_flux"),
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
    desired_errors_2d = [
        {  # t = 0.5 [s]
            "error_pressure": 0.29827252435393803,
            "error_darcy_flux": 0.1439958241169773,
            "error_displacement": 0.4058988627472109,
            "error_force": 0.16888424201570365,
            "error_temperature": 0.28326347851869543,
            "error_energy_flux": 0.143777694649805,
        },
        {  # t = 1.0 [s]
            "error_pressure": 0.3010187539654323,
            "error_darcy_flux": 0.1433298407071359,
            "error_displacement": 0.4058374630516512,
            "error_force": 0.16890514221989356,
            "error_temperature": 0.27340859892936803,
            "error_energy_flux": 0.14567435981074586,
        },
    ]
    desired_errors_3d = [
        {  # t = 0.5 [s]
            "error_pressure": 0.28233484316067864,
            "error_darcy_flux": 0.15250403155748207,
            "error_displacement": 0.4482633717525385,
            "error_force": 0.21455044408871599,
            "error_temperature": 0.2668424103750087,
            "error_energy_flux": 0.14699614090181903,
        },
        {  # t = 1.0 [s]
            "error_pressure": 0.28113090124404544,
            "error_darcy_flux": 0.1487747552893976,
            "error_displacement": 0.448144662794714,
            "error_force": 0.21457738933054352,
            "error_temperature": 0.2593080002077755,
            "error_energy_flux": 0.14847494492738805,
        },
    ]

    return [desired_errors_2d, desired_errors_3d]


@pytest.mark.parametrize("dim_idx", [0, 1])
@pytest.mark.parametrize(
    "var",
    ["pressure", "darcy_flux", "displacement", "force", "temperature", "energy_flux"],
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

        For this functional test, we are comparing errors for the pressure, fluxes,
        displacement, and forces. The errors are measured using the discrete relative
        L2-error norm. The desired errors were obtained by running the model using the
        physical constants from :meth:`~material_constants` on a Cartesian grid with 16
        cells in 2d and 64 in 3d. We test the errors for two different times, namely:
        0.5 [s],  and 1.0 [s].

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
        whereas Backward Euler is linearly convergent.

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
    # Loop through the models.
    for model_idx, model in enumerate(
        [ManuThermoPoroMechSetup2d, ManuThermoPoroMechSetup3d]
    ):
        ooc_setup: list[dict[str, float]] = []
        # Loop through grid type.
        for grid_type in ["cartesian", "simplex"]:
            # We do not perform a convergence analysis with simplices in 3d.
            if model_idx == 1 and grid_type == "simplex":
                continue
            else:
                # Use same parameters for both 2d and 3d.
                params = {
                    "grid_type": grid_type,
                    "material_constants": material_constants,
                    "meshing_arguments": {"cell_size": 0.25},
                    "perturbation": 0.3,
                    "heterogeneity": 10.0,
                }
                # Use 4 levels of refinement for 2d and 3 levels for 3d.
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
            "ooc_displacement": 2.056709801235223,
            "ooc_darcy_flux": 1.7001969755444224,
            "ooc_energy_flux": 1.737188125750453,
            "ooc_force": 1.4425687190971186,
            "ooc_pressure": 2.1102255203961358,
            "ooc_temperature": 2.310453793430966,
        },
        {  # simplex
            "ooc_displacement": 2.0903886074133236,
            "ooc_darcy_flux": 1.5662133051661693,
            "ooc_energy_flux": 1.5864494813545396,
            "ooc_force": 1.4820831222977098,
            "ooc_pressure": 1.9846518952465777,
            "ooc_temperature": 2.1924242696392753,
        },
    ]

    desired_ooc_3d = [
        {  # Cartesian
            "ooc_displacement": 2.0401748849429717,
            "ooc_darcy_flux": 1.2226767170350672,
            "ooc_energy_flux": 1.3090025057692818,
            "ooc_force": 1.2472669444997415,
            "ooc_pressure": 2.038904548181216,
            "ooc_temperature": 2.2095880011696294,
        }
    ]

    return [desired_ooc_2d, desired_ooc_3d]


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize(
    "var",
    ["pressure", "darcy_flux", "displacement", "force", "temperature", "energy_flux"],
)
@pytest.mark.parametrize("grid_type_idx", [0])
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
