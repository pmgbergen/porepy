"""Testing the isothermal linear tracer setups and some of the CF machinery."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.linear_tracer import (
    LinearTracerSaveData,
    SimplePipe2D,
    TracerFlowSetup_1p,
    TracerFlowSetup_3p,
)


@pytest.fixture(scope="module")
def results(request: pytest.FixtureRequest) -> list[LinearTracerSaveData]:
    """Results of the 1-phase, 2-component linear tracer model"""
    cell_size, model = request.param
    # Run verification setup and retrieve results for three different times
    material_constants = {
        "solid": pp.SolidConstants(porosity=1.0, permeability=1.0, residual_aperture=1),
    }
    time_manager = pp.TimeManager(
        schedule=[0, 10, 30, 80, 100], dt_init=10, constant_dt=True
    )
    model_params = {
        "material_constants": material_constants,
        "time_manager": time_manager,
        "meshing_arguments": {"cell_size": cell_size},
        "prepare_simulation": False,
        "times_to_export": [],
    }
    if model == "1p":
        setup = TracerFlowSetup_1p(model_params)
    elif model == "3p":
        setup = TracerFlowSetup_3p(model_params)
        # To create phase fractions as variables and have a representation fo h_mix
        setup.params["equilibrium_type"] = "dummy"
    else:
        raise ValueError(f"Unknown model fixture parametrization {model}.")
    setup.prepare_simulation()

    # Setting dt and end time schedule according to cfl condition and approximate
    # flow velocity. Works only assuming the test does not work with I/O of times.
    sd = setup.mdg.subdomains()[0]
    dt = setup.exact_sol.dt_from_cfl(sd)

    time_manager = pp.TimeManager(
        schedule=[0, 3 * dt, 6 * dt, 9 * dt, 10 * dt],
        dt_init=dt,
        constant_dt=True,
    )
    setup.ad_time_step.set_value(dt)
    setup.time_manager = time_manager
    pp.run_time_dependent_model(setup, model_params)
    return setup.results


# First parametrization is over number of cells in pipe and setup for fixture.
# Second to test all scheduled indices.
@pytest.mark.parametrize(
    "results", [(SimplePipe2D.pipe_length / 40, "1p")], indirect=["results"]
)
@pytest.mark.parametrize("time_index", [0, 1, 2, 3])
def test_linear_tracer_1p_diffusive(
    time_index: int, results: list[LinearTracerSaveData]
) -> None:
    """Testing the simulation results for the linear tracer with 1-phase.

    Checking that the L2-errors for pressure are small, and that the maximum
    number of iterations does not exceed 2.

    Checks compares the tracer fraction with an analytical solution accounting
    for the numerical diffusion of the schemes used (Upwinding & backward Euler).

    """

    sol_data = results[time_index]

    # After the first time step, more iterations are possible because the pressure
    # must converge to its stationary profile
    if time_index == 0:
        assert sol_data.num_iter <= 5
    # After pressure converged, linear transport should converge within 1 iteration.
    # But due to Upwinding, it is sometimes 2
    else:
        assert sol_data.num_iter <= 2

    # testing errors in pressure (exact)
    np.testing.assert_allclose(sol_data.error_p, 0.0, atol=1e-7, rtol=0.0)

    # NOTE due to the hyperbolic nature, the error in the tracer fraction should
    # converge to zero linearly, which is checked in a separate test. Here we check only
    # that the error is not getting worse with ongoing code development.
    expected_error_z = [
        0.1128851432293962,
        0.1344878688028854,
        0.1489878242519587,
        0.15299862798949995,
    ]
    expected_error_diffused_z = [
        0.020278908152713,
        0.018535428299520068,
        0.01703231859562048,
        0.016602357208773923,
    ]
    # Allow 10% margin for machine reasons.
    assert (
        sol_data.error_diffused_z_tracer <= 1.1 * expected_error_diffused_z[time_index]
    )
    assert sol_data.error_z_tracer <= 1.1 * expected_error_z[time_index]


# Expected to fail because Upwinding and backward euler are diffusive
@pytest.mark.xfail
@pytest.mark.parametrize(
    "results", [(SimplePipe2D.pipe_length / 40, "1p")], indirect=["results"]
)
@pytest.mark.parametrize("time_index", [0, 1, 2, 3])
def test_linear_tracer_1p_exact(
    time_index: int, results: list[LinearTracerSaveData]
) -> None:
    """Compares the tracer fraction with the exact solution with a steep front.

    This test is expected to fail since the numerical algorithm used is diffusive.

    """
    sol_data = results[time_index]
    np.testing.assert_allclose(sol_data.error_z_tracer, 0.0, atol=1e-7, rtol=0.0)

@pytest.mark.skipped
def test_linear_tracer_1p_ooc():
    """Tests the order of convergence for the tracer fraction, which is expected to be
    quadratic."""
    max_iterations = 80
    newton_tol = 1e-6
    newton_tol_increment = newton_tol

    # Breakthrough time is roughly 100 seconds in this setup. We take the half time
    # where the front is roughly in the middle and both sides of the modified solution
    # which includes diffusiones are shown in the solution
    time_manager = pp.TimeManager([0, 50], 10, True)

    params = {
        "material_constants": {
            "solid": pp.SolidConstants(
                porosity=1, permeability=1, residual_aperture=1.0
            ),
        },
        "time_manager": time_manager,
        "max_iterations": max_iterations,
        "nl_convergence_tol": newton_tol_increment,
        "nl_convergence_tol_res": newton_tol,
        "times_to_export": [],
        "grid_type": "cartesian",
        "meshing_arguments": {"cell_size": 1.0},
    }

    conv_analysis = ConvergenceAnalysis(
        model_class=TracerFlowSetup_1p,
        model_params=deepcopy(params),
        levels=5,
        # Constant flow velocity means halving of grid size should be followed by
        # halving of time step size (CFL).
        spatial_refinement_rate=2,
        temporal_refinement_rate=2,
    )
    results = conv_analysis.run_analysis()
    ooc = conv_analysis.order_of_convergence(results)

    # Both should converge roughly linear when compared to the exact front and the
    # solution to the modified equation (which includes diffusion). The latter should
    # should converge super-linearly, while the former roughly linearly.
    # These values are snapshots of the state when the tests are written. AS the code
    # improves, they should be updated.
    expected_ooc = {
        "ooc_z_tracer": 0.7474414018761917,
        "ooc_diffused_z_tracer": 1.233078730659626,
    }
    # NOTE checking OOC for pressure makes no sense since it is at machine presicion.
    # Problem is linear and incompressible, which means p converges immediately.

    np.testing.assert_allclose(
        ooc["ooc_z_tracer"],
        expected_ooc["ooc_z_tracer"],
        atol=1e-1,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        ooc["ooc_diffused_z_tracer"],
        expected_ooc["ooc_diffused_z_tracer"],
        atol=1e-1,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "results",
    [
        (SimplePipe2D.pipe_length / 10, "3p"),
        pytest.param((SimplePipe2D.pipe_length / 100, "3p"), marks=pytest.mark.skipped),
    ],  # reason: slow
    indirect=["results"],
)
@pytest.mark.parametrize("time_index", [0, 1, 2, 3])
def test_linear_tracer_3p(time_index: int, results: list[LinearTracerSaveData]) -> None:
    """Tests the 3-phase model and whether it obtains the expected solution.

    Note:
        This is also an integration test for SurrogateFactory and LocalElimination
    """

    sol_data = results[time_index]

    # After the first time step, more iterations are possible because the pressure
    # must converge to its stationary profile
    if time_index == 0:
        assert sol_data.num_iter <= 5
    # After pressure converged, linear transport should converge within 1 iteration.
    # But due to Upwinding, it is sometimes 2
    else:
        assert sol_data.num_iter <= 2

    np.testing.assert_allclose(sol_data.error_p, 0.0, atol=1e-7, rtol=0.0)

    # It takes some time for the model to reach isothermal condition once the pressure
    # profile changes from constant to linear. This is due to the fluid internal
    # energy consiting of enthalpy and pressure work.
    if time_index > 1:
        np.testing.assert_allclose(sol_data.error_T, 0.0, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(sol_data.error_h, 0.0, atol=1e-7, rtol=0.0)
    else:
        np.testing.assert_allclose(sol_data.error_T, 0.0, atol=1e-1, rtol=0.0)
        np.testing.assert_allclose(sol_data.error_h, 0.0, atol=1e-1, rtol=0.0)

    np.testing.assert_allclose(
        sol_data.errors_phase_fractions, 0.0, atol=1e-7, rtol=0.0
    )
    np.testing.assert_allclose(sol_data.errors_saturations, 0.0, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        sol_data.errors_partial_fractions, 0.0, atol=1e-7, rtol=0.0
    )
