"""Testing the isothermal linear tracer setups and some of the CF machinery."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.linear_tracer import (
    LinearTracerSaveData,
    SimplePipe2D,
    TracerFlowSetup_1p,
    TracerFlowSetup_1p_ff,
    TracerFlowSetup_3p,
)


@pytest.fixture(scope="module")
def results(request: pytest.FixtureRequest) -> list[LinearTracerSaveData]:
    """Results of the 1-phase, 2-component linear tracer model"""
    cell_size, model = cast(tuple[float, type[pp.PorePyModel]], request.param)
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

    setup = model(model_params)
    if isinstance(setup, TracerFlowSetup_3p):
        # To create phase fractions as variables and have a representation fo h_mix
        setup.params["equilibrium_type"] = "dummy"

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
    "results",
    [
        (SimplePipe2D.pipe_length / 40, TracerFlowSetup_1p),
        (SimplePipe2D.pipe_length / 40, TracerFlowSetup_1p_ff),
    ],
    indirect=["results"],
)
@pytest.mark.parametrize("time_index", [0, 1, 2, 3])
def test_linear_tracer_1p(time_index: int, results: list[LinearTracerSaveData]) -> None:
    """Testing the simulation results for the linear tracer with 1-phase.

    Checking that the L2-errors for pressure are small, and that the maximum
    number of iterations does not exceed 2.

    Checks compares the tracer fraction with an analytical solution accounting
    for the numerical diffusion of the schemes used (Upwinding & backward Euler).

    """

    sol_data = results[time_index]

    # After the first time step, more iterations are possible because the pressure
    # must converge to its stationary profile.
    if time_index == 0:
        assert sol_data.num_iter <= 5
    # After pressure converged, linear transport should converge within 1 iteration.
    # But due to Upwinding, it is sometimes 2.
    else:
        assert sol_data.num_iter <= 2

    # Testing errors in pressure (exact).
    np.testing.assert_allclose(sol_data.error_p, 0.0, atol=1e-7, rtol=0.0)

    # NOTE: Due to the hyperbolic nature, the error in the tracer fraction should
    # converge to zero linearly, which is checked in a separate test. Here we check only
    # that the error is not getting worse with ongoing code development.
    expected_error_z = [
        0.0703124998509167,
        0.10151366980995292,
        0.1251925522284108,
        0.13214752531256774,
    ]
    expected_error_diffused_z = [
        0.014075708718460218,
        0.016326064144512435,
        0.017363373976532474,
        0.017615673816732902,
    ]
    # Allow 10% margin for machine reasons.
    assert (
        sol_data.error_diffused_z_tracer <= 1.1 * expected_error_diffused_z[time_index]
    )
    assert sol_data.error_z_tracer <= 1.1 * expected_error_z[time_index]


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("model_class", [TracerFlowSetup_1p, TracerFlowSetup_1p_ff])
def test_linear_tracer_1p_ooc(
    model_class: type[TracerFlowSetup_1p] | type[TracerFlowSetup_1p_ff],
) -> None:
    """Tests the order of convergence for the tracer fraction, which is expected to be
    slightly super-linear in the L1 norm, and roughly linear in the L2 norm."""
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
        model_class=model_class,
        model_params=deepcopy(params),
        levels=3,
        # Constant flow velocity means halving of grid size should be followed by
        # halving of time step size (CFL) (though not necessary for implicit scheme)
        spatial_refinement_rate=2,
        temporal_refinement_rate=2,
    )
    results = conv_analysis.run_analysis()
    ooc = conv_analysis.order_of_convergence(results)

    # Expected convergence rate towards exact solution is linear using the L1 norm.
    # For some reasons we are super-linear.
    # The convergence rate towards the modified solution which includes diffusion is
    # almost quadratic. TODO investigate.
    # Below values are snapshots from 05.02.2025 for 3 levels of refinement.
    # Fir higher levels the values tend towards 1.5 and 2 respectively.
    expected_ooc = {
        # 7 levels, which takes too much time for tests.
        # "ooc_z_tracer": 1.482169632646451,
        # "ooc_diffused_z_tracer": 1.9254070734727107,
        # 3 levels
        "ooc_z_tracer": 1.4388862433052907,
        "ooc_diffused_z_tracer": 1.733463542010078,
    }
    # NOTE checking OOC for pressure makes no sense since error is at machine presicion.
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
        (SimplePipe2D.pipe_length / 10, TracerFlowSetup_3p),
        pytest.param(
            (SimplePipe2D.pipe_length / 100, TracerFlowSetup_3p),
            marks=pytest.mark.skipped,  # reason: slow
        ),
    ],
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
