"""Testing the isothermal linear tracer setups and some of the CF machinery."""
from __future__ import annotations

import pytest

import numpy as np
import porepy as pp

from tests.functional.setups.linear_tracer import (
    LinearTracerSaveData,
    TracerFlowSetup_1p,
    TracerFlowSetup_3p,
)

@pytest.fixture(scope="module")
def results_1p(request: pytest.FixtureRequest) -> list[LinearTracerSaveData]:
    """Results of the 1-phase, 2-component linear tracer model"""
    # Run verification setup and retrieve results for three different times
    material_constants = {
        "solid": pp.SolidConstants(porosity=1., permeability=1., residual_aperture=1),
    }
    time_manager = pp.TimeManager([0, 10, 30, 80, 100], 10, True)
    model_params = {
        "material_constants": material_constants,
        "time_manager": time_manager,
        'num_cells': request.param,
        'prepare_simulation': False,
    }
    setup = TracerFlowSetup_1p(model_params)
    setup.prepare_simulation()

    # modifying dt and end time schedule according to cfl condition and approximate
    # flow velocity. Works only assuming the test does not work with I/O of times
    sd = setup.mdg.subdomains()[0]
    dt = setup.exact_sol.dt_from_cfl(sd)
    v = setup.exact_sol.flow_velocity(sd)
    setup.time_manager.dt_init = dt
    setup.time_manager.dt = dt
    T_final = setup.pipe_length / v
    if setup.time_manager.schedule[-1] <= T_final:
        setup.time_manager.schedule = np.array(
            setup.time_manager.schedule.tolist() + [T_final]
        )
        setup.time_manager.time_final = T_final
    pp.run_time_dependent_model(setup, model_params)
    return setup.results

# First parametrization is over number of cells in pipe for fixture.
# Second to test all schedule indices.
@pytest.mark.parametrize('results_1p', [10], indirect=['results_1p'])
@pytest.mark.parametrize('time_index', [0, 1, 2, 3])
def test_linear_tracer_1p_diffusive(
    time_index: int,
    results_1p: list[LinearTracerSaveData]
) -> None:
    """Testing the simulation results for the linear tracer with 1-phase.

    Checking that the L2-errors for pressure are small, and that the maximum
    number of iterations does not exceed 2.

    Checks compares the tracer fraction with an pseudo-analytical solution accounting
    for the numerical diffusion of the schemes used (Upwinding & backward Euler). TODO

    """

    sol_data = results_1p[time_index]

    np.testing.assert_allclose(sol_data.error_p, 0., atol=1e-7, rtol=0.)

    # After the first time step, more iterations are possible because the pressure
    # must converge to its stationary profile
    if time_index == 0:
        assert sol_data.num_iter <= 5
    # After pressure converged, linear transport should converge within 1 iteration.
    # But due to Upwinding, it is sometimes 2
    else:
        assert sol_data.num_iter <= 2


# Expected to fail because Upwinding and backward euler are diffusive
@pytest.mark.xfail
@pytest.mark.parametrize('results_1p', [10], indirect=['results_1p'])
@pytest.mark.parametrize('time_index', [0, 1, 2, 3])
def test_linear_tracer_1p_exact(
    time_index: int,
    results_1p: list[LinearTracerSaveData]
) -> None:
    """Compares the tracer fraction with the exact solution with a steep front.
    
    This test is expected to fail since the numerical algorithm used is diffusive.

    """
    sol_data = results_1p[time_index]
    np.testing.assert_allclose(sol_data.error_z_tracer, 0., atol=1e-7, rtol=0.)
