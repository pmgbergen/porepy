"""Tests of the initialization of a model to a steady state.

We test:
    - The pipeline: finding, saving, loading and discarding the steady state, and that
      the model is restored after initialization, also if it fails.
    - That time dependent boundary values are frozen at the time of the first time step,
      and unfrozen after initialization.
    - That the reference state is consistent with the model after initialization.

"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.initialization.initialization import (
    InitializationError,
    _InitializationContext,
    _validate_steady_state,
    default_residual_tolerances,
    initialization_pipeline,
)
from porepy.applications.test_utils import models

PRESSURE_OFFSET = 1e3
"""Number used for the pressure term magnitude in the tests."""
SOURCE_TERM_OFFSET = 1e-3
"""Number used for the source term magnitude in the tests."""


class RampedBoundary(pp.PorePyModel):
    """Nonzero boundary values which grow linearly with time.

    At time t, pressure and temperature on the boundary are
    (1 + t / day) * PRESSURE_OFFSET and 1, respectively.

    """

    def ramp(self) -> float:
        """Growth factor of the boundary values at the current time."""
        return 1 + self.time_manager.time / pp.DAY

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Pressure on the boundary, growing with time."""
        return np.full(bg.num_cells, PRESSURE_OFFSET * self.ramp())

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Temperature on the boundary, growing with time."""
        return np.full(bg.num_cells, self.ramp())

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Displacement on the boundary, growing with time."""
        values = np.zeros((self.nd, bg.num_cells))
        values[0] = 1e-4 * self.ramp()
        return values.ravel("F")


class RampedSource(pp.PorePyModel):
    """Fluid source of SOURCE_TERM_OFFSET * (1 + t / day) in each cell.

    The source is updated with the time dependent data of the model.

    """

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Add the ramped source to the fluid source."""
        return super().fluid_source(subdomains) + pp.ad.TimeDependentDenseArray(
            "ramped_source", subdomains
        )

    def update_time_dependent_ad_arrays(self) -> None:
        """Update the ramped source with the current time."""
        super().update_time_dependent_ad_arrays()
        for sd, data in self.mdg.subdomains(return_data=True):
            values = np.full(
                sd.num_cells, SOURCE_TERM_OFFSET * (1 + self.time_manager.time / pp.DAY)
            )
            pp.set_solution_values("ramped_source", values, data, iterate_index=0)


TpsaPoromechanics = models.add_mixin(
    pp.poromechanics.TpsaPoromechanicsMixin, models.Poromechanics
)

ZERO = defaultdict(lambda: 0.0)
"""Tolerances which are never satisfied."""


def time_manager(constant_dt: bool = False) -> pp.TimeManager:
    """Create a time manager of five days with time steps of one day.

    Parameters:
        constant_dt: Whether the time step is constant.

    Returns:
        The time manager.

    """
    if constant_dt:
        return pp.TimeManager(
            schedule=[0, 5 * pp.DAY], dt_init=pp.DAY, constant_dt=True
        )
    return pp.TimeManager(
        pp.time_stepper.Schedule(
            intervals=[
                pp.time_stepper.TimeInterval.create(
                    t_start=0,
                    dt_start=pp.DAY,
                    constraints=[pp.time_stepper.TargetNonlinearIterations()],
                    name="days",
                )
            ],
            t_end=5 * pp.DAY,
        )
    )


def make_model(folder: Path, model_class=models.Thermoporomechanics, **params):
    """Create a model with a fracture, time dependent data and time steps of one day.

    Parameters:
        folder: Output folder of the model.
        model_class: Class of the model, extended by the time dependent data.
        **params: Parameters of the model, overriding the defaults.

    Returns:
        The model, not prepared.

    """
    model_class = models.add_mixin(RampedSource, model_class)
    return models.add_mixin(RampedBoundary, model_class)(
        {
            "fracture_indices": [0],
            "grid_type": "cartesian",
            "times_to_export": [],
            "folder_name": str(folder),
            "time_manager": pp.TimeManager(
                schedule=[0, 3 * pp.DAY], dt_init=pp.DAY, constant_dt=True
            ),
            **params,
        }
    )


def initialize(model, rate_tolerances=None, residual_tolerances=None, **kwargs):
    """Initialize a model with tolerances that are reached within a time step.

    Parameters:
        model: The model, not prepared.
        rate_tolerances: Tolerances of the rate of change. Defaults to 1e-3.
        residual_tolerances: Tolerances of the residual. Defaults to the library ones.
        **kwargs: Keyword arguments of :func:`initialization_pipeline`, overriding the
            defaults of this function.

    Returns:
        The initialized model.

    """
    return initialization_pipeline(
        model,
        rate_tolerances=defaultdict(lambda: 1e-3)
        if rate_tolerances is None
        else rate_tolerances,
        residual_tolerances=default_residual_tolerances()
        if residual_tolerances is None
        else residual_tolerances,
        **{"time_manager": time_manager(), "earliest_stop_time": 0, **kwargs},
    )


def assert_restored(model, original_class, original_time_manager, folder: Path):
    """Assert that the class, time manager, clock and folder of the model are original.

    Also assert that the porosity is not constant, i.e., the original constitutive laws
    are applied.

    Parameters:
        model: The model after initialization.
        original_class: Class of the model before initialization.
        original_time_manager: Time manager of the model before initialization.
        folder: Output folder of the model before initialization.

    """
    assert model.__class__ is original_class
    assert model.time_manager is original_time_manager
    assert model.time_manager.time == 0
    assert model.params["folder_name"] == str(folder)
    assert model.exporter._folder_name == folder

    # Porosity must not be constant: evaluate it in the current and a perturbed state.
    eq_sys = model.equation_system
    porosity = model.porosity(model.mdg.subdomains())
    state = eq_sys.get_variable_values(iterate_index=0)
    porosity_original = eq_sys.evaluate(porosity, state=state)
    rng = np.random.default_rng(0)
    porosity_perturbed = porosity.value(eq_sys, state=state + rng.random(state.shape))
    assert not np.allclose(porosity_original, porosity_perturbed)


@pytest.mark.parametrize(
    "saved_state, logged, not_logged",
    [
        (
            "none",
            ["Saved state to", "Initialization is complete"],
            ["Trying to load", "discarded"],
        ),
        (
            "valid",
            ["Trying to load", "Initialization is complete"],
            ["Saved state to", "discarded"],
        ),
        (
            "wrong_mesh",
            [
                "Saved state is inconsistent",
                "It is discarded",
                "Saved state to",
                "Initialization is complete",
            ],
            [],
        ),
        (
            "not_steady",
            ["Not a steady state", "Saved state to", "Initialization is complete"],
            [],
        ),
    ],
)
def test_pipeline(tmp_path, caplog, saved_state, logged, not_logged):
    """This test instantiates a PorePy model and runs it through the initialization
    pipeline in various conditions (see parametrization), which all ultimately lead to
    a successfully initialized model.

    The test checks that:
    - the pipeline reported certain steps through logging;
    - the original model is restored;
    - the steady state is achieved.

    """
    caplog.set_level(logging.INFO)
    folder = tmp_path / "simulation"
    if saved_state != "none":
        # Prepare pre-saved steady state.

        fractures = {}
        if saved_state == "wrong_mesh":
            # Different grids for the wrong_mesh test case.
            fractures = {"fracture_indices": []}

        initialize(make_model(folder, **fractures))

        if saved_state == "not_steady":
            # Perturb the pressure to make the state non-steady. A uniform shift would
            # not change the residual.
            path = folder / "steady_state.npz"
            arrays = dict(np.load(path))
            for key in arrays:
                if "::pressure::" in key and "reference_solutions" not in key:
                    rng = np.random.default_rng(0)
                    arrays[key] = arrays[key] + PRESSURE_OFFSET * rng.random(
                        arrays[key].shape
                    )
            np.savez(path, **arrays)
        caplog.clear()

    model = make_model(folder)
    original_class, original_time_manager = model.__class__, model.time_manager
    initialize(model)

    # Check that the pipeline logged and not logged certain messages.
    for message in logged:
        assert message in caplog.text
    for message in not_logged:
        assert message not in caplog.text
    # Check that the original model is restored.
    assert_restored(model, original_class, original_time_manager, folder)
    # Check that the original model agrees this is a steady state.
    assert _validate_steady_state(model, default_residual_tolerances())


@pytest.mark.parametrize("failure", ["solver", "never_steady", "saved_state_is_bad"])
def test_pipeline_failure(tmp_path, failure):
    """If no steady state is found, an error is raised and the model is restored.

    An invalid saved state is discarded.

    """
    folder = tmp_path / "simulation"
    if failure == "saved_state_is_bad":
        initialize(make_model(folder))
    model = make_model(folder)
    original_class, original_time_manager = model.__class__, model.time_manager
    kwargs = {}
    if failure == "solver":
        # A constant time step cannot be decreased, so the time stepping fails at once.
        kwargs = {
            "time_manager": time_manager(constant_dt=True),
            "nonlinear_solver": pp.solvers.NewtonSolver(
                params={
                    "nl_max_iterations": 2,
                    "nl_convergence_inc_atol": 0,
                    "nl_convergence_res_atol": 0,
                }
            ),
        }
    else:
        kwargs = {"rate_tolerances": ZERO, "residual_tolerances": ZERO}

    with pytest.raises(InitializationError):
        initialize(model, **kwargs)

    assert_restored(model, original_class, original_time_manager, folder)
    assert not (folder / "steady_state.npz").exists()


def test_data_time_frozen(tmp_path):
    """Test that the found reference state corresponds to the frozen time of the first
    time step."""
    model = make_model(tmp_path / "simulation")
    initialize(model)

    # The boundary values, also as reference values, are those of the first time step.
    expected_time = pp.DAY  # First dt = pp.DAY, as defined in the time manager.
    checked = 0
    for _, data in model.mdg.boundaries(return_data=True):
        if model.pressure_variable in data.get(pp.ITERATE_SOLUTIONS, {}):
            for kwargs in [{"reference": True}, {"iterate_index": 0}]:
                pressure = pp.get_solution_values(
                    model.pressure_variable, data, **kwargs
                )
                np.testing.assert_allclose(
                    pressure, PRESSURE_OFFSET * (1 + expected_time / pp.DAY)
                )
                checked += 1
    # Without this the test passes if no boundary grid carries the pressure.
    assert checked > 0, "no boundary values of the pressure were checked"


@pytest.mark.parametrize(
    "hook", ["after_nonlinear_convergence", "after_nonlinear_failure"]
)
def test_time_dependent_data_in_initialization_context(tmp_path, hook):
    """Inside the initialization context, the time dependent data is that of frozen
    time. Outside - the original time is restored, and the data corresponds to the
    original time.

    """
    model = make_model(tmp_path / "simulation")
    # The data time is the first target time of the simulation.
    ramp = lambda time: 1 + time / pp.DAY  # type: ignore
    expected_ramp = ramp(pp.DAY)
    clock = 2 * pp.DAY

    def values_of_time_step() -> tuple[float, np.ndarray, np.ndarray]:
        """Emulate the time stepper, which sets the clock before the time step."""
        model.time_manager.time = clock
        model.before_time_step()
        source = model.equation_system.evaluate(
            model.fluid_source(model.mdg.subdomains())
        )
        pressure = [
            pp.get_solution_values(model.pressure_variable, data, iterate_index=0)
            for _, data in model.mdg.boundaries(return_data=True)
            if model.pressure_variable in data.get(pp.ITERATE_SOLUTIONS, {})
        ]
        assert len(pressure) > 0
        return model.time_manager.time, source, np.concatenate(pressure)

    with _InitializationContext(model, time_manager()):
        model.prepare_simulation()
        clock_in_step, source, pressure = values_of_time_step()
        assert clock_in_step == pp.DAY * (expected_ramp - 1)
        np.testing.assert_allclose(source, SOURCE_TERM_OFFSET * expected_ramp)
        np.testing.assert_allclose(pressure, PRESSURE_OFFSET * expected_ramp)
        # Call either after_nonlinear_convergence or *_failure, each should restore time
        getattr(model, hook)()
        assert model.time_manager.time == clock

    # Without the freeze, the values are those of the clock.
    clock_in_step, source, pressure = values_of_time_step()
    assert clock_in_step == clock
    np.testing.assert_allclose(source, SOURCE_TERM_OFFSET * ramp(clock))
    np.testing.assert_allclose(pressure, PRESSURE_OFFSET * ramp(clock))


@pytest.mark.parametrize(
    "model_class, perturbations",
    [
        (models.Thermoporomechanics, ["mechanical_stress", "pressure_stress"]),
        (TpsaPoromechanics, ["mechanical_stress"]),
    ],
)
def test_reference_state(tmp_path, model_class, perturbations):
    """After initialization, total stress equals reference stress, porosity equals
    reference porosity and shear dilation is zero.

    """
    model = make_model(tmp_path / "simulation", model_class)
    initialize(model)

    sd = model.mdg.subdomains(dim=model.nd)
    operators = {name: getattr(model, name)(sd) for name in perturbations}
    operators["stress"] = model.stress(sd) - model.reference_stress(sd)
    operators["porosity"] = model.porosity(sd) - model.reference_porosity(sd)
    operators["shear_dilation"] = model.shear_dilation_gap(
        model.mdg.subdomains(dim=model.nd - 1)
    )
    for name, operator in operators.items():
        values = model.equation_system.evaluate(operator)
        np.testing.assert_allclose(values, 0, atol=1e-10, err_msg=name)
