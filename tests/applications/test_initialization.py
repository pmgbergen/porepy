"""Tests of the initialization of a model to a steady state.

We test:
    - The pipeline: finding, saving, loading and discarding the steady state, and that
      the model is restored after initialization, also if it fails.
    - That time dependent boundary values are frozen at the time of the first time step.
    - That the reference state is consistent with the model after initialization.
    - The storage of the state, and the criterion to stop the time stepping.

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
    _validate_steady_state,
    default_residual_tolerances,
    initialization_pipeline,
)
from porepy.applications.initialization.model_state_storage import (
    ModelStateStorage,
    SteadyStateMismatch,
)
from porepy.applications.initialization.steady_state_early_stop import (
    SteadyStateEarlyStopCriterion,
    SteadyStateModelRunnerSuccess,
)
from porepy.applications.test_utils import models


class RampedBoundary(pp.PorePyModel):
    """Nonzero boundary values which grow linearly with time. At time t, pressure and
    temperature on the boundary are (1 + t / day) * 1e3 and 1, respectively.

    """

    def ramp(self) -> float:
        return 1 + self.time_manager.time / pp.DAY

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, 1e3 * self.ramp())

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.full(bg.num_cells, self.ramp())

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        values[0] = 1e-4 * self.ramp()
        return values.ravel("F")


TpsaPoromechanics = models.add_mixin(
    pp.poromechanics.TpsaPoromechanicsMixin, models.Poromechanics
)

INFINITE = defaultdict(lambda: np.inf)
"""Tolerances which are always satisfied."""
ZERO = defaultdict(lambda: 0.0)
"""Tolerances which are never satisfied."""


def time_manager(constant_dt: bool = False) -> pp.TimeManager:
    """Time manager of five days with time steps of one day."""
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
    """Model with a fracture, nonzero boundary values and time steps of one day."""
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
    """Initialization with tolerances that are reached within a time step."""
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
    assert model.__class__ is original_class
    assert model.time_manager is original_time_manager
    assert model.time_manager.time == 0
    assert model.params["folder_name"] == str(folder)


@pytest.mark.parametrize(
    "saved_state, logged, not_logged",
    [
        ("none", ["Saved steady state"], ["Trying to load"]),
        ("valid", ["Trying to load"], ["Saved steady state", "discarded"]),
        ("wrong_mesh", ["inconsistent", "Saved steady state"], []),
        ("not_steady", ["Not a steady state", "Saved steady state"], []),
    ],
)
def test_pipeline(tmp_path, caplog, saved_state, logged, not_logged):
    """A saved state is used if it is consistent with the model and a steady state.
    Otherwise, it is replaced by a state found by time stepping. In all cases, the model
    is restored to the original one, and is in a steady state.

    """
    caplog.set_level(logging.INFO)
    folder = tmp_path / "simulation"
    if saved_state != "none":
        # A different set of fractures gives different grids.
        fractures = {"fracture_indices": [0, 1]} if "mesh" in saved_state else {}
        initialize(make_model(folder, **fractures))
        if saved_state == "not_steady":
            path = folder / "steady_state.npz"
            arrays = dict(np.load(path))
            arrays["variables"] = arrays["variables"] + 1.0
            np.savez(path, **arrays)
        caplog.clear()

    model = make_model(folder)
    original_class, original_time_manager = model.__class__, model.time_manager
    initialize(model)

    for message in logged:
        assert message in caplog.text
    for message in not_logged:
        assert message not in caplog.text
    assert_restored(model, original_class, original_time_manager, folder)
    assert model.exporter._folder_name == folder
    assert _validate_steady_state(model, default_residual_tolerances())


@pytest.mark.parametrize("failure", ["solver", "never_steady", "saved_state"])
def test_pipeline_failure(tmp_path, failure):
    """If no steady state is found, an error is raised and an invalid saved state is
    discarded. The model is restored."""
    folder = tmp_path / "simulation"
    if failure == "saved_state":
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


@pytest.mark.parametrize("data_time", [None, 3 * pp.DAY])
def test_data_time_frozen(tmp_path, data_time):
    """Boundary values are those of the first time step of the simulation, or of the
    given data time, although the clock of the initialization is different."""
    params = {} if data_time is None else {"initialization_data_time": data_time}
    model = make_model(tmp_path / "simulation", **params)
    initialize(model)

    # Pressure is determined by the Dirichlet boundary values.
    variables = model.equation_system.get_variables(
        [model.pressure_variable], model.mdg.subdomains()
    )
    pressure = model.equation_system.get_variable_values(
        variables=variables, iterate_index=0
    )
    expected_time = pp.DAY if data_time is None else data_time
    np.testing.assert_allclose(pressure, 1e3 * (1 + expected_time / pp.DAY), rtol=1e-6)


@pytest.mark.parametrize(
    "model_class, perturbations",
    [
        (models.Thermoporomechanics, ["mechanical_stress", "pressure_stress"]),
        (TpsaPoromechanics, ["mechanical_stress"]),
    ],
)
def test_reference_state(tmp_path, model_class, perturbations):
    """After initialization, the model is in its reference state: Perturbations from
    the reference state vanish, and the total stress and porosity are equal to it."""
    model = make_model(tmp_path / "simulation", model_class)
    initialize(model)

    sd = model.mdg.subdomains(dim=model.nd)
    operators = {name: getattr(model, name)(sd) for name in perturbations}
    operators["stress"] = model.stress(sd) - model.reference_stress(sd)
    operators["porosity"] = model.porosity(sd) - model.reference_porosity(sd)
    for name, operator in operators.items():
        values = model.equation_system.evaluate(operator)
        np.testing.assert_allclose(values, 0, atol=1e-10, err_msg=name)


@pytest.mark.parametrize(
    "corruption",
    [
        "none",
        "missing_key",
        "changed_layout",
        "wrong_stress_shape",
        "extra_array",
        "non_finite",
        "unreadable",
    ],
)
def test_state_storage(tmp_path, corruption):
    """A saved state is loaded into the model, also as the reference state. Inconsistent
    states are rejected, and leave the model unchanged."""
    model = make_model(tmp_path / "simulation")
    model.prepare_simulation()
    storage = ModelStateStorage(tmp_path / "state.npz")
    storage.save_state(model)
    system = model.equation_system
    state = system.get_variable_values(iterate_index=0)

    path = storage.path
    arrays = dict(np.load(path))
    if corruption == "missing_key":
        del arrays["variables"]
    elif corruption == "changed_layout":
        arrays["variable_names"] = arrays["variable_names"][::-1]
    elif corruption == "wrong_stress_shape":
        arrays["reference_stress_0"] = arrays["reference_stress_0"][:-1]
    elif corruption == "extra_array":
        arrays["extra"] = np.zeros(1)
    elif corruption == "non_finite":
        arrays["variables"][0] = np.nan
    if corruption != "none":
        np.savez(path, **arrays)
    if corruption == "unreadable":
        path.write_bytes(b"not an archive")

    # Change the state of the model, to see that it is loaded or not changed.
    changed = state + 1.0
    system.set_variable_values(changed, iterate_index=0)
    if corruption == "none":
        storage.load_state(model)
        expected = state
    else:
        with pytest.raises(SteadyStateMismatch):
            storage.load_state(model)
        expected = changed
    np.testing.assert_array_equal(system.get_variable_values(iterate_index=0), expected)
    if corruption == "none":
        np.testing.assert_array_equal(system.get_variable_values(reference=True), state)


@pytest.mark.parametrize(
    "rate_tolerance, residual_tolerance, earliest_stop_time, stops",
    [
        (np.inf, np.inf, 0, True),
        (0, np.inf, 0, False),
        (np.inf, 0, 0, False),
        (np.inf, np.inf, 10 * pp.DAY, False),
    ],
)
def test_early_stop(
    tmp_path, rate_tolerance, residual_tolerance, earliest_stop_time, stops
):
    """The time stepping stops if both the rate of change and the residual are below
    the tolerances, after the earliest stop time. The first call has no rate."""
    model = make_model(tmp_path / "simulation")
    model.prepare_simulation()
    criterion = SteadyStateEarlyStopCriterion(
        rate_tolerances=defaultdict(lambda: rate_tolerance),
        residual_tolerances=defaultdict(lambda: residual_tolerance),
        earliest_stop_time=earliest_stop_time,
    )

    assert criterion.simulation_should_stop(model) is None
    model.time_manager.time += pp.DAY
    status = criterion.simulation_should_stop(model)
    assert isinstance(status, SteadyStateModelRunnerSuccess) == stops
