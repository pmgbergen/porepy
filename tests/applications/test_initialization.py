"""Tests of the initialization of a model to a steady state.

We test:
    - The pipeline: finding, saving, loading and discarding the steady state, and that
      the model is restored after initialization, also if it fails.
    - That time dependent boundary values are frozen at the time of the first time step.
    - That the reference state is consistent with the model after initialization.
    - The storage of the state.

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
from porepy.applications.initialization.model_state_storage import (
    ModelStateStorage,
    LoadedStateMismatch,
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


class RampedSource(pp.PorePyModel):
    """Fluid source of 1e-3 * (1 + t / day) in each cell, updated with the time
    dependent data of the model."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return super().fluid_source(subdomains) + pp.ad.TimeDependentDenseArray(
            "ramped_source", subdomains
        )

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        for sd, data in self.mdg.subdomains(return_data=True):
            values = np.full(sd.num_cells, 1e-3 * (1 + self.time_manager.time / pp.DAY))
            pp.set_solution_values("ramped_source", values, data, iterate_index=0)


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
    """Model with a fracture, time dependent boundary values and source, and time steps
    of one day."""
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
        ("none", ["Saved state"], ["Trying to load"]),
        ("valid", ["Trying to load"], ["Saved state", "discarded"]),
        ("wrong_mesh", ["inconsistent", "Saved state"], []),
        ("not_steady", ["Not a steady state", "Saved state"], []),
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
            # Perturb the pressure in the same way at the iterate and the time step
            # (a uniform shift would not change the residual), but not its reference.
            for key in arrays:
                if "::pressure::" in key and "reference_solutions" not in key:
                    rng = np.random.default_rng(0)
                    arrays[key] = arrays[key] + 1e3 * rng.random(arrays[key].shape)
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


def test_data_time_frozen(tmp_path):
    """Boundary values are those of the first time step of the simulation, although the clock of the initialization is different."""
    model = make_model(tmp_path / "simulation")
    initialize(model)

    # The boundary values, also as reference values, are those of the first time step.
    expected_time = pp.DAY
    for _, data in model.mdg.boundaries(return_data=True):
        if model.pressure_variable in data.get(pp.ITERATE_SOLUTIONS, {}):
            for kwargs in [{"reference": True}, {"iterate_index": 0}]:
                pressure = pp.get_solution_values(
                    model.pressure_variable, data, **kwargs
                )
                np.testing.assert_allclose(pressure, 1e3 * (1 + expected_time / pp.DAY))


@pytest.mark.parametrize(
    "hook", ["after_nonlinear_convergence", "after_nonlinear_failure"]
)
def test_time_dependent_data_in_initialization_context(tmp_path, hook):
    """Inside the initialization context, the boundary values and the source term of a
    time step are those of the data time, whatever the clock is. The clock is restored
    when the nonlinear loop has finished, with either hook. Outside the context, they
    are those of the clock."""
    model = make_model(tmp_path / "simulation")
    # The data time is the first target time of the simulation.
    ramp = lambda time: 1 + time / pp.DAY  # noqa: E731
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
        np.testing.assert_allclose(source, 1e-3 * expected_ramp)
        np.testing.assert_allclose(pressure, 1e3 * expected_ramp)
        getattr(model, hook)()
        assert model.time_manager.time == clock

    # Without the freeze, the values are those of the clock.
    clock_in_step, source, pressure = values_of_time_step()
    assert clock_in_step == clock
    np.testing.assert_allclose(source, 1e-3 * ramp(clock))
    np.testing.assert_allclose(pressure, 1e3 * ramp(clock))


def test_mechanics_required(tmp_path):
    """A model without mechanics cannot be initialized."""
    model = models.MassBalance({"folder_name": str(tmp_path / "simulation")})
    with pytest.raises(ValueError, match="mechanics"):
        initialize(model)


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
        "missing_array",
        "wrong_shape",
        "extra_array",
        "non_finite",
        "unreadable",
    ],
)
def test_state_storage(tmp_path, corruption):
    """A saved state is loaded into the model exactly as it was saved. Inconsistent
    states are rejected, and leave the model unchanged."""
    model = make_model(tmp_path / "simulation")
    model.prepare_simulation()
    system = model.equation_system
    # A state in which the current, previous and reference values all differ.
    rng = np.random.default_rng(0)
    system.set_variable_values(rng.random(system.num_dofs()), reference=True)
    system.set_variable_values(rng.random(system.num_dofs()), time_step_index=0)
    system.set_variable_values(rng.random(system.num_dofs()), iterate_index=0)
    storage = ModelStateStorage(tmp_path / "state.npz")
    storage.save_state(model)
    saved = storage.capture_state(model)

    path = storage.path
    arrays = dict(np.load(path))
    key = next(k for k in arrays if "::iterate_solutions::" in k and "sd" in k)
    if corruption == "missing_array":
        del arrays[key]
    elif corruption == "wrong_shape":
        arrays[key] = arrays[key][:-1]
    elif corruption == "extra_array":
        arrays["extra"] = np.zeros(1)
    elif corruption == "non_finite":
        arrays[key][0] = np.nan
    if corruption != "none":
        np.savez(path, **arrays)
    if corruption == "unreadable":
        path.write_bytes(b"not an archive")

    # Change the state of the model, to see that it is loaded or not changed.
    for kwargs in [{"reference": True}, {"time_step_index": 0}, {"iterate_index": 0}]:
        system.set_variable_values(rng.random(system.num_dofs()), **kwargs)
    changed = storage.capture_state(model)
    if corruption == "none":
        storage.load_state(model)
        expected = saved
    else:
        with pytest.raises(LoadedStateMismatch):
            storage.load_state(model)
        expected = changed
    result = storage.capture_state(model)
    assert result.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name], err_msg=name)
