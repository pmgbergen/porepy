"""Tests of functionality of :class:`~porepy.viz.solver_statistics.SolverStatistics`."""

import json
from pathlib import Path

import pytest
from deepdiff import DeepDiff

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceInfoHistory,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    ConvergenceStatusHistory,
    SimulationStatus,
)
from porepy.viz.solver_statistics import (
    NonlinearSolverAndTimeStatistics,
    NonlinearSolverStatistics,
    SolverStatistics,
    SolverStatisticsFactory,
    TimeStatistics,
)

# ! ---- Helper classes ---- !


class DummyModel(pp.SinglePhaseFlow):
    """Dummy model overwriting pp.SinglePhaseFlow to know whether it is nonlinear and/or
    time-dependent.
    """

    def _is_time_dependent(self):
        return self.params["time_dependent"]

    def _is_nonlinear_problem(self):
        return self.params["nonlinear_problem"]


class DummySubdomain:
    def __init__(self, dim, num_cells):
        self.dim = dim
        self.num_cells = num_cells


# ! ---- Reference statistics ---- !


def reference_solver_statistics_dict() -> dict:
    """Reference data bult from scratch in different tests below."""
    return {
        # SolverStatistics data
        "global": {
            "num_cells": {"0": 6, "1": 5, "2": 4},
            "num_domains": {"0": 1, "1": 1, "2": 1},
            "simulation_status_history": [
                SimulationStatus.FAILED,
                SimulationStatus.SUCCESSFUL,
            ],
            "final_simulation_status": SimulationStatus.SUCCESSFUL,
            "num_entries": 2,
        },
        # Custom data from SolverStatistics
        "0": {"foo": ["bar1", "bar2"], "simulation_status": SimulationStatus.FAILED},
        "1": {"foo": ["bar3"], "simulation_status": SimulationStatus.SUCCESSFUL},
    }


def reference_nonlinear_solver_statistics_dict() -> dict:
    """Reference data built from scratch in different tests below."""
    # Start from SolverStatistics reference data.
    reference_dict = reference_solver_statistics_dict()

    # Add NonlinearSolverStatistics data to global
    reference_dict["global"].update(
        {
            "num_iterations_history": [2, 1],
            "total_num_iterations": 3,
            "total_num_waisted_iterations": 2,
            "final_convergence_status": {
                "crit1": ConvergenceStatus.CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            },
        }
    )
    # Add NonlinearSolverStatistics data to each iteration
    reference_dict["0"].update(
        {
            # NonlinearSolverStatistics data for first outer iteration
            "num_iterations": 2,
            "simulation_status": SimulationStatus.FAILED,
            "convergence_status": {
                "crit1": [
                    ConvergenceStatus.NOT_CONVERGED,
                    ConvergenceStatus.DIVERGED,
                ],
                "crit2": [
                    ConvergenceStatus.NOT_CONVERGED,
                    ConvergenceStatus.CONVERGED,
                ],
            },
            "convergence_info": {
                "crit1": [1.0, 0.5],
                "crit2": [2.0, 1.5],
            },
        }
    )
    reference_dict["1"].update(
        {
            # NonlinearSolverStatistics data for second outer iteration
            "num_iterations": 1,
            "simulation_status": SimulationStatus.SUCCESSFUL,
            "convergence_status": {
                "crit1": [ConvergenceStatus.CONVERGED],
                "crit2": [ConvergenceStatus.CONVERGED],
            },
            "convergence_info": {
                "crit1": [0.1],
                "crit2": [0.2],
            },
        }
    )
    return reference_dict


def reference_time_statistics_dict() -> dict:
    """Reference data built from scratch in different tests below."""
    # Start from SolverStatistics reference data.
    reference_dict = reference_solver_statistics_dict()

    # Add TimeStatistics data to global.
    # Overwrite simulation_status_history - more meaningful for linear, time-dependent
    # problems.
    reference_dict["global"].update(
        {
            "simulation_status_history": [
                SimulationStatus.SUCCESSFUL,
                SimulationStatus.SUCCESSFUL,
            ],
            "final_time_reached": True,
            "total_num_time_steps": 2,
            "total_num_failed_time_steps": 0,
        }
    )
    # Add TimeStatistics data to each iteration
    reference_dict["0"].update(
        {
            "simulation_status": SimulationStatus.SUCCESSFUL,
            "time_index": 0,
            "time": 2.0,
            "dt": 2.0,
            "final_time_reached": False,
        }
    )
    reference_dict["1"].update(
        {
            "time_index": 1,
            "time": 2.5,
            "dt": 0.5,
            "final_time_reached": True,
        }
    )

    return reference_dict


def reference_nonlinear_time_statistics_dict() -> dict:
    """Reference data bult from scratch in different tests below."""
    # Start from NonlinearSolverStatistics reference data.
    reference_dict = reference_nonlinear_solver_statistics_dict()

    # Context: The narrative is a time-dependent nonlinear problem,
    # which is solved with two time steps. The first time step requires two nonlinear
    # iterations and fails. It is retried and solved in one nonlinear iteration,
    # here with a larger time step, which is successful and reaches the final time.

    # Add TimeStatistics data to global.
    reference_dict["global"].update(
        {
            "final_time_reached": True,
            "total_num_time_steps": 2,
            "total_num_failed_time_steps": 1,
        }
    )
    # Add TimeStatistics data to each iteration
    reference_dict["0"].update(
        {
            "time_index": 0,
            "time": 2.0,
            "dt": 2.0,
            "final_time_reached": False,
        }
    )
    reference_dict["1"].update(
        {
            "time_index": 1,
            "time": 2.5,
            "dt": 2.5,
            "final_time_reached": True,
        }
    )

    return reference_dict


# ! ---- SolverStatistics tests ---- !


def test_solver_statistics_initialization():
    """Tests initialization of SolverStatistics."""
    stats = SolverStatistics()
    assert stats.index == -1
    assert stats.path is None
    assert stats.num_cells == {}
    assert stats.num_domains == {}
    assert stats.simulation_status == SimulationStatus.IN_PROGRESS
    assert stats.simulation_status_history == []
    assert stats.custom_data == {}


def test_solver_statistic_attributes():
    """Tests initialized statistics in model integration."""
    model = DummyModel({"time_dependent": False, "nonlinear_problem": False})
    model.prepare_simulation()

    # Basic attributes of pp.SolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "index")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert hasattr(model.nonlinear_solver_statistics, "num_cells")
    assert hasattr(model.nonlinear_solver_statistics, "num_domains")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status_history")
    assert hasattr(model.nonlinear_solver_statistics, "custom_data")

    # Check that SolverStatistics has not path for storing.
    assert model.nonlinear_solver_statistics.path is None


@pytest.mark.parametrize(
    "subdomains, expected_num_cells, expected_num_domains",
    [
        ([DummySubdomain(0, 1)], {"0": 1}, {"0": 1}),
        (
            [DummySubdomain(1, 2), DummySubdomain(0, 3)],
            {"0": 3, "1": 2},
            {"0": 1, "1": 1},
        ),
        (
            [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)],
            {"0": 6, "1": 5, "2": 4},
            {"0": 1, "1": 1, "2": 1},
        ),
        (
            [DummySubdomain(2, 4), DummySubdomain(2, 5), DummySubdomain(1, 6)],
            {"1": 6, "2": 9},
            {"1": 1, "2": 2},
        ),
    ],
)
def test_log_mesh_information(subdomains, expected_num_cells, expected_num_domains):
    stats = SolverStatistics()
    stats.log_mesh_information(subdomains)
    assert stats.num_cells == expected_num_cells
    assert stats.num_domains == expected_num_domains


@pytest.mark.parametrize(
    "statuses",
    [
        [SimulationStatus.IN_PROGRESS, SimulationStatus.SUCCESSFUL],
        [
            SimulationStatus.FAILED,
            SimulationStatus.SUCCESSFUL,
            SimulationStatus.IN_PROGRESS,
        ],
    ],
)
def test_log_simulation_status(statuses):
    """Test logging simulation status in SolverStatistics."""
    stats = SolverStatistics()
    for s in statuses:
        stats.increase_index()
        stats.log_simulation_status(s)
    assert stats.simulation_status_history[: len(statuses)] == statuses
    assert stats.simulation_status_history == statuses
    assert stats.simulation_status == statuses[-1]


@pytest.mark.parametrize(
    "append, initial, update, expected",
    [
        # Empty initial data - just add new data.
        (True, {}, {"c": 4}, {"c": [4]}),
        (False, {}, {"c": 4}, {"c": 4}),
        # Single key - append or replace existing value.
        (True, {"a": [1]}, {"a": 2}, {"a": [1, 2]}),
        (False, {"a": [1]}, {"a": 2}, {"a": 2}),
        (True, {"a": 1}, {"a": 2}, {"a": [1, 2]}),
        (False, {"a": 1}, {"a": 2}, {"a": 2}),
        # No overlap in keys - keep existing values.
        (True, {"a": [1]}, {"b": 3}, {"a": [1], "b": [3]}),
        (False, {"a": [1]}, {"b": 3}, {"a": [1], "b": 3}),
        (True, {"a": 1}, {"b": 3}, {"a": 1, "b": [3]}),
        (False, {"a": 1}, {"b": 3}, {"a": 1, "b": 3}),
        # Overlap in keys - append or replace existing values.
        (True, {"a": [1]}, {"a": 2, "b": 3}, {"a": [1, 2], "b": [3]}),
        (False, {"a": [1]}, {"a": 2, "b": 3}, {"a": 2, "b": 3}),
        (True, {"a": 1}, {"a": 2, "b": 3}, {"a": [1, 2], "b": [3]}),
        (False, {"a": 1}, {"a": 2, "b": 3}, {"a": 2, "b": 3}),
    ],
)
def test_log_custom_data(append, initial, update, expected):
    stats = SolverStatistics()
    stats.custom_data = initial.copy()
    stats.log_custom_data(append=append, **update)
    assert stats.custom_data == expected


def test_solver_statistics_increase_index():
    """Test advancing SolverStatistics."""
    stats = SolverStatistics()
    stats.index = 5
    stats.custom_data = {"foo": "bar"}
    stats.increase_index()
    assert stats.index == 6
    assert stats.custom_data == {}


def test_solver_statistics_append_global_data():
    """Test appending global data from SolverStatistics."""
    # Initialize data dict
    data = {}

    # Initialize SolverStatistics and log some data.
    stats = SolverStatistics()

    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Mimic 1. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)

    # Append global data.
    out = stats.append_global_data(data)

    # Mimic 2. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)

    # Append global data.
    out = stats.append_global_data(data)

    # Compare against reference data.
    reference_data = {"global": reference_solver_statistics_dict()["global"]}
    assert (
        DeepDiff(
            out,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_solver_statistics_append_custom_data():
    """Test appending custom data with both append=True and append=False options."""
    stats = SolverStatistics()

    # Main data object.
    data = {}

    # Scenario 1. Log new custom data without appending.
    stats.increase_index()
    stats.log_custom_data(append=False, **{"foo": "bar1"})
    # Need to make sure that the data dict has the correct 'index' keys before
    # appending. This is typically ensured when calling the append_data method.
    data["0"] = {}
    data = stats.append_custom_data(data)
    assert DeepDiff(data["0"], {"foo": "bar1"}) == {}

    # Scenario 2. Log new custom data with appending.
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    data = stats.append_custom_data(data)
    assert DeepDiff(data["0"], {"foo": ["bar1", "bar2"]}) == {}

    # Scenario 3. Mimick new iteration before appending.
    stats.increase_index()
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    data["1"] = {}
    data = stats.append_custom_data(data)

    # Test that data was appended correctly, and previous data is not affected.
    reference_data = reference_solver_statistics_dict()
    # Only test custom data here.
    reference_data.pop("global")
    # Remove simulation status from comparison.
    reference_data["0"].pop("simulation_status")
    reference_data["1"].pop("simulation_status")

    assert DeepDiff(data, reference_data) == {}


def test_solver_statistics_append_iterative_data():
    """Test that append_iterative_data essentially just passes a dictionary."""
    stats = SolverStatistics()
    stats.increase_index()
    stats.simulation_status = SimulationStatus.FAILED
    data = {"0": {}}
    out = stats.append_iterative_data(data)
    assert (
        DeepDiff(
            out,
            {"0": {"simulation_status": SimulationStatus.FAILED}},
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_solver_statistics_append_data_solver_statistics():
    """Check whether solver statistics are correctly appended to data dict."""
    # Prepare initial data dict
    data = {}

    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Define SolverStatistics instance
    stats = SolverStatistics()

    # Mimick 1. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(**{"foo": "bar1"})
    stats.log_custom_data(append=True, **{"foo": "bar2"})

    # Summarize.
    out = stats.append_data(data)

    # Mimick 2. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})

    # Summarize.
    out = stats.append_data(data)

    # Compare against reference data.
    assert (
        DeepDiff(
            out,
            reference_solver_statistics_dict(),
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_solver_statistics_save():
    """Check whether solver statistics are correctly saved to file.

    Mimick two iterations to test the accumulation of data in the saved file.

    """
    if Path("solver_statistics.json").exists():
        Path("solver_statistics.json").unlink()
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Define SolverStatistics instance
    stats = SolverStatistics(path=Path("solver_statistics.json"))

    # Mimick 1. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(**{"foo": "bar1"})
    stats.log_custom_data(append=True, **{"foo": "bar2"})

    # Save to file.
    stats.save()

    # Mimick 2. iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("solver_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    assert (
        DeepDiff(
            data,
            reference_solver_statistics_dict(),
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("solver_statistics.json").unlink()


@pytest.mark.parametrize(
    ("path", "exists"),
    [
        ("solver_statistics.json", True),
        ("solver_statistics", True),
        (None, False),
    ],
)
def test_solver_statistics_save_in_model(path, exists):
    """Check whether solver statistics are correctly saved to file
    when integrated in model workflow."""

    # Make sure the exporting folder is empty.
    folder = Path("visualization")
    if folder.exists():
        for f in folder.glob("*"):
            f.unlink()

    # Create the model with the specified parameters.
    params = {
        "time_dependent": True,  # Needed to trigger saving during prepare_simulation.
        "nonlinear_problem": False,
        "folder_name": "visualization",
        "solver_statistics_file_name": path,
    }
    model = DummyModel(params)
    model.prepare_simulation()

    # Check whether file was saved and has correct suffix.
    if exists:
        assert model.nonlinear_solver_statistics.path.exists()
        assert model.nonlinear_solver_statistics.path.suffix == ".json"
    else:
        assert model.nonlinear_solver_statistics.path is None

    # Clean up.
    for f in folder.glob("*"):
        f.unlink()


# ! ---- NonlinearSolverStatistics tests ---- !


def test_nonlinear_solver_statistics_initialization():
    """Tests initialization of NonlinearSolverStatistics with SolverStatistics."""
    stats = NonlinearSolverStatistics()

    # Check SolverStatistics attributes.
    assert stats.index == -1
    assert stats.path is None
    assert stats.num_cells == {}
    assert stats.num_domains == {}
    assert stats.simulation_status == SimulationStatus.IN_PROGRESS
    assert stats.simulation_status_history == []
    assert stats.custom_data == {}

    # Check NonlinearSolverStatistics attributes.
    assert stats.num_iterations == 0
    assert stats.convergence_status == {}
    assert stats.convergence_info == {}
    assert stats.num_iterations_history == []


def test_nonlinear_solver_statistics_attributes():
    """Tests initialized statistics in model integration."""
    model = DummyModel({"time_dependent": False, "nonlinear_problem": True})
    model.prepare_simulation()

    # Basic attributes of pp.SolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "index")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert hasattr(model.nonlinear_solver_statistics, "num_cells")
    assert hasattr(model.nonlinear_solver_statistics, "num_domains")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status_history")
    assert hasattr(model.nonlinear_solver_statistics, "custom_data")

    # Check that SolverStatistics has not path for storing.
    assert model.nonlinear_solver_statistics.path is None

    # Specific attributes of pp.NonlinearSolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "num_iterations")
    assert hasattr(model.nonlinear_solver_statistics, "num_iterations_history")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_status")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_info")


def test_nonlinear_solver_statistics_log_convergence_status():
    """Tests logging convergence status and info in NonlinearSolverStatistics."""
    stats = NonlinearSolverStatistics()
    stats.increase_index()

    # 1. Iteration
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {"crit1": ConvergenceStatus.CONVERGED, "crit2": ConvergenceStatus.DIVERGED}
        )
    )
    assert (
        DeepDiff(
            stats.convergence_status,
            ConvergenceStatusHistory(
                {
                    "crit1": [ConvergenceStatus.CONVERGED],
                    "crit2": [ConvergenceStatus.DIVERGED],
                }
            ),
        )
        == {}
    )
    assert stats.num_iterations == 1

    # 2. Iteration
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.NOT_CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    assert (
        DeepDiff(
            stats.convergence_status,
            ConvergenceStatusHistory(
                {
                    "crit1": [
                        ConvergenceStatus.CONVERGED,
                        ConvergenceStatus.NOT_CONVERGED,
                    ],
                    "crit2": [ConvergenceStatus.DIVERGED, ConvergenceStatus.CONVERGED],
                }
            ),
        )
        == {}
    )
    assert stats.num_iterations == 2


def test_nonlinear_solver_statistics_log_convergence_info():
    """Tests logging convergence info in NonlinearSolverStatistics."""
    stats = NonlinearSolverStatistics()

    #  1. Iteration
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 1.0, "crit2": 2.0}))
    assert (
        DeepDiff(
            stats.convergence_info,
            ConvergenceInfoHistory({"crit1": [1.0], "crit2": [2.0]}),
        )
        == {}
    )

    # 2. Iteration
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.5, "crit2": 1.5}))
    assert (
        DeepDiff(
            stats.convergence_info,
            ConvergenceInfoHistory({"crit1": [1.0, 0.5], "crit2": [2.0, 1.5]}),
        )
        == {}
    )


def test_nonlinear_solver_statistics_increase_index():
    """Tests increasing index of NonlinearSolverStatistics."""
    stats = NonlinearSolverStatistics()
    stats.increase_index()
    stats.num_iterations = 2
    stats.convergence_status = {
        "crit1": [SimulationStatus.SUCCESSFUL, SimulationStatus.FAILED],
        "crit2": [SimulationStatus.FAILED, SimulationStatus.SUCCESSFUL],
    }
    stats.convergence_info = {
        "crit1": [1.0, 0.5],
    }
    stats.increase_index()
    assert stats.index == 1
    assert stats.custom_data == {}
    assert stats.num_iterations == 0
    assert stats.convergence_status == {}
    assert stats.convergence_info == {}


def test_nonlinear_solver_statistics_append_global_data():
    """Tests appending global data from NonlinearSolverStatistics."""
    stats = NonlinearSolverStatistics()

    # Previous data logged.
    stats.num_cells = {"0": 6, "1": 5, "2": 4}
    stats.num_domains = {"0": 1, "1": 1, "2": 1}
    stats.simulation_status_history = [
        SimulationStatus.FAILED,
        SimulationStatus.SUCCESSFUL,
    ]
    stats.num_iterations_history = [2, 1]
    stats.index = 1
    stats.num_iterations = 1
    stats.convergence_status = ConvergenceStatusHistory(
        {
            "crit1": [ConvergenceStatus.NOT_CONVERGED, ConvergenceStatus.CONVERGED],
            "crit2": [ConvergenceStatus.CONVERGED, ConvergenceStatus.CONVERGED],
        }
    )

    # Append global data.
    data = {}
    out = stats.append_global_data(data)
    print(out)

    # Compare against reference data.
    reference_data = reference_nonlinear_solver_statistics_dict()
    reference_data.pop("0")
    reference_data.pop("1")

    # Check that SolverStatistics global data is present.
    assert DeepDiff(out, reference_data, ignore_string_type_changes=True) == {}


def test_nonlinear_solver_statistics_append_iterative_data():
    """Tests appending iterative data from NonlinearSolverStatistics."""

    stats = NonlinearSolverStatistics()

    # Define statistics after 1. iteration.
    stats.increase_index()
    stats.num_iterations = 2
    stats.simulation_status_history = [
        SimulationStatus.FAILED,
    ]
    stats.convergence_status = ConvergenceStatusHistory(
        {
            "crit1": [ConvergenceStatus.NOT_CONVERGED, ConvergenceStatus.DIVERGED],
            "crit2": [ConvergenceStatus.NOT_CONVERGED, ConvergenceStatus.CONVERGED],
        }
    )
    stats.convergence_info = ConvergenceInfoHistory(
        {
            "crit1": [1.0, 0.5],
            "crit2": [2.0, 1.5],
        }
    )

    # Make sure that the data dict has the correct 'index' key before appending.
    # Typically prepared when calling the `append_data` method.
    data = {"0": {}}
    out = stats.append_iterative_data(data)

    # Compare against reference data. Restrict to "0", ignore custom data, and cast.
    reference_data = {"0": reference_nonlinear_solver_statistics_dict()["0"]}
    reference_data["0"].pop("foo")
    assert (
        DeepDiff(
            out,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
            ignore_type_in_groups=[(dict, ConvergenceInfoHistory)],
        )
        == {}
    )


def test_nonlinear_solver_statistics_append_data():
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize NonlinearSolverStatistics.
    stats = NonlinearSolverStatistics()

    # ! ---- Mimick FAILED first (outer) iteration ---- !

    # 1. sub-iteration of FAILED iteration
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.IN_PROGRESS)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.NOT_CONVERGED,
                "crit2": ConvergenceStatus.NOT_CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 1.0, "crit2": 2.0}))

    # 2. sub-iteration of FAILED iteration
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.DIVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.5, "crit2": 1.5}))

    # Append data.
    data = {}
    data = stats.append_data(data)

    # ! ---- Mimick SUCCESSFUL second (outer) iteration ---- !

    # 1. sub-iteration of FAILED iteration
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.1, "crit2": 0.2}))

    # Save to file.
    data = stats.append_data(data)

    # Compare against reference data.
    assert (
        DeepDiff(
            data,
            reference_nonlinear_solver_statistics_dict(),
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
            ignore_type_in_groups=[(dict, ConvergenceInfoHistory)],
        )
        == {}
    )


def test_nonlinear_solver_statistics_save():
    """Tests saving NonlinearSolverStatistics to file."""

    # Initialize NonlinearSolverStatistics.
    stats = NonlinearSolverStatistics(path=Path("nonlinear_solver_statistics.json"))

    # Mimick status after two iterations (see e.g.
    # test_nonlinear_solver_statistics_append_data).
    stats.num_cells = {"0": 6, "1": 5, "2": 4}
    stats.num_domains = {"0": 1, "1": 1, "2": 1}
    stats.simulation_status_history = [
        SimulationStatus.FAILED,
        SimulationStatus.SUCCESSFUL,
    ]
    stats.index = 1
    stats.num_iterations_history = [2, 1]
    stats.num_iterations = 1
    stats.convergence_status = ConvergenceStatusHistory(
        {
            "crit1": [ConvergenceStatus.CONVERGED],
            "crit2": [ConvergenceStatus.CONVERGED],
        }
    )
    stats.convergence_info = ConvergenceInfoHistory(
        {
            "crit1": [0.1],
            "crit2": [0.2],
        }
    )

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("nonlinear_solver_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_nonlinear_solver_statistics_dict()
    reference_data.pop("0")  # Only test data for second iteration here.
    reference_data["1"].pop("foo")  # Remove custom data.

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
            ignore_type_in_groups=[(dict, ConvergenceInfoHistory)],
        )
        == {}
    )

    # Clean up.
    Path("nonlinear_solver_statistics.json").unlink()


def test_nonlinear_solver_statistics_integration():
    """Tests the complete workflow of NonlinearSolverStatistics."""
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize NonlinearSolverStatistics.
    stats = NonlinearSolverStatistics(path=Path("nonlinear_solver_statistics.json"))

    # ! ---- Mimick FAILED first (outer) iteration ---- !

    # 1. sub-iteration of FAILED iteration
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.IN_PROGRESS)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.NOT_CONVERGED,
                "crit2": ConvergenceStatus.NOT_CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 1.0, "crit2": 2.0}))

    # 2. sub-iteration of FAILED iteration
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.DIVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.5, "crit2": 1.5}))

    # Save to file.
    stats.save()

    # ! ---- Mimick SUCCESSFUL second (outer) iteration ---- !

    # 1. sub-iteration of SUCCESSFUL iteration
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.1, "crit2": 0.2}))

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("nonlinear_solver_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_nonlinear_solver_statistics_dict()
    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("nonlinear_solver_statistics.json").unlink()


# TimeStatistics tests
def test_time_statistics_initialization():
    """Test initialization of TimeStatistics."""
    stats = TimeStatistics()

    # Check SolverStatistics attributes.
    assert stats.index == -1
    assert stats.path is None
    assert stats.num_cells == {}
    assert stats.num_domains == {}
    assert stats.simulation_status == SimulationStatus.IN_PROGRESS
    assert stats.simulation_status_history == []
    assert stats.custom_data == {}

    # Check TimeStatistics attributes.
    assert stats.time_index == 0
    assert stats.time == 0.0
    assert stats.dt == 0.0
    assert stats.final_time_reached is False


def test_time_statistics_attributes():
    """Tests initialized statistics in model integration."""
    model = DummyModel({"time_dependent": True, "nonlinear_problem": False})
    model.prepare_simulation()

    # Basic attributes of pp.SolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "index")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert hasattr(model.nonlinear_solver_statistics, "num_cells")
    assert hasattr(model.nonlinear_solver_statistics, "num_domains")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status_history")
    assert hasattr(model.nonlinear_solver_statistics, "custom_data")

    # Check that SolverStatistics has not path for storing.
    assert model.nonlinear_solver_statistics.path is None

    # Check attributes of pp.TimeStatistics
    assert hasattr(model.nonlinear_solver_statistics, "time_index")
    assert hasattr(model.nonlinear_solver_statistics, "time")
    assert hasattr(model.nonlinear_solver_statistics, "dt")
    assert hasattr(model.nonlinear_solver_statistics, "final_time_reached")


@pytest.mark.parametrize(
    ("time_index", "time", "dt", "final_time_reached"),
    [
        (3, 1.5, 0.1, True),
        (5, 2.0, 0.2, False),
    ],
)
def test_time_statistics_log_time_information(time_index, time, dt, final_time_reached):
    """ "Test logging time information in TimeStatistics."""
    stats = TimeStatistics()
    stats.log_time_information(time_index, time, dt, final_time_reached)
    assert stats.time_index == time_index
    assert stats.time == time
    assert stats.dt == dt
    assert stats.final_time_reached is final_time_reached


def test_time_statistics_increase_index():
    """Tests advancing TimeStatistics - check that time statistics are kept."""
    stats = TimeStatistics()
    stats.increase_index()
    stats.time_index = 5
    stats.time = 2.0
    stats.dt = 0.2
    stats.final_time_reached = True
    stats.increase_index()
    assert stats.index == 1
    assert stats.custom_data == {}
    assert stats.time_index == 5
    assert stats.time == 2.0
    assert stats.dt == 0.2
    assert stats.final_time_reached is True


def test_time_statistics_append_global_data():
    """Tests appending global data from TimeStatistics."""
    stats = TimeStatistics()

    # Define global data and time specific data of reference as defined after two time
    # steps.
    stats.time_index = 1
    stats.time = 2.5
    stats.dt = 0.5
    stats.final_time_reached = True

    # Previous data logged.
    stats.num_cells = {"0": 6, "1": 5, "2": 4}
    stats.num_domains = {"0": 1, "1": 1, "2": 1}
    stats.simulation_status_history = [
        SimulationStatus.SUCCESSFUL,
        SimulationStatus.SUCCESSFUL,
    ]
    stats.index = 1

    # Append global data.
    data = {}
    out = stats.append_global_data(data)

    # Compare against reference data.
    reference_data = reference_time_statistics_dict()
    reference_data.pop("0")
    reference_data.pop("1")

    assert (
        DeepDiff(
            out,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_time_statistics_append_iterative_data():
    """Tests appending iterative data from TimeStatistics."""
    stats = TimeStatistics()

    # Define reference statistics after 1. time step.
    stats.index = 1
    stats.simulation_status = SimulationStatus.SUCCESSFUL
    stats.time_index = 1
    stats.time = 2.5
    stats.dt = 0.5
    stats.final_time_reached = True

    # Make sure that the data dict has the correct 'index' key before appending.
    # Typically prepared when calling the `append_data` method.
    data = {"1": {}}
    out = stats.append_iterative_data(data)

    # Compare against reference data. Restrict to "1".
    reference_data = {"1": reference_time_statistics_dict()["1"]}
    # Remove custom data.
    reference_data["1"].pop("foo")

    assert (
        DeepDiff(
            out,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_time_statistics_append_data():
    """Tests appending data from TimeStatistics.

    Identical to test_time_statistics_integration but with data appending instead of
    saving.

    """
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize TimeStatistics.
    stats = TimeStatistics()

    # ! ---- Mimick two time steps ---- !

    # 1. Mimick first time step.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_time_information(0, 2.0, 2.0, False)

    # Append data.
    data = {}
    data = stats.append_data(data)

    # 2. Mimick second time step
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_time_information(1, 2.5, 0.5, True)

    # Save to file.
    data = stats.append_data(data)
    print(data)
    # Compare against reference data.
    reference_data = reference_time_statistics_dict()

    print(reference_data)

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_time_statistics_save():
    """Tests saving TimeStatistics to file."""
    stats = TimeStatistics(path=Path("time_statistics.json"))

    # Mimick status after two time steps (see e.g.
    # test_nonlinear_solver_statistics_append_data).
    stats.num_cells = {"0": 6, "1": 5, "2": 4}
    stats.num_domains = {"0": 1, "1": 1, "2": 1}
    stats.simulation_status_history = [
        SimulationStatus.SUCCESSFUL,
        SimulationStatus.SUCCESSFUL,
    ]
    stats.simulation_status = SimulationStatus.SUCCESSFUL
    stats.index = 1
    stats.time_index = 1
    stats.time = 2.5
    stats.dt = 0.5
    stats.final_time_reached = True

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("time_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_time_statistics_dict()
    # Drop "0" and custom data.
    reference_data.pop("0")
    reference_data["1"].pop("foo")

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("time_statistics.json").unlink()


def test_time_statistics_integration():
    """Tests the integration of TimeStatistics with other components.

    Combine with SolverStatistics and mimick a time-dependent simulation step.

    """
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize TimeStatistics.
    stats = TimeStatistics(path=Path("time_statistics.json"))

    # ! ---- Mimick two time steps ---- !

    # 1. Mimick first time step.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_time_information(0, 2.0, 2.0, False)

    # Save to file and advance counter.
    stats.save()

    # 2. Mimick second time step
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_time_information(1, 2.5, 0.5, True)

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("time_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_time_statistics_dict()

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("time_statistics.json").unlink()


# ! ---- NonlinearSolverAndTimeStatistics tests ---- !


def test_nonlinear_solver_and_time_statistics_initialization():
    """Test initialization of NonlinearSolverAndTimeStatistics."""
    stats = NonlinearSolverAndTimeStatistics()

    # Check SolverStatistics attributes.
    assert stats.index == -1
    assert stats.path is None
    assert stats.num_cells == {}
    assert stats.num_domains == {}
    assert stats.simulation_status == SimulationStatus.IN_PROGRESS
    assert stats.simulation_status_history == []
    assert stats.custom_data == {}

    # Check NonlinearSolverStatistics attributes.
    assert stats.num_iterations == 0
    assert stats.convergence_status == {}
    assert stats.convergence_info == {}
    assert stats.num_iterations_history == []

    # Check TimeStatistics attributes.
    assert stats.time_index == 0
    assert stats.time == 0.0
    assert stats.dt == 0.0
    assert stats.final_time_reached is False


def test_nonlinear_solver_and_time_statistics_attributes():
    """Tests initialized statistics in model integration."""
    model = DummyModel({"time_dependent": True, "nonlinear_problem": True})
    model.prepare_simulation()

    # Basic attributes of pp.SolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "index")
    assert hasattr(model.nonlinear_solver_statistics, "path")
    assert hasattr(model.nonlinear_solver_statistics, "num_cells")
    assert hasattr(model.nonlinear_solver_statistics, "num_domains")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status")
    assert hasattr(model.nonlinear_solver_statistics, "simulation_status_history")
    assert hasattr(model.nonlinear_solver_statistics, "custom_data")

    # Check that SolverStatistics has not path for storing.
    assert model.nonlinear_solver_statistics.path is None

    # Check attributes of pp.NonlinearSolverStatistics
    assert hasattr(model.nonlinear_solver_statistics, "num_iterations")
    assert hasattr(model.nonlinear_solver_statistics, "num_iterations_history")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_status")
    assert hasattr(model.nonlinear_solver_statistics, "convergence_info")

    # Check attributes of pp.TimeStatistics
    assert hasattr(model.nonlinear_solver_statistics, "time_index")
    assert hasattr(model.nonlinear_solver_statistics, "time")
    assert hasattr(model.nonlinear_solver_statistics, "dt")
    assert hasattr(model.nonlinear_solver_statistics, "final_time_reached")


def test_nonlinear_solver_and_time_statistics_append_data():
    """Tests appending data from NonlinearSolverAndTimeStatistics.

    Identical to test_nonlinear_solver_and_time_statistics_integration but with data
    appending instead of saving.

    """
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize NonlinearSolverAndTimeStatistics.
    stats = NonlinearSolverAndTimeStatistics()

    # ! ---- Mimick two time steps (first fails, second succeeds) ---- !

    # 1. Mimick first (failing) time step.
    # It is solved using two nonlinear iterations (stopped due to DIVERGED).
    # 1. iteration of FAILED time step.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.IN_PROGRESS)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.NOT_CONVERGED,
                "crit2": ConvergenceStatus.NOT_CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 1.0, "crit2": 2.0}))
    stats.log_time_information(0, 2.0, 2.0, False)

    # 2. iteration of FAILED time step
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.DIVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.5, "crit2": 1.5}))
    stats.log_time_information(0, 2.0, 2.0, False)

    # Append data.
    data = {}
    data = stats.append_data(data)

    # 2. Mimick second time step using 1 iteration of SUCCESSFUL iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.1, "crit2": 0.2}))
    stats.log_time_information(1, 2.5, 2.5, True)

    # Append data.
    data = stats.append_data(data)

    # Compare against reference data.
    reference_data = reference_nonlinear_time_statistics_dict()
    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )


def test_nonlinear_solver_and_time_statistics_save():
    """Tests saving NonlinearSolverAndTimeStatistics to file."""
    # Initialize NonlinearSolverAndTimeStatistics.
    stats = NonlinearSolverAndTimeStatistics(
        path=Path("nonlinear_solver_and_time_statistics.json")
    )

    # Mimick status after two time steps (see
    # test_nonlinear_solver_and_time_statistics_append_data).
    stats.num_cells = {"0": 6, "1": 5, "2": 4}
    stats.num_domains = {"0": 1, "1": 1, "2": 1}
    stats.simulation_status = SimulationStatus.SUCCESSFUL
    stats.simulation_status_history = [
        SimulationStatus.FAILED,
        SimulationStatus.SUCCESSFUL,
    ]
    stats.index = 1
    stats.num_iterations_history = [2, 1]
    stats.time_index = 1
    stats.time = 2.5
    stats.dt = 2.5
    stats.final_time_reached = True
    stats.num_iterations = 1
    stats.convergence_status = ConvergenceStatusHistory(
        {
            "crit1": [ConvergenceStatus.CONVERGED],
            "crit2": [ConvergenceStatus.CONVERGED],
        }
    )
    stats.convergence_info = ConvergenceInfoHistory(
        {
            "crit1": [0.1],
            "crit2": [0.2],
        }
    )

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("nonlinear_solver_and_time_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_nonlinear_time_statistics_dict()
    reference_data.pop("0")  # Only test data for second iteration here.
    reference_data["1"].pop("foo")  # Remove custom data.

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("nonlinear_solver_and_time_statistics.json").unlink()


def test_nonlinear_solver_and_time_statistics_integration():
    """Tests the complete workflow of NonlinearSolverAndTimeStatistics."""
    # Use some dummy domains.
    subdomains = [DummySubdomain(2, 4), DummySubdomain(1, 5), DummySubdomain(0, 6)]

    # Initialize NonlinearSolverAndTimeStatistics.
    stats = NonlinearSolverAndTimeStatistics(
        path=Path("nonlinear_solver_and_time_statistics.json")
    )

    # ! ---- Mimick two time steps (first fails, second succeeds) ---- !

    # 1. Mimick first (failing) time step.
    # It is solved using two nonlinear iterations (stopped due to DIVERGED).
    # 1. iteration of FAILED time step.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.IN_PROGRESS)
    stats.log_custom_data(append=True, **{"foo": "bar1"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.NOT_CONVERGED,
                "crit2": ConvergenceStatus.NOT_CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 1.0, "crit2": 2.0}))
    stats.log_time_information(0, 2.0, 2.0, False)

    # 2. iteration of FAILED time step
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.FAILED)
    stats.log_custom_data(append=True, **{"foo": "bar2"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.DIVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.5, "crit2": 1.5}))
    stats.log_time_information(0, 2.0, 2.0, False)

    # Save to file.
    stats.save()

    # 2. Mimick second time step using 1 iteration of SUCCESSFUL iteration.
    stats.increase_index()
    stats.log_mesh_information(subdomains)
    stats.log_simulation_status(SimulationStatus.SUCCESSFUL)
    stats.log_custom_data(append=True, **{"foo": "bar3"})
    stats.log_convergence_status(
        ConvergenceStatusCollection(
            {
                "crit1": ConvergenceStatus.CONVERGED,
                "crit2": ConvergenceStatus.CONVERGED,
            }
        )
    )
    stats.log_convergence_info(ConvergenceInfoCollection({"crit1": 0.1, "crit2": 0.2}))
    stats.log_time_information(1, 2.5, 2.5, True)

    # Save to file.
    stats.save()

    # Check saved data.
    with open(Path("nonlinear_solver_and_time_statistics.json"), "r") as f:
        data = json.load(f)

    # Compare against reference data.
    reference_data = reference_nonlinear_time_statistics_dict()

    assert (
        DeepDiff(
            data,
            reference_data,
            ignore_numeric_type_changes=True,
            ignore_string_type_changes=True,
        )
        == {}
    )

    # Clean up.
    Path("nonlinear_solver_and_time_statistics.json").unlink()


# ! ---- SolverStatisticsFactory tests ---- !


@pytest.mark.parametrize(
    "is_nonlinear, is_time, expected_cls",
    [
        (False, False, SolverStatistics),
        (True, False, NonlinearSolverStatistics),
        (False, True, TimeStatistics),
        (True, True, NonlinearSolverAndTimeStatistics),
    ],
)
def test_solver_statistics_factory_unit(is_nonlinear, is_time, expected_cls):
    """Test the SolverStatisticsFactory to ensure it returns the correct class type
    based on the input flags for nonlinearity and time dependency.

    """
    cls = SolverStatisticsFactory.create_statistics_type(is_nonlinear, is_time)
    assert cls is expected_cls


@pytest.mark.parametrize(
    "is_nonlinear, is_time, expected_cls",
    [
        (False, False, SolverStatistics),
        (True, False, NonlinearSolverStatistics),
        (False, True, TimeStatistics),
        (True, True, NonlinearSolverAndTimeStatistics),
    ],
)
def test_solver_statistics_factory_integration(is_nonlinear, is_time, expected_cls):
    """Test integration of solver statistics factory within models. Make sure that the
    correct solver statistics object is created based on model properties.
    """
    params = {"time_dependent": is_time, "nonlinear_problem": is_nonlinear}
    model = DummyModel(params)
    model.prepare_simulation()
    assert hasattr(model, "nonlinear_solver_statistics")
    assert isinstance(model.nonlinear_solver_statistics, expected_cls)


def test_solver_statistic_factory_nonlinear_and_time_dependent():
    """Test integration of solver statistics factory within models. Make sure that the
    correct solver statistics object is created for a Poromechanics model.

    """
    model = DummyModel({"time_dependent": True, "nonlinear_problem": True})
    model.prepare_simulation()
    assert hasattr(model, "nonlinear_solver_statistics")
    assert isinstance(model.nonlinear_solver_statistics, pp.SolverStatistics)
    assert isinstance(model.nonlinear_solver_statistics, pp.NonlinearSolverStatistics)
    assert isinstance(model.nonlinear_solver_statistics, pp.TimeStatistics)
    assert isinstance(
        model.nonlinear_solver_statistics, pp.NonlinearSolverAndTimeStatistics
    )
