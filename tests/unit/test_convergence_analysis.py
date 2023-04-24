"""Module containing tests for the ConvergenceAnalysis class."""
from __future__ import annotations

import porepy as pp
import numpy as np
import pytest

from copy import deepcopy
from porepy.applications.convergence_analysis import ConvergenceAnalysis


@pytest.fixture(scope="module")
def stationary_mock_model() -> 'StationaryMockModel':
    """Set a stationary mock model."""

    class StationaryMockModel:
        """A mock model for unit-testing purposes."""
        def __init__(self, params: dict):
            self.params = params

        def prepare_simulation(self):
            """Prepare simulation by setting minimum required arguments."""
            ...

        def _is_time_dependent(self) -> bool:
            """Define whether the model is time-dependent."""
            return False

        def meshing_arguments(self) -> dict[str, float]:
            """Set the meshing arguments."""
            return self.params.get("meshing_arguments", {"cell_size": 1.0})

    return StationaryMockModel


@pytest.fixture(scope="module")
def time_dependent_mock_model() -> 'TimeDependentMockModel':
    """Set a stationary mock model."""

    class TimeDependentMockModel:
        """A mock model for unit-testing purposes."""
        def __init__(self, params: dict):
            self.params = params

        def prepare_simulation(self):
            """Prepare simulation by setting minimum required arguments."""
            self.time_manager = self.set_time_manager()

        def _is_time_dependent(self) -> bool:
            """Define whether the model is time-dependent."""
            return True

        def set_time_manager(self) -> pp.TimeManager:
            """Set the time manager."""
            return self.params.get("time_manager", pp.TimeManager([0, 1], 1, True))

        def meshing_arguments(self) -> dict[str, float]:
            """Set the meshing arguments."""
            return self.params.get("meshing_arguments", {"cell_size": 1.0})

    return TimeDependentMockModel


class TestInitializationAndSanityChecks:
    """The following tests are written to check the sanity of the input parameters"""

    def test_instantiation_stationary(self, stationary_mock_model):
        """Test initialization of attributes for a stationary mock model."""
        conv = ConvergenceAnalysis(model_class=stationary_mock_model, model_params={})
        assert conv.spatial_refinement_rate == 1
        assert conv.temporal_refinement_rate == 1
        assert conv.in_space
        assert not conv.in_time
        assert conv.levels == 1
        assert not conv._is_time_dependent
        assert len(conv.model_params) == 1
        assert conv.model_params[0]["meshing_arguments"]["cell_size"] == 1.0

    def test_raise_error_space_and_time_false(self, stationary_mock_model):
        """Test that an error is raised when 'in_space' and 'in_time' are False."""
        with pytest.raises(ValueError) as excinfo:
            msg = "At least one type of analysis should be performed."
            ConvergenceAnalysis(
                model_class=stationary_mock_model,
                model_params={},
                in_time=False,
                in_space=False,
            )
        assert msg in str(excinfo.value)

    def test_warns_space_true_and_refinement_one(self, time_dependent_mock_model):
        """Test warning is raised when in_space=False and spatial_refinement_rate>1."""
        msg = "'spatial_refinement_rate' is not being used."
        with pytest.warns() as record:
            ConvergenceAnalysis(
                model_class=time_dependent_mock_model,
                model_params={},
                in_space=False,
                spatial_refinement_rate=2,
                in_time=True,
                temporal_refinement_rate=2,
            )
        assert str(record[0].message) == msg

    def test_warns_time_true_and_refinement_one(self, time_dependent_mock_model):
        """Test warning is raised when in_time=False and temporal_refinement_rate>1."""
        msg = "'temporal_refinement_rate' is not being used."
        with pytest.warns() as record:
            ConvergenceAnalysis(
                model_class=time_dependent_mock_model,
                model_params={},
                in_space=True,
                spatial_refinement_rate=2,
                in_time=False,
                temporal_refinement_rate=2,
            )
        assert str(record[0].message) == msg

    def test_raise_error_when_non_constant_dt_used(self, time_dependent_mock_model):
        """Test if an error is raised when a non-constant time step is used."""
        msg = "Analysis in time only supports constant time step."
        with pytest.raises(NotImplementedError) as excinfo:
            ConvergenceAnalysis(
                model_class=time_dependent_mock_model,
                model_params={"time_manager": pp.TimeManager([0, 1], 0.1, False)},
                in_time=True,
                spatial_refinement_rate=2,
            )
        assert msg in str(excinfo.value)

    def test_list_of_meshing_arguments(self, stationary_mock_model):
        """Test if the list of mesh sizes is correctly obtained."""
        conv = ConvergenceAnalysis(
            model_class=stationary_mock_model,
            model_params={"meshing_arguments": {"cell_size": 0.2}},
            levels=3,
            spatial_refinement_rate=2,
        )
        known_cell_sizes = [0.2, 0.1, 0.05]
        actual_cell_sizes: list[float] = []
        for param in deepcopy(conv.model_params):
            actual_cell_sizes.append(param["meshing_arguments"]["cell_size"])
        np.testing.assert_array_almost_equal(known_cell_sizes, actual_cell_sizes)

    def test_list_of_time_managers(self, time_dependent_mock_model):
        """Test if the list of time managers is correctly obtained."""
        conv = ConvergenceAnalysis(
            model_class=time_dependent_mock_model,
            model_params={"time_manager": pp.TimeManager([0, 1], 0.2, True)},
            levels=4,
            in_space=False,
            in_time=True,
            temporal_refinement_rate=4,
        )
        known_time_steps = [0.2, 0.05, 0.0125, 0.003125]
        actual_time_steps: list[float] = []
        for param in deepcopy(conv.model_params):
            actual_time_steps.append(param["time_manager"].dt)
        np.testing.assert_array_almost_equal(known_time_steps, actual_time_steps)