"""Module containing tests for the ConvergenceAnalysis class.

Tested functionality:
    - Default parameters at instantiation and sanity checks.
    - The `run_analysis()` method.
    - The `order_of_convergence()` method.
    - The `export_error_to_txt()` method.

"""
from __future__ import annotations

import porepy as pp
import numpy as np
import pytest

from copy import deepcopy
from dataclasses import dataclass
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from typing import Literal

# -----> Fixtures that are required on a module level.
@pytest.fixture(scope="module")
def stationary_mock_model() -> 'StationaryMockModel':
    """Set a stationary mock model.

    Returns:
        Stationary mock model.

    """
    class StationaryMockModel:
        """A stationary mock model for unit-testing purposes.

        Parameters:
            params: Model parameters.

        """
        def __init__(self, params: dict):
            self.params = params

        def prepare_simulation(self) -> None:
            """Prepare simulation. Nothing to do here."""

        def _is_time_dependent(self) -> bool:
            """Define whether the model is time-dependent."""
            return False

        def meshing_arguments(self) -> dict[str, float]:
            """Set the meshing arguments."""
            return self.params.get("meshing_arguments", {"cell_size": 1.0})

    return StationaryMockModel


@pytest.fixture(scope="module")
def time_dependent_mock_model() -> 'TimeDependentMockModel':
    """Set a time-dependent mock model.

    Returns:
        Time-dependent mock model.

    """
    class TimeDependentMockModel:
        """A time-dependent mock model for unit-testing purposes.

        Parameters:
            params: Model parameters.

        """
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


@pytest.fixture(scope="module")
def stationary_model() -> 'StationaryModel':
    """Stationary flow model.

    Returns:
        Stationary flow model with default parameters.

    """
    @dataclass
    class StationaryModelSaveData:
        """Data class to store errors."""

        error_var_0: float  # error associated with variable 0
        error_var_1: float  # error associated with variable 1

    class StationaryModelDataSaving(VerificationDataSaving):
        """Class that collects and store data."""

        def collect_data(self) -> StationaryModelSaveData:
            """Collect and return data.

            Returns:
                Data class with attributes ``error_var_0`` and ``error_var_1``.

            """
            # First error is set as the inverse of the number of cells.
            error_var_0 = 1 / self.mdg.subdomains()[0].num_cells

            # Second error is set as the inverse of four times the number of cells.
            error_var_1 = 1 / (4 * self.mdg.subdomains()[0].num_cells)

            # Instantiate data class
            collected_data = StationaryModelSaveData(error_var_0, error_var_1)

            return collected_data

    class StationaryModelSolutionStrategy(pp.SolutionStrategy):
        """Solution strategy for the stationary flow model."""

        def __init__(self, params: dict):
            super().__init__(params)
            self.results: list[StationaryModelSaveData] = []

        def _is_nonlinear_problem(self) -> bool:
            """Whether the model is non-linear."""
            return False

        def _is_time_dependent(self) -> bool:
            """Whether the model is time-dependent."""
            return False

    class StationaryModel(
        StationaryModelSolutionStrategy,
        StationaryModelDataSaving,
        SinglePhaseFlow,
    ):
        """Mixer class for the stationary flow model."""

    return StationaryModel


@pytest.fixture(scope="module")
def time_dependent_model() -> 'TimeDependentModel':
    """Time-dependent flow model.

    Returns:
        Time-dependent flow model with default parameters.

    """
    @dataclass
    class TimeDependentModelSaveData:
        """Collect and return data.

        Returns:
            Data class with attributes ``error_var_0`` and ``error_var_1``.

        """
        error_var_0: float  # error associated with variable 0
        error_var_1: float  # error associated with variable 1

    class TimeDependentModelDataSaving(VerificationDataSaving):
        """Class that collects and store data."""

        def collect_data(self) -> TimeDependentModelSaveData:
            """Collect and return data.

            Returns:
                Data class with attributes ``error_var_0`` and ``error_var_1``.

            """
            # Spatial error is set as the inverse of the number of cells.
            error_var_0 = 1 / self.mdg.subdomains()[0].num_cells

            # Temporal error is set as the inverse of the time step.
            error_var_1 = 1 / self.time_manager.dt

            # Instantiate data class
            collected_data = TimeDependentModelSaveData(error_var_0, error_var_1)

            return collected_data

    class TimeDependentModelSolutionStrategy(pp.SolutionStrategy):
        """Solution strategy for the time-dependent flow model."""

        def __init__(self, params: dict):
            super().__init__(params)
            self.results: list[TimeDependentModelSaveData] = []

        def _is_nonlinear_problem(self) -> bool:
            """Whether the problem is non-linear."""
            return True

        def _is_time_dependent(self) -> bool:
            """Whether the problem is time-dependent."""
            return True

    class TimeDependentModel(
        TimeDependentModelSolutionStrategy,
        TimeDependentModelDataSaving,
        SinglePhaseFlow,
    ):
        """Mixer class for the time-dependent flow model."""

    return TimeDependentModel


# -----> Tests for default parameters at instantiation and sanity checks.
class TestInitializationAndSanityChecks:
    """The following tests are written to check the sanity of the input parameters."""

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
        msg = "At least one type of analysis should be performed."
        with pytest.raises(ValueError) as excinfo:
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


# -----> Tests for the `run_analysis()` method.
class TestRunAnalysis:
    """Collection of tests to check that `run_analysis()` is working correctly."""

    def test_stationary_model(self, stationary_model):
        """Check that successively refined stationary flow models are correctly run."""
        conv = ConvergenceAnalysis(
            model_class=stationary_model,
            model_params={},
            levels=2,
            spatial_refinement_rate=2,
        )
        results = conv.run_analysis()
        assert results[0].error_var_0 == 0.25
        assert results[0].error_var_1 == 0.0625
        assert results[1].error_var_0 == 0.0625
        assert results[1].error_var_1 == 0.015625

    def test_time_dependent_model(self, time_dependent_model):
        """Check that successively refined dynamic flow models are correctly run."""
        conv = ConvergenceAnalysis(
            model_class=time_dependent_model,
            model_params={},
            levels=2,
            in_space=True,
            spatial_refinement_rate=2,
            in_time=True,
            temporal_refinement_rate=4,
        )
        results = conv.run_analysis()
        assert results[0].error_var_0 == 0.25
        assert results[0].error_var_1 == 1.0
        assert results[1].error_var_0 == 0.0625
        assert results[1].error_var_1 == 4.0


# -----> Tests for order of convergence (OOC)
@pytest.fixture(scope="class")
def list_of_results_space() -> list['ResultSimulation0', 'ResultSimulation1']:
    """List results for a spatial analysis.

    Note:
        We assume that this list of results was obtained with a static model,
        and that the convergence analysis took place in 2 levels, with a spatial
        refinement rate of 2.

    Returns:
        Mocked list of results in space.

    """
    @dataclass
    class ResultsSimulation0:
        """Data class for the first simulation."""
        error_var_0: float = 10  # error associated with variable 0
        error_var_1: float = 20  # error associated with variable 1
        var_0: float = 42  # value of the variable 0
        var_1: float = 24  # value of the variable 1
        cell_diameter: float = 0.5  # cell diameter of the grid
        num_dofs: int = 4  # number of degrees of freedom

    @dataclass
    class ResultsSimulation1:
        """Data class for the second simulation."""
        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        cell_diameter: float = 0.25  # cell diameter of the grid
        num_dofs: int = 8  # number of degrees of freedom

    return [ResultsSimulation0, ResultsSimulation1]


@pytest.fixture(scope="class")
def list_of_results_time() -> list['ResultSimulation0', 'ResultSimulation1']:
    """List results for a spatio-temporal analysis.

    Note:
        We assume that this list of results was obtained with a time-dependent model,
        and that the convergence analysis took place in 2 levels, with a temporal
        refinement rate of 4.

    Returns:
        Mocked list of results in time.

    """
    @dataclass
    class ResultsSimulation0:
        error_var_0: float = 10  # error associated with variable 0
        error_var_1: float = 20  # error associated with variable 1
        var_0: float = 42  # value of the variable 0
        var_1: float = 24  # value of the variable 1
        dt: float = 1.0  # time step of the simulation
        cell_diameter: float = 0.5  # cell diameter of the grid
        num_dofs: int = 4  # number of degrees of freedom

    @dataclass
    class ResultsSimulation1:
        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        dt: float = 0.5  # time step of the simulation
        cell_diameter: float = 0.5  # cell diameter of the grid
        num_dofs: int = 4   # number of degrees of freedom

    return [ResultsSimulation0, ResultsSimulation1]


@pytest.fixture(scope="class")
def list_of_results_space_time() -> list['ResultSimulation0', 'ResultSimulation1']:
    """List results for a spatio-temporal analysis.

    Note:
        We assume that this list of results was obtained with a time-dependent model,
        and that the convergence analysis took place in 2 levels, with a spatial
        refinement rate of 2 and a temporal refinement of 4.

    Returns:
        Mocked list of results in time and space.

    """
    @dataclass
    class ResultsSimulation0:
        error_var_0: float = 10  # error associated with variable 0
        error_var_1: float = 20  # error associated with variable 1
        var_0: float = 42  # value of the variable 0
        var_1: float = 24  # value of the variable 1
        cell_diameter: float = 0.5
        dt: float = 1.0  # time step of the simulation
        num_dofs: int = 4   # number of degrees of freedom

    @dataclass
    class ResultsSimulation1:
        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        cell_diameter: float = 0.25
        dt: float = 0.25  # time step of the simulation
        num_dofs: int = 8   # number of degrees of freedom

    return [ResultsSimulation0, ResultsSimulation1]


@pytest.fixture(scope="class")
def list_of_results_for_ooc(
        list_of_results_space,
        list_of_results_time,
        list_of_results_space_time
) -> list:
    """Collect the list of results in a list."""
    return [
        list_of_results_space,
        list_of_results_time,
        list_of_results_space_time,
        list_of_results_space_time,
    ]


@pytest.fixture(scope="class")
def convergence_analysis_for_ooc(
        stationary_mock_model, time_dependent_mock_model
) -> list:
    """Collect the convergence analysis instances in a list."""
    conv_space = ConvergenceAnalysis(
        model_class=stationary_mock_model,
        model_params={},
        levels=2,
        in_space=True,
        in_time=False,
        spatial_refinement_rate=2,
        temporal_refinement_rate=1,
    )
    conv_time = ConvergenceAnalysis(
        model_class=time_dependent_mock_model,
        model_params={},
        levels=2,
        in_space=False,
        in_time=True,
        spatial_refinement_rate=1,
        temporal_refinement_rate=4,
    )
    conv_space_time = ConvergenceAnalysis(
        model_class=time_dependent_mock_model,
        model_params={},
        levels=2,
        in_space=True,
        in_time=True,
        spatial_refinement_rate=2,
        temporal_refinement_rate=4,
    )
    return [conv_space, conv_time, conv_space_time, conv_space_time]


class TestOrderOfConvergence:
    """Collection of tests to check that `order_of_convergence()` is working fine."""

    def test_if_variables_equal_to_none_pull_all_attributes_starting_with_error(
            self,
            list_of_results_space: list,
            stationary_mock_model: 'StationaryMockModel',
    ) -> None:
        """Check that all variables starting with "error_" are pulled if variables=None.

        Parameters:
            list_of_results_space: List of results.
            stationary_mock_model: Stationary mock model.

        """
        conv = ConvergenceAnalysis(
            model_class=stationary_mock_model,
            model_params={},
            levels=2,
            spatial_refinement_rate=2,
        )
        ooc = conv.order_of_convergence(list_of_results=list_of_results_space)
        assert len(ooc.keys()) == 2
        assert "ooc_var_0" in ooc.keys()
        assert "ooc_var_1" in ooc.keys()

    @pytest.mark.parametrize("var_name", ["error_var_0", "error_var_1"])
    def test_if_variables_pull_only_given_names(
        self,
        var_name: str,
        list_of_results_space: list,
        stationary_mock_model: 'StationaryMockModel',
    ) -> None:
        """Check only given variables are pulled from the list of results.

        Parameters:
            var_name: Name of the variable that has to be pulled from the list of
                results.
            list_of_results_space: List of results.
            stationary_mock_model: Stationary mock model.

        """
        conv = ConvergenceAnalysis(
            model_class=stationary_mock_model,
            model_params={},
            levels=2,
            spatial_refinement_rate=2,
        )
        ooc = conv.order_of_convergence(
            list_of_results=list_of_results_space,
            variables=[var_name],
        )
        assert len(ooc.keys()) == 1
        if var_name == "error_var_0":
            assert f"ooc_var_0" in ooc.keys()
        else:
            assert f"ooc_var_1" in ooc.keys()

    @pytest.mark.parametrize(
        "list_idx, conv_idx, x_axis, base_log_x, base_log_y",
        [
            (0, 0, "cell_diameter", 2, 2),
            (1, 1, "time_step", 4, 4),
            (2, 2, "cell_diameter", 2, 2),
            (3, 3, "time_step", 4, 2),
        ],
    )
    def test_order_of_convergence(
            self,
            list_idx: int,
            conv_idx: int,
            x_axis: Literal["cell_diameter", "time_step"],
            base_log_x: int,
            base_log_y: int,
            list_of_results_for_ooc: list[list],
            convergence_analysis_for_ooc: list[ConvergenceAnalysis],
    ) -> None:
        """Test order of convergence in space, time, and space-time.

        Note:
            The list of lists of results was carefully manufactured so that the order
            of convergence matches for the three type of analysis. Note that this
            might not be the case in real analyses.

            It is also worth mentioning that we obtain the same OOC for both
            spatio-temporal analyses regardless of the x-axis that we've chosen,
            PROVIDED that the bases of the logarithms reflect the refinement rates
            accordingly. In this case, we use a base 2 when we use "cell_diameter" as
            x-axis and base 4 when we use "time_step" as x-axis.

        Parameters:
            list_idx: Index acting on `list_of_results_for_ooc`.
            conv_idx: Index action on `convergence_analysis_for_ooc`.
            x_axis: Whether to use cell diameters or time steps to determine the OOC.
            base_log_x: Base of the logarithm for the x-data.
            base_log_y: Base of the logarithm for the y-data.
            list_of_results_for_ooc: List of lists of results.
            convergence_analysis_for_ooc: List of convergence analysis objects.

        """
        conv = convergence_analysis_for_ooc[conv_idx]
        results = list_of_results_for_ooc[list_idx]
        ooc = conv.order_of_convergence(
            list_of_results=results,
            x_axis=x_axis,
            base_log_x_axis=base_log_x,
            base_log_y_axis=base_log_y,
        )
        assert len(ooc.keys()) == 2
        np.testing.assert_almost_equal(ooc["ooc_var_0"], 1.0, decimal=10)
        np.testing.assert_almost_equal(ooc["ooc_var_1"], 2.0, decimal=10)


class TestExportErrors:
    """Collection of tests to check `run_simulations()` is working fine."""
    ...
