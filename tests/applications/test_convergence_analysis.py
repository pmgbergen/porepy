"""Module containing tests for the ConvergenceAnalysis class.

Tested functionality includes:
    - Default parameters at instantiation, sanity checks, and helper methods.
    - The `run_analysis()` method.
    - The `order_of_convergence()` method.
    - The `export_error_to_txt()` method.

"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.mdg_library import square_with_orthogonal_fractures
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.utils.txt_io import read_data_from_txt
from porepy.viz.data_saving_model_mixin import VerificationDataSaving


# -----> Fixtures that are required on a module level.
@pytest.fixture(scope="module")
def stationary_mock_model():
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
def time_dependent_mock_model():
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
def conv_analysis_in_space(stationary_mock_model) -> ConvergenceAnalysis:
    """Set an instance of ConvergenceAnalysis in space.

    Parameters:
        stationary_mock_model: Stationary mock model.

    Returns:
        Instantiated convergence analysis object for an analysis in space.

    """
    return ConvergenceAnalysis(
        model_class=stationary_mock_model,
        model_params={},
        levels=2,
        spatial_refinement_rate=2,
        temporal_refinement_rate=1,
    )


@pytest.fixture(scope="module")
def conv_analysis_in_time(time_dependent_mock_model) -> ConvergenceAnalysis:
    """Set an instance of ConvergenceAnalysis in time.

    Parameters:
        time_dependent_mock_model: Time-dependent mock model.

    Returns:
        Instantiated convergence analysis object for an analysis in time.

    """
    return ConvergenceAnalysis(
        model_class=time_dependent_mock_model,
        model_params={},
        levels=2,
        spatial_refinement_rate=1,
        temporal_refinement_rate=4,
    )


@pytest.fixture(scope="module")
def conv_analysis_in_space_and_time(time_dependent_mock_model) -> ConvergenceAnalysis:
    """Set an instance of ConvergenceAnalysis in space-time.

    Parameters:
        time_dependent_mock_model: Time-dependent mock model.

    Returns:
        Instantiated convergence analysis object for an analysis in time and space.

    """
    return ConvergenceAnalysis(
        model_class=time_dependent_mock_model,
        model_params={},
        levels=2,
        spatial_refinement_rate=2,
        temporal_refinement_rate=4,
    )


@pytest.fixture(scope="module")
def list_of_results_space() -> list:
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

    @dataclass
    class ResultsSimulation1:
        """Data class to store results from the second simulation."""

        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        cell_diameter: float = 0.25  # cell diameter of the grid

    return [ResultsSimulation0, ResultsSimulation1]


@pytest.fixture(scope="module")
def list_of_results_time() -> list:
    """List results for a temporal analysis.

    Note:
        We assume that this list of results was obtained with a time-dependent model,
        and that the convergence analysis took place in 2 levels, with a temporal
        refinement rate of 4.

    Returns:
        Mocked list of results in time.

    """

    @dataclass
    class ResultsSimulation0:
        """Data class to store results from the first simulation."""

        error_var_0: float = 10  # error associated with variable 0
        error_var_1: float = 20  # error associated with variable 1
        var_0: float = 42  # value of the variable 0
        var_1: float = 24  # value of the variable 1
        dt: float = 1.0  # time step of the simulation
        cell_diameter: float = 0.5  # cell diameter of the grid

    @dataclass
    class ResultsSimulation1:
        """Data class to store results from the second simulation."""

        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        dt: float = 0.5  # time step of the simulation
        cell_diameter: float = 0.5  # cell diameter of the grid

    return [ResultsSimulation0, ResultsSimulation1]


@pytest.fixture(scope="module")
def list_of_results_space_time() -> list:
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
        """Data class to store results from the first simulation."""

        error_var_0: float = 10  # error associated with variable 0
        error_var_1: float = 20  # error associated with variable 1
        var_0: float = 42  # value of the variable 0
        var_1: float = 24  # value of the variable 1
        cell_diameter: float = 0.5
        dt: float = 1.0  # time step of the simulation

    @dataclass
    class ResultsSimulation1:
        """Data class to store results from the second simulation."""

        error_var_0: float = 5  # error associated with variable 0
        error_var_1: float = 5  # error associated with variable 1
        var_0: float = 41  # value of the variable 0
        var_1: float = 23  # value of the variable 1
        cell_diameter: float = 0.25
        dt: float = 0.25  # time step of the simulation

    return [ResultsSimulation0, ResultsSimulation1]


# -----> TEST: Initialization, sanity checks, and helper methods.
class TestInstantiationSanityCheckAndHelperMethods:
    """The following tests are written to check the sanity of the input parameters."""

    def test_instantiation_stationary(self, stationary_mock_model) -> None:
        """Test initialization of attributes for a stationary mock model.

        Parameters:
            stationary_mock_model: Stationary mock model.

        """
        conv = ConvergenceAnalysis(model_class=stationary_mock_model, model_params={})
        assert conv.spatial_refinement_rate == 1
        assert conv.temporal_refinement_rate == 1
        assert conv.levels == 1
        assert not conv._is_time_dependent
        assert len(conv.model_params) == 1
        assert conv.model_params[0]["meshing_arguments"]["cell_size"] == 1.0

    @pytest.mark.parametrize(
        "spatial_rate, temporal_rate",
        [
            (-1, 1),
            (1, -1),
        ],
    )
    def test_error_raised_if_rates_smaller_than_one(
        self,
        spatial_rate: int,
        temporal_rate: int,
        stationary_mock_model,
    ) -> None:
        """Check that error is raised when rates are smaller than one.

        Parameters:
            spatial_rate: Spatial refinement rate.
            temporal_rate: Temporal refinement rate.
            stationary_mock_model: Stationary mock model.

        """
        msg = "Refinement rate cannot be less than 1."
        with pytest.raises(ValueError) as excinfo:
            ConvergenceAnalysis(
                model_class=stationary_mock_model,
                model_params={},
                spatial_refinement_rate=spatial_rate,
                temporal_refinement_rate=temporal_rate,
            )
        assert msg in str(excinfo.value)

    def test_warning_is_raised_when_both_rates_are_one(
        self,
        time_dependent_mock_model,
    ) -> None:
        """Check that warning is raised when both rates are equal to one.

        Parameters:
            time_dependent_mock_model: Time-dependent mock model.

        """
        msg = "No refinement (in space or time) will be performed."
        with pytest.warns() as record:
            ConvergenceAnalysis(
                model_class=time_dependent_mock_model,
                model_params={},
                spatial_refinement_rate=1,
                temporal_refinement_rate=1,
            )
        assert str(record[0].message) == msg

    def test_error_is_raised_when_temporal_rate_larger_than_one_for_stationary_model(
        self,
        stationary_mock_model,
    ) -> None:
        """Check that error is raised when temporal rate > 1 and model is stationary.

        Parameters:
            stationary_mock_model: Stationary mock model.

        """
        msg = "Analysis in time not available for stationary models."
        with pytest.raises(ValueError) as excinfo:
            ConvergenceAnalysis(
                model_class=stationary_mock_model,
                model_params={},
                temporal_refinement_rate=2,
            )
        assert msg in str(excinfo.value)

    def test_raise_error_when_non_constant_dt_used(
        self, time_dependent_mock_model
    ) -> None:
        """Test if an error is raised when a non-constant time step is used.

        Parameters:
            time_dependent_mock_model: Time-dependent mock model.

        """
        msg = "Analysis in time only supports constant time step."
        with pytest.raises(NotImplementedError) as excinfo:
            ConvergenceAnalysis(
                model_class=time_dependent_mock_model,
                model_params={"time_manager": pp.TimeManager([0, 1], 0.1, False)},
                spatial_refinement_rate=2,
                temporal_refinement_rate=2,
            )
        assert msg in str(excinfo.value)

    def test_get_list_of_meshing_arguments(self, stationary_mock_model) -> None:
        """Test if the list of mesh sizes is correctly obtained.

        Parameters:
            stationary_mock_model: Stationary mock model.

        """
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

    def test_get_list_of_time_managers(self, time_dependent_mock_model) -> None:
        """Test if the list of time managers is correctly obtained.

        Parameters:
            time_dependent_mock_model: Time-dependent mock model.

        """
        conv = ConvergenceAnalysis(
            model_class=time_dependent_mock_model,
            model_params={"time_manager": pp.TimeManager([0, 1], 0.2, True)},
            levels=4,
            temporal_refinement_rate=4,
        )
        known_time_steps = [0.2, 0.05, 0.0125, 0.003125]
        actual_time_steps: list[float] = []
        for param in deepcopy(conv.model_params):
            actual_time_steps.append(param["time_manager"].dt)
        np.testing.assert_array_almost_equal(known_time_steps, actual_time_steps)

    @pytest.mark.parametrize("variables", [None, ["error_var_0"], ["error_var_1"]])
    def test_filter_variables_from_list_of_results(
        self,
        variables: list[str] | None,
        conv_analysis_in_space: ConvergenceAnalysis,
        list_of_results_space: list,
    ) -> None:
        """Test if the variables are correctly filtered from the list of results."""
        names = conv_analysis_in_space._filter_variables_from_list_of_results(
            list_of_results=list_of_results_space,
            variables=variables,
        )
        if variables == ["error_var_0"]:
            assert names == ["error_var_0"]
        elif variables == ["error_var_1"]:
            assert names == ["error_var_1"]
        else:
            assert names == ["error_var_0", "error_var_1"]


# -----> TEST: The `run_analysis()` method.
@pytest.fixture(scope="class")
def stationary_model():
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


@pytest.fixture(scope="class")
def time_dependent_model():
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
            spatial_refinement_rate=2,
            temporal_refinement_rate=4,
        )
        results = conv.run_analysis()
        assert results[0].error_var_0 == 0.25
        assert results[0].error_var_1 == 1.0
        assert results[1].error_var_0 == 0.0625
        assert results[1].error_var_1 == 4.0


# -----> TEST: The `order_of_convergence` method.
@pytest.fixture(scope="class")
def list_of_results_for_ooc(
    list_of_results_space, list_of_results_time, list_of_results_space_time
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
    conv_analysis_in_space,
    conv_analysis_in_time,
    conv_analysis_in_space_and_time,
) -> list:
    """Collect the convergence analysis instances in a list."""
    return [
        conv_analysis_in_space,
        conv_analysis_in_time,
        conv_analysis_in_space_and_time,
        conv_analysis_in_space_and_time,
    ]


class TestOrderOfConvergence:
    """Collection of tests to check that `order_of_convergence()` is working fine."""

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
            spatio-temporal analyses regardless of the x-axis that we've chosen. This
            only holds because the bases of the logarithms that we've provided reflect
            the refinement rates accordingly. In this case, we use a base 2 when we
            use "cell_diameter" as x-axis and base 4 when we use "time_step" as x-axis.

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

    def test_order_of_convergence_with_reduced_range(
        self,
        stationary_mock_model,
    ) -> None:
        """Test order of convergence for a subset of the data.

        Parameters:
            stationary_mock_model: Stationary mock model.

        """

        @dataclass
        class MockDataClass:
            """Minimal data class to save error and cell diameter."""

            error_var: float
            cell_diameter: float

        # Create convergence analysis object
        conv = ConvergenceAnalysis(
            model_class=stationary_mock_model,
            model_params={},
            levels=4,
            spatial_refinement_rate=2,
        )

        # Set linear convergence only for the last two levels
        results = [
            MockDataClass(error_var=42, cell_diameter=1.0),
            MockDataClass(error_var=np.pi, cell_diameter=0.5),
            MockDataClass(error_var=1.0, cell_diameter=0.25),
            MockDataClass(error_var=0.5, cell_diameter=0.125),
        ]

        # Order of convergence for the reduced range (last two levels) should be 1.0
        ooc_reduced_range = conv.order_of_convergence(
            list_of_results=results,
            data_range=slice(-2, None, None),
        )

        assert np.isclose(ooc_reduced_range["ooc_var"], 1.0, 1e-10)


# -----> TEST: The `export_errors_to_txt` method.
class TestExportErrors:
    """Collection of tests to check if `export_errors_to_txt()` is working fine."""

    def test_export_errors_for_stationary_model(
        self,
        conv_analysis_in_space: ConvergenceAnalysis,
        list_of_results_space: list,
    ):
        """Test if all errors are exported correctly for a stationary model."""
        conv_analysis_in_space.export_errors_to_txt(list_of_results_space)
        read_data = read_data_from_txt("error_analysis.txt")

        assert len(read_data.keys()) == 3
        np.testing.assert_equal(read_data["cell_diameter"], np.array([0.5, 0.25]))
        np.testing.assert_equal(read_data["error_var_0"], np.array([10.0, 5.0]))
        np.testing.assert_equal(read_data["error_var_1"], np.array([20.0, 5.0]))

        os.remove("error_analysis.txt")

    def test_export_errors_for_time_dependent_model(
        self,
        conv_analysis_in_space_and_time: ConvergenceAnalysis,
        list_of_results_space_time: list,
    ):
        """Test if all errors are exported correctly for a time-dependent model."""
        conv_analysis_in_space_and_time.export_errors_to_txt(list_of_results_space_time)
        read_data = read_data_from_txt("error_analysis.txt")

        assert len(read_data.keys()) == 4
        np.testing.assert_equal(read_data["cell_diameter"], np.array([0.5, 0.25]))
        np.testing.assert_equal(read_data["time_step"], np.array([1.0, 0.25]))
        np.testing.assert_equal(read_data["error_var_0"], np.array([10.0, 5.0]))
        np.testing.assert_equal(read_data["error_var_1"], np.array([20.0, 5.0]))

        os.remove("error_analysis.txt")


@pytest.fixture(scope="module")
def grids() -> list[pp.Grid, pp.MortarGrid]:
    """Create a mixed-dimensional grid on a unit square with a single fracture.

    Returns:
        A list containing one subdomain grid (a Cartesian 2x2 grid) and one mortar
        grid (a one dimensional mortar grid with 4 mortar cells).

    """
    mdg, _ = square_with_orthogonal_fractures(
        grid_type="cartesian",
        meshing_args={"cell_size": 0.5},
        fracture_indices=[0],
        size=1.0,
    )
    return [mdg.subdomains()[0], mdg.interfaces()[0]]


@pytest.mark.parametrize("is_relative", [False, True])
@pytest.mark.parametrize(
    "is_sd, is_cc, is_scalar",
    [
        (True, True, True),  # subdomain scalar cell-centered quantity
        (True, False, True),  # subdomain scalar face-centered quantity
        (True, True, False),  # subdomain vector cell-centered quantity
        (True, False, False),  # subdomain vector face-centered quantity
        (False, True, True),  # interface scalar cell-centered quantity
        (False, True, False),  # interface vector cell-centered quantity
    ],
)
def test_l2_error(
    is_sd: bool,
    is_scalar: bool,
    is_cc: bool,
    is_relative: bool,
    grids: list[pp.Grid, pp.MortarGrid],
) -> None:
    """Test whether the discrete L2-error is computed correctly.

    The test sets arrays of ones as for the true array, and arrays of zeros for the
    approximate arrays. The absolute l2-error is thus the square root of the sum of
    the measure of each element (cell_volume when is_cc=True and face_area when
    is_cc=False) in each grid. The relative l2-error is always 1.0 in all cases.

    Parameters:
        is_sd: Whether the error should be evaluated in a subdomain grid. False
            implies evaluation in an interface grid.
        is_scalar: Whether the array is corresponds to a scalar quantity. False
            implies a vector quantity.
        is_cc: Whether the array is a cell-centered quantity. False implies a
            face-centered quantity.
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """

    # Retrieve grid
    if is_sd:
        grid = grids[0]  # subdomain grid
    else:
        grid = grids[1]  # interface grid

    # Retrieve number of degrees of freedom and set the true array
    if is_cc:
        if is_scalar:
            ndof = grid.num_cells
            true_l2_error = np.sqrt(np.sum(grid.cell_volumes))
        else:
            if is_sd:
                ndof = 2 * grid.num_cells
                true_l2_error = np.sqrt(2 * np.sum(grid.cell_volumes))
            else:
                ndof = grid.num_cells
                true_l2_error = np.sqrt(np.sum(grid.cell_volumes))
    else:
        if is_scalar:
            ndof = grid.num_faces
            true_l2_error = np.sqrt(np.sum(grid.face_areas))
        else:
            ndof = 2 * grid.num_faces
            true_l2_error = np.sqrt(2 * np.sum(grid.face_areas))

    # Compute actual error
    actual_l2_error = ConvergenceAnalysis.l2_error(
        grid=grid,
        true_array=np.ones(ndof),
        approx_array=np.zeros(ndof),
        is_cc=is_cc,
        is_scalar=is_scalar,
        relative=is_relative,
    )

    # Compare
    if not is_relative:
        assert np.isclose(actual_l2_error, true_l2_error)
    else:
        assert np.isclose(actual_l2_error, 1.0)


def test_l2_error_division_by_zero_error(grids: list[pp.Grid, pp.MortarGrid]) -> None:
    """Test whether a division by zero error is raised.

    This error should be raised when the denominator is zero while computing the
    relative error.

    Parameters:
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """
    msg = "Attempted division by zero."
    with pytest.raises(ZeroDivisionError) as excinfo:
        # Attempt to obtain L2-relative error with true array of zeros
        ConvergenceAnalysis.l2_error(
            grid=grids[0],
            true_array=np.zeros(4),
            approx_array=np.random.random(4),
            is_cc=True,
            is_scalar=True,
            relative=True,
        )
    assert msg in str(excinfo.value)


def test_l2_error_not_implemented_error(grids: list[pp.Grid, pp.MortarGrid]) -> None:
    """Test whether a not implemented error is raised.

    The error should be raised when a face-centered quantity is passed together with
    a mortar grid.

    Parameters:
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """
    msg = "Interface variables can only be cell-centered."
    with pytest.raises(NotImplementedError) as excinfo:
        # Attempt to compute the error for a face-centered quantity on a mortar grid
        ConvergenceAnalysis.l2_error(
            grid=grids[1],
            true_array=np.ones(6),
            approx_array=np.random.random(6),
            is_cc=False,
            is_scalar=True,
        )
    assert msg in str(excinfo.value)
