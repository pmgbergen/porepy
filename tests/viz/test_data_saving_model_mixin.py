"""Tests of functionality found within data_saving_model_mixin.py.

The following is covered:

- Test that only the specified exported times are exported.
- Test that the physical times are exported, and that there is a match between the
  specified schedule and all exported files.
- Test the compute_slip_tendency static method for a variety of cases, including edge
  cases such as zero normal traction, positive normal traction, and custom tolerances.

"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.momentum_balance import MomentumBalance
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting


class DataSavingModelMixinModel(SquareDomainOrthogonalFractures, MomentumBalance):
    """Model for testing data saving."""

    def write_pvd_and_vtu(self) -> None:
        """Logger for the times that are exported.

        This method is called for every time step that is to be exported. It is now
        converted to a logger, meaning that every time step that is to be exported is
        logged in the model attribute exported_times.

        """
        self.exported_times.append(self.time_manager.time)


@pytest.mark.parametrize(
    "times_to_export", [None, [], [0.0, 0.5, 0.6], [0.0, 0.2, 0.5, 0.4, 1.0]]
)
def test_export_chosen_times(times_to_export):
    """Testing if only exported times are exported.

    We test exporting of:
    * All time steps
    * No time steps
    * A selection of time steps in ascending order
    * A selection of time steps in random order

    """
    time_steps = 10
    tf = 1.0
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    model_params = {
        "time_manager": time_manager,
        "times_to_export": times_to_export,
    }

    model = DataSavingModelMixinModel(model_params)
    model.exported_times = []
    pp.run_time_dependent_model(model)

    # The actual test of exported times based on the log stored in model.exported_times:
    if times_to_export is None:
        times_to_export = np.linspace(0.0, tf, time_steps + 1)
        assert np.allclose(model.exported_times, times_to_export)
    else:
        assert np.allclose(model.exported_times, np.sort(times_to_export))


@pytest.mark.parametrize(
    "times_to_export, expected_times",
    [
        (
            None,
            [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
        ),
        ([0.0, 0.1, 0.2], [0.0, 0.1, 0.2]),
    ],
)
def test_exported_times_consistency_with_files(times_to_export, expected_times):
    """Verify consistency between exported times and visualization files.

    The test ensures that:
        * Exported times match the times listed in the `times.json` file.
        * Exported times align with the timesteps in the `data.pvd` file.
        * The correct number of visualization files are generated.
    """
    folder_name = "viz_test_data_saving"
    time_steps = 10
    tf = 0.25
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    params = {
        "time_manager": time_manager,
        "times_to_export": times_to_export,
        "folder_name": folder_name,
    }

    model = pp.SinglePhaseFlow(params)
    pp.run_time_dependent_model(model)

    # Read times.json to get time data.
    times_file = Path(folder_name) / "times.json"
    with open(times_file, "r") as f:
        times_data = json.load(f)

    # Compare exported times with the times in times.json.
    assert np.allclose(model.time_manager.exported_times, times_data["time"])

    # Compare exported times with the expected times.
    assert np.allclose(model.time_manager.exported_times, expected_times)

    # Check that the correct number of files are exported.
    # Parse the PVD file.
    pvd_file = Path(folder_name) / "data.pvd"
    tree = ET.parse(pvd_file)
    root = tree.getroot()

    # Extract unique timesteps.
    timesteps = set()
    for dataset in root.findall(".//DataSet"):
        timestep = dataset.get("timestep")
        timesteps.add(float(timestep))

    # Compare unique timesteps with times.json.
    assert np.allclose(sorted(timesteps), times_data["time"])


class TestComputeSlipTendency:
    """Test suite for the compute_slip_tendency static method."""

    def test_normal_contact(self):
        """Test slip tendency computation for normal contact conditions."""
        # 2D case: traction shape (2, 3) for 3 cells, one friction coefficient value for
        # each cell.
        traction = np.array([[3.0, 0.0, 1.0], [-10.0, -5.0, -2.0]])
        friction = np.array([0.6, 0.5, 0.4])

        result = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        expected = np.array(
            [
                -np.linalg.norm([3.0]) / (-10.0 * 0.6),  # 0.5
                -np.linalg.norm([0.0]) / (-5.0 * 0.5),  # 0.0
                -np.linalg.norm([1.0]) / (-2.0 * 0.4),  # 1.25
            ]
        )

        assert np.allclose(result, expected)

    def test_zero_normal_traction_gives_nan(self):
        """Test that zero normal traction results in NaN slip tendency."""
        # Open fracture: normal traction = 0
        traction = np.array([[1.0, 2.0], [0.0, -5.0]])  # First cell has zero normal
        friction = np.array([0.6, 0.5])

        result = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        # First cell should be NaN, second should be valid.
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isclose(result[1], -2.0 / (-5.0 * 0.5))

    def test_3d_case(self):
        """Test slip tendency computation in 3D."""
        # 3D case: traction shape (3, 2) for 2 cells
        # First cell: Shear = [3, 4], Normal = -10, friction = 0.6
        # Expected: -sqrt(3^2 + 4^2) / (-10 * 0.6) = -5 / -6 = 0.8333
        traction = np.array([[3.0, 1.0], [4.0, 2.0], [-10.0, -8.0]])
        friction = np.array([0.6, 0.4])

        result = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        expected = np.array(
            [
                -np.linalg.norm([3.0, 4.0]) / (-10.0 * 0.6),  # 5/6 = 0.8333...
                -np.linalg.norm([1.0, 2.0]) / (-8.0 * 0.4),  # sqrt(5)/3.2 = 0.6989...
            ]
        )

        assert np.allclose(result, expected)

    def test_no_input_modification(self):
        """Test that input arrays are not modified."""
        traction = np.array([[1.0, 2.0], [0.0, -5.0]])
        friction = np.array([0.6, 0.5])

        # Make copies to check against.
        traction_orig = traction.copy()
        friction_orig = friction.copy()

        _ = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        # Verify inputs weren't modified.
        assert np.array_equal(traction, traction_orig)
        assert np.array_equal(friction, friction_orig)

    def test_near_zero_normal_traction(self):
        """Test that near-zero normal traction (within tolerance) gives NaN."""
        # Use a value that is close to zero but not exactly zero
        traction = np.array([[1.0, 2.0], [1e-12, -5.0]])
        friction = np.array([0.6, 0.5])

        result = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        # Should be NaN due to np.isclose tolerance defaulting to 1e-8 in the method.
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_positive_normal_traction(self):
        """Test behavior with positive (tensile) normal traction.

        Note: This is non-physical for contact mechanics but mathematically valid.
        """
        traction = np.array([[3.0], [5.0]])  # Positive (tensile) normal traction
        friction = np.array([0.6])

        result = FractureDeformationExporting.compute_slip_tendency(traction, friction)

        # Result should be negative (non-physical but mathematically correct).
        expected = -3.0 / (5.0 * 0.6)
        assert np.isclose(result[0], expected)
        assert result[0] < 0  # Negative slip tendency

    def test_uniform_friction_coefficient(self):
        """Test with uniform friction coefficient across all cells."""
        traction = np.array([[1.0, 2.0, 3.0], [-2.0, -4.0, -6.0]])
        friction = 0.5  # Scalar - should be broadcast

        # Should work with scalar friction coefficient.
        friction_array = np.full(3, friction)
        result = FractureDeformationExporting.compute_slip_tendency(
            traction, friction_array
        )

        expected = -np.array([1.0, 2.0, 3.0]) / (-np.array([2.0, 4.0, 6.0]) * 0.5)
        assert np.allclose(result, expected)

    def test_custom_tolerance(self):
        """Test that custom tolerance parameter works correctly."""
        # Test with a value that is within default tolerance but outside custom one.
        traction = np.array([[1.0, 2.0, 3.0], [1e-6, 1e-10, -5.0]])
        friction = np.array([0.6, 0.5, 0.4])

        # With default tolerance (1e-8), first value should not be NaN.
        result_default = FractureDeformationExporting.compute_slip_tendency(
            traction, friction
        )
        assert not np.isnan(result_default[0])  # 1e-6 > 1e-8
        assert np.isnan(result_default[1])  # 1e-10 < 1e-8
        assert not np.isnan(result_default[2])

        # With stricter tolerance (1e-5), first value should be NaN
        result_strict = FractureDeformationExporting.compute_slip_tendency(
            traction, friction, atol=1e-5
        )
        assert np.isnan(result_strict[0])  # 1e-6 < 1e-5
        assert np.isnan(result_strict[1])
        assert not np.isnan(result_strict[2])

        # With looser tolerance (1e-12), second value should not be NaN.
        result_loose = FractureDeformationExporting.compute_slip_tendency(
            traction, friction, atol=1e-12
        )
        assert not np.isnan(result_loose[0])
        assert not np.isnan(result_loose[1])  # 1e-10 > 1e-12
        assert not np.isnan(result_loose[2])
