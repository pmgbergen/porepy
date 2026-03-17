"""Unit tests for status objects and convergence/divergence criteria.

Overview of tests:
- SimulationStatus and ConvergenceStatus enums and their check methods
- ConvergenceStatusCollection and ConvergenceStatusHistory for tracking status evolution
- ConvergenceInfoCollection and ConvergenceInfoHistory for tracking convergence metrics
- Absolute, relative, and combined convergence criteria (single and multiphysics)
- Absolute, relative, and combined divergence criteria (single and multiphysics)
- NanDivergenceCriterion and MaxIterationsCriterion
- Collections of convergence and divergence criteria

"""

import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp

# Import "non-public" classes.
from porepy.numerics.nonlinear.convergence_check import (
    AbsoluteConvergenceCriterion,
    AbsoluteDivergenceCriterion,
    CombinedConvergenceCriterion,
    CombinedDivergenceCriterion,
    ConvergenceInfoCollection,
    ConvergenceInfoHistory,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    ConvergenceStatusHistory,
    NanDivergenceCriterion,
    RelativeConvergenceCriterion,
    RelativeDivergenceCriterion,
    SimulationStatus,
)


# Reused custom metrics for convergence criteria tests.
def single_physics_metric():
    """Immitate single physics metric like pp.EuclideanMetric."""
    return lambda x: np.linalg.norm(x)


def multiphysics_metric():
    """Immitate dict-based metric like pp.VariableBasedEuclideanMetric."""
    return lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)}


def multiphysics_check(info, expected_value):
    """Check dict-based metric info for multiphysics criteria."""
    return np.isclose(info["a"], expected_value) and np.isclose(
        info["b"], 2.0 * expected_value
    )


# ! ---- SIMULATION STATUS TESTS ---- !


def test_simulation_status_methods():
    """Test the status check methods of SimulationStatus enum members."""
    s = SimulationStatus
    assert s.IN_PROGRESS.is_in_progress()
    assert s.SUCCESSFUL.is_successful()
    assert s.FAILED.is_failed()
    assert s.STOPPED.is_stopped()

    assert not s.IN_PROGRESS.is_successful()
    assert not s.IN_PROGRESS.is_failed()
    assert not s.IN_PROGRESS.is_stopped()

    assert not s.SUCCESSFUL.is_in_progress()
    assert not s.SUCCESSFUL.is_failed()
    assert not s.SUCCESSFUL.is_stopped()

    assert not s.FAILED.is_in_progress()
    assert not s.FAILED.is_successful()
    assert not s.FAILED.is_stopped()

    assert not s.STOPPED.is_in_progress()
    assert not s.STOPPED.is_successful()
    assert not s.STOPPED.is_failed()


def test_simulation_status_str():
    """Test the string representation of SimulationStatus enum members."""
    s = SimulationStatus
    assert str(s.IN_PROGRESS) == "in_progress"
    assert str(s.SUCCESSFUL) == "successful"
    assert str(s.FAILED) == "failed"
    assert str(s.STOPPED) == "stopped"


# ! ---- CONVERGENCE STATUS TESTS ---- !


def test_convergence_status_methods():
    """Test the status check methods of ConvergenceStatus enum members."""
    s = ConvergenceStatus

    assert s.CONVERGED.is_converged()
    assert s.NOT_CONVERGED.is_not_converged()
    assert s.DIVERGED.is_diverged()

    assert not s.CONVERGED.is_not_converged()
    assert not s.CONVERGED.is_diverged()

    assert not s.NOT_CONVERGED.is_converged()
    assert not s.NOT_CONVERGED.is_diverged()

    assert not s.DIVERGED.is_converged()
    assert not s.DIVERGED.is_not_converged()


def test_convergence_status_str():
    """Test the string representation of ConvergenceStatus enum members."""
    s = ConvergenceStatus
    assert str(s.CONVERGED) == "converged"
    assert str(s.NOT_CONVERGED) == "not_converged"
    assert str(s.DIVERGED) == "diverged"


@pytest.mark.parametrize(
    "c1, c2, c3, expected_status",
    [
        # All converged
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            [True, False, False],
        ),
        # All not converged
        (
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False],
        ),
        # All diverged
        (
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
            [False, False, True],
        ),
        # Mixed: converged, converged, not converged
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False],
        ),
        # Mixed: not converged, converged, not onverged
        (
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False],
        ),
        # Mixed: diverged, converged, nc
        (
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, True],
        ),
    ],
)
def test_convergence_status_collection_parametrized(c1, c2, c3, expected_status):
    """Test behavior for prescribed collections of ConvergenceStatus objects."""

    collection = ConvergenceStatusCollection({"crit1": c1, "crit2": c2, "crit3": c3})

    status = [
        collection.is_converged(),
        collection.is_not_converged(),
        collection.is_diverged(),
    ]

    assert status == expected_status


def test_convergence_status_collection_union_with_overlap():
    """Test behavior for union of ConvergenceStatusCollection objects with overlap."""
    c1 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.CONVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    c2 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.DIVERGED,
            "crit3": ConvergenceStatus.CONVERGED,
        }
    )

    try:
        _ = c1.union(c2)
        is_assertion_error = False
    except AssertionError:
        is_assertion_error = True

    assert is_assertion_error, "Expected an AssertionError due to overlapping keys."


def test_convergence_status_collection_union_without_overlap():
    """Test union of ConvergenceStatusCollection objects without overlap."""
    c1 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.CONVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    c2 = ConvergenceStatusCollection(
        {
            "crit3": ConvergenceStatus.DIVERGED,
            "crit4": ConvergenceStatus.CONVERGED,
        }
    )

    union = c1.union(c2)

    for crit in ["crit1", "crit2", "crit3", "crit4"]:
        assert crit in union

    # NOTE: No need to check values here, as those are covered in previous tests.


def test_convergence_status_history_append():
    """Test the correct recursive append of ConvergenceStatusCollection objects."""

    # Start with empty and check structure.
    history = ConvergenceStatusHistory()
    assert len(history.keys()) == 0

    # Add first entry and check structure.
    c1 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.CONVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    history.append(c1)
    assert len(history.keys()) == 2
    for key in history:
        assert isinstance(history[key], list)
        assert len(history[key]) == 1

    # Add second entry and check structure.
    c2 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.DIVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    history.append(c2)
    assert len(history.keys()) == 2
    for key in history:
        assert isinstance(history[key], list)
        assert len(history[key]) == 2

    # Check values.
    assert history["crit1"] == [ConvergenceStatus.CONVERGED, ConvergenceStatus.DIVERGED]
    assert history["crit2"] == [
        ConvergenceStatus.NOT_CONVERGED,
        ConvergenceStatus.NOT_CONVERGED,
    ]


def test_convergence_history_to_str():
    """Test the string representation of ConvergenceStatusHistory."""
    history = ConvergenceStatusHistory()
    c1 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.CONVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    c2 = ConvergenceStatusCollection(
        {
            "crit1": ConvergenceStatus.DIVERGED,
            "crit2": ConvergenceStatus.NOT_CONVERGED,
        }
    )
    history.append(c1)
    history.append(c2)

    # Get string representation.
    history_str = history.to_str()

    # Check values.
    assert history_str["crit1"] == ["converged", "diverged"]
    assert history_str["crit2"] == [
        "not_converged",
        "not_converged",
    ]


# ! ---- CONVERGENCE INFO TESTS ---- !


def test_convergence_info_history():
    """Same as for ConvergenceStatusHistory, but for ConvergenceInfoHistory."""

    # Start with empty and check structure.
    history = ConvergenceInfoHistory()
    assert len(history.keys()) == 0

    # Add first entry and check structure.
    c1 = ConvergenceInfoCollection(
        {
            "crit1": 1.0,  # Criterion with float output.
            "crit2": {
                "v1": 2.0,
                "v2": 1.0,
            },  # Criterion with dict output (e.g., per variable).
        }
    )
    history.append(c1)
    assert len(history.keys()) == 2
    assert len(history["crit1"]) == 1
    for key in history["crit2"]:
        assert len(history["crit2"][key]) == 1

    # Add second entry and check structure.
    c2 = ConvergenceInfoCollection(
        {
            "crit1": 0.1,
            "crit2": {"v1": 0.2, "v2": 0.1},
        }
    )
    history.append(c2)
    assert len(history.keys()) == 2
    assert len(history["crit1"]) == 2
    for key in history["crit2"]:
        assert len(history["crit2"][key]) == 2

    # Check values.
    assert np.allclose(history["crit1"], [1.0, 0.1])
    assert DeepDiff(history["crit2"], {"v1": [2.0, 0.2], "v2": [1.0, 0.1]}) == {}


# ! ---- CRITERIA TESTS INDEPENDENT OF METRICS ---- !


@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (NanDivergenceCriterion, "value"),
        (pp.IncrementBasedNanCriterion, "increment"),
        (pp.ResidualBasedNanCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_status"),
    [(1.0, ConvergenceStatus.CONVERGED), (np.nan, ConvergenceStatus.DIVERGED)],
)
def test_nan_divergence_criterion(CriterionClass, key, value, expected_status):
    """Test of the general NanDivergenceCriterion."""
    crit = CriterionClass()
    status = crit.check(**{key: np.array([value])})
    assert status == expected_status


@pytest.mark.parametrize(
    ("iteration_index", "max_iterations", "expected_status"),
    [
        (0, 3, ConvergenceStatus.CONVERGED),  # Before first iteration
        (1, 3, ConvergenceStatus.CONVERGED),  # First active iteration
        (2, 3, ConvergenceStatus.CONVERGED),  # Second active iteration
        (3, 3, ConvergenceStatus.DIVERGED),  # Third active iteration (max reached)
        (4, 3, ConvergenceStatus.DIVERGED),
    ],
)
def test_max_iterations_criterion(iteration_index, max_iterations, expected_status):
    """Test of the MaxIterationsCriterion."""
    crit = pp.MaxIterationsCriterion(max_iterations=max_iterations)
    status = crit.check(num_iterations=iteration_index)
    assert status == expected_status


# ! ---- CRITERIA TESTS USING A SINGLE METRIC ---- !


@pytest.mark.parametrize(
    "metric",
    [
        single_physics_metric,
        multiphysics_metric,
    ],
)
@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (AbsoluteConvergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteCriterion, "increment"),
        (pp.ResidualBasedAbsoluteCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("tol", "value", "expected_status", "expected_info"),
    [
        (1e-3, [1e-4], ConvergenceStatus.CONVERGED, 1e-4),
        (1e-3, [1e-2, 1e-2], ConvergenceStatus.NOT_CONVERGED, np.sqrt(2) * 1e-2),
    ],
)
def test_absolute_convergence_criterion(
    metric, CriterionClass, key, tol, value, expected_status, expected_info
):
    """Test of the general AbsoluteConvergenceCriterion."""
    crit = CriterionClass(tol=tol, metric=metric())
    status, info = crit.check(**{key: np.array(value)})
    assert status == expected_status
    if isinstance(info, dict):
        # Tailor check to multiphysics metric.
        assert multiphysics_check(info, expected_info)
    else:
        # Tailor check to single_physics_metric.
        assert np.isclose(info, expected_info)


@pytest.mark.parametrize(
    "metric",
    [
        single_physics_metric,
        multiphysics_metric,
    ],
)
@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (AbsoluteDivergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteDivergenceCriterion, "increment"),
        (pp.ResidualBasedAbsoluteDivergenceCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("tol", "value", "expected_status"),
    [
        (1e-3, [1e-4], ConvergenceStatus.CONVERGED),
        (1e-3, [1e-2, 1e-2], ConvergenceStatus.DIVERGED),
    ],
)
def test_absolute_divergence_criterion(
    metric, CriterionClass, key, value, tol, expected_status
):
    """Test of the general AbsoluteDivergenceCriterion."""
    crit = CriterionClass(tol=tol, metric=metric())
    status = crit.check(**{key: np.array(value)})
    assert status == expected_status


@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        (RelativeConvergenceCriterion, "value", "reference"),
        (pp.IncrementBasedRelativeCriterion, "increment", "reference_increment"),
        (pp.ResidualBasedRelativeCriterion, "residual", "reference_residual"),
    ],
)
@pytest.mark.parametrize(
    ("tol", "value", "reference_value", "expected_status", "expected_info"),
    [
        (
            1e-2,
            [1e-5],
            1e-2,
            ConvergenceStatus.CONVERGED,
            1e-3,
        ),  # rel = 1e-5/1e-2 = 1e-3 < tol
        (
            1e-2,
            [1e-2, 1e-2],
            1e-1,
            ConvergenceStatus.NOT_CONVERGED,
            np.sqrt(2) * 1e-1,
        ),  # rel = sqrt(2)*1e-2/1e-1 = ~0.014 > tol
    ],
)
def test_relative_convergence_criterion_single_physics(
    CriterionClass,
    key,
    reference_key,
    tol,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the general RelativeConvergenceCriterion with a single physics metric."""
    crit = CriterionClass(tol=tol, metric=single_physics_metric())
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status
    assert np.isclose(info, expected_info)


@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        (RelativeConvergenceCriterion, "value", "reference"),
        (pp.IncrementBasedRelativeCriterion, "increment", "reference_increment"),
        (pp.ResidualBasedRelativeCriterion, "residual", "reference_residual"),
    ],
)
@pytest.mark.parametrize(
    ("tol", "value", "reference_value", "expected_status", "expected_info"),
    [
        (1e-2, [1e-5], 1e-2, ConvergenceStatus.CONVERGED, 1e-3),
        (
            1e-2,
            [1e-2, 1e-2],
            [1e-1, 1e-1],
            ConvergenceStatus.NOT_CONVERGED,
            1e-1,
        ),
    ],
)
def test_relative_convergence_criterion_multiphysics(
    CriterionClass,
    key,
    reference_key,
    tol,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the general RelativeConvergenceCriterion for a multiphysics metric."""
    crit = CriterionClass(tol=tol, metric=multiphysics_metric())
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status
    # Do not use the multiphysics check here, as the relative norm cancels the scaling.
    assert np.isclose(info["a"], expected_info)
    assert np.isclose(info["b"], expected_info)


@pytest.mark.parametrize(
    "metric",
    [
        single_physics_metric,
        multiphysics_metric,
    ],
)
@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        (RelativeDivergenceCriterion, "value", "reference"),
        (
            pp.IncrementBasedRelativeDivergenceCriterion,
            "increment",
            "reference_increment",
        ),
        (pp.ResidualBasedRelativeDivergenceCriterion, "residual", "reference_residual"),
    ],
)
@pytest.mark.parametrize(
    ("tol", "value", "reference_value", "expected_status"),
    [
        (1e-2, [1e-5], 1e-2, ConvergenceStatus.CONVERGED),  # rel = 0.001 < tol
        (1e-2, [1e-2, 1e-2], 1e-1, ConvergenceStatus.DIVERGED),  # rel = ~0.014 > tol
    ],
)
def test_relative_divergence_criterion(
    metric,
    CriterionClass,
    key,
    reference_key,
    tol,
    value,
    reference_value,
    expected_status,
):
    """Test of the general RelativeDivergenceCriterion."""
    crit = CriterionClass(tol=tol, metric=metric())
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


@pytest.mark.parametrize(
    "metric",
    [
        single_physics_metric,
        multiphysics_metric,
    ],
)
@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        (CombinedConvergenceCriterion, "value", "reference"),
        (pp.IncrementBasedCombinedCriterion, "increment", "reference_increment"),
        (pp.ResidualBasedCombinedCriterion, "residual", "reference_residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "reference_value", "expected_status", "expected_info"),
    [
        ([1e-5], 1e-2, ConvergenceStatus.CONVERGED, 1e-5),
        ([1e-2, 1e-2], 1e-1, ConvergenceStatus.NOT_CONVERGED, np.sqrt(2) * 1e-2),
    ],
)
def test_combined_convergence_criterion(
    metric,
    CriterionClass,
    key,
    reference_key,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the CombinedConvergenceCriterion."""
    crit = CriterionClass(
        atol=1e-2,
        rtol=1e-2,
        metric=metric(),
    )
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status
    if isinstance(info, dict):
        # Tailor check to multiphysics metric.
        assert multiphysics_check(info, expected_info)
    else:
        # Tailor check to single_physics_metric.
        assert np.isclose(info, expected_info)


@pytest.mark.parametrize(
    "metric",
    [
        single_physics_metric,
        multiphysics_metric,
    ],
)
@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        (CombinedDivergenceCriterion, "value", "reference"),
        (
            pp.IncrementBasedCombinedDivergenceCriterion,
            "increment",
            "reference_increment",
        ),
        (pp.ResidualBasedCombinedDivergenceCriterion, "residual", "reference_residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "reference_value", "expected_status"),
    [
        ([1e-5], 1e-2, ConvergenceStatus.CONVERGED),
        ([1e-2, 1e-2], 1e-1, ConvergenceStatus.DIVERGED),
    ],
)
def test_combined_divergence_criterion(
    metric, CriterionClass, key, reference_key, value, reference_value, expected_status
):
    """Test of the CombinedDivergenceCriterion."""
    crit = CriterionClass(atol=1e-2, rtol=1e-2, metric=metric())
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


# ! ---- CRITERIA TESTS USING COLLECTION OF METRICS ---- !


@pytest.mark.parametrize(
    (
        "value",
        "expected_status_crit_single",
        "expected_status_crit_multi",
        "expected_status_collection",
    ),
    [
        (
            1e-4,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
        ),
        (
            2e-3,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
        ),
        (
            1e-2,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
        ),
    ],
)
def test_convergence_criteria_collection(
    value,
    expected_status_crit_single,
    expected_status_crit_multi,
    expected_status_collection,
):
    """Test ConvergenceCriteria with multiple criteria."""
    # Create two simple absolute convergence criteria with different tolerances.
    crit_single = AbsoluteConvergenceCriterion(tol=1e-3, metric=single_physics_metric())
    crit_multi = AbsoluteConvergenceCriterion(tol=1e-2, metric=multiphysics_metric())

    # Create a collection.
    criteria = pp.numerics.nonlinear.convergence_check.ConvergenceCriteria(
        {"crit_single": crit_single, "crit_multi": crit_multi}
    )

    status, info = criteria.check(value=np.array([value]))
    assert status["crit_single"] == expected_status_crit_single
    assert status["crit_multi"] == expected_status_crit_multi
    if expected_status_collection == ConvergenceStatus.CONVERGED:
        assert status.is_converged()
    else:
        assert status.is_not_converged()
    assert np.isclose(info["crit_single"], value)
    assert multiphysics_check(info["crit_multi"], value)


@pytest.mark.parametrize(
    (
        "value",
        "expected_status_crit_single",
        "expected_status_crit_multi",
        "expected_status_collection",
    ),
    [
        (
            1e-4,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
        ),
        (
            2e-3,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.DIVERGED,
        ),
        (
            1e-2,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
        ),
    ],
)
def test_divergence_criteria_collection(
    value,
    expected_status_crit_single,
    expected_status_crit_multi,
    expected_status_collection,
):
    """Test DivergenceCriteria with multiple criteria."""
    # Create two simple absolute divergence criteria with different tolerances
    crit_single = AbsoluteDivergenceCriterion(tol=1e-3, metric=single_physics_metric())
    crit_multi = AbsoluteDivergenceCriterion(tol=1e-2, metric=multiphysics_metric())

    # Create a collection
    criteria = pp.numerics.nonlinear.convergence_check.DivergenceCriteria(
        {"crit_single": crit_single, "crit_multi": crit_multi}
    )

    status = criteria.check(value=np.array([value]))
    assert status["crit_single"] == expected_status_crit_single
    assert status["crit_multi"] == expected_status_crit_multi
    if expected_status_collection == ConvergenceStatus.CONVERGED:
        assert status.is_converged()
    else:
        assert status.is_diverged()
