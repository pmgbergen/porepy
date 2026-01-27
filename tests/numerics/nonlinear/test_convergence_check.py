"""Unit tests for convergence status classes and convergence/divergence criteria."""

import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp

# Import "non-public" classes tests in convergence_check.
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


def test_convergence_status_methods():
    """Test the status check methods of ConvergenceStatus enum members."""
    s = ConvergenceStatus

    # Make sure each category method works
    assert s.CONVERGED.is_converged()
    assert s.NOT_CONVERGED.is_not_converged()
    assert s.DIVERGED.is_diverged()
    assert s.CYCLED.is_cycled()
    assert s.STAGNATED.is_stagnated()
    assert s.NAN.is_nan()
    assert s.MAX_ITERATIONS_REACHED.is_max_iterations_reached()
    assert s.STOPPED.is_stopped()


@pytest.mark.parametrize(
    "status",
    [
        ConvergenceStatus.CONVERGED,
        ConvergenceStatus.NOT_CONVERGED,
        ConvergenceStatus.STOPPED,
    ],
)
def test_convergence_status_not_failed(status):
    """Test the negative result of is_failed method of ConvergenceStatus members."""
    assert not status.is_failed()


@pytest.mark.parametrize(
    "status",
    [
        ConvergenceStatus.DIVERGED,
        ConvergenceStatus.CYCLED,
        ConvergenceStatus.STAGNATED,
        ConvergenceStatus.NAN,
        ConvergenceStatus.MAX_ITERATIONS_REACHED,
    ],
)
def test_convergence_status_failed(status):
    """Test the positive result of is_failed method of ConvergenceStatus members."""
    assert status.is_failed()


def test_convergence_status_str():
    """Test the string representation of ConvergenceStatus enum members."""
    s = ConvergenceStatus
    assert str(s.CONVERGED) == "converged"
    assert str(s.NOT_CONVERGED) == "not_converged"
    assert str(s.DIVERGED) == "diverged"
    assert str(s.CYCLED) == "cycled"
    assert str(s.STAGNATED) == "stagnated"
    assert str(s.NAN) == "nan"
    assert str(s.MAX_ITERATIONS_REACHED) == "max_iterations_reached"
    assert str(s.STOPPED) == "stopped"


@pytest.mark.parametrize(
    "c1, c2, c3, expected_status",
    [
        # All converged
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            [True, False, False, False, False, False, False, False, False],
        ),
        # All not converged
        (
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, False, False, False, False],
        ),
        # All diverged
        (
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.DIVERGED,
            [False, False, True, False, False, False, False, False, True],
        ),
        # Mixed: converged, converged, not converged
        (
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, False, False, False, False],
        ),
        # Mixed: not converged, converged, not onverged
        (
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, False, False, False, False],
        ),
        # Mixed: diverged, converged, nc
        (
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, True, False, False, False, False, False, True],
        ),
        # Mixed: cycled, c, nc
        (
            ConvergenceStatus.CYCLED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, True, False, False, False, False, True],
        ),
        # Mixed: stagnated, c, nc
        (
            ConvergenceStatus.STAGNATED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, True, False, False, False, True],
        ),
        # Mixed: nan, c, nc
        (
            ConvergenceStatus.NAN,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, True, False, False, True],
        ),
        # Mixed: max iterations reached, c, nc
        (
            ConvergenceStatus.MAX_ITERATIONS_REACHED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, False, True, False, True],
        ),
        # Mixed: stopped, c, nc
        (
            ConvergenceStatus.STOPPED,
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            [False, True, False, False, False, False, False, True, False],
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
        collection.is_cycled(),
        collection.is_stagnated(),
        collection.is_nan(),
        collection.is_max_iterations_reached(),
        collection.is_stopped(),
        collection.is_failed(),
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


# Test general divergence and convergence criteria.
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
    ("CriterionClass", "key"),
    [
        (AbsoluteConvergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteCriterion, "increment"),
        (pp.ResidualBasedAbsoluteCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_status", "expected_info"),
    [
        ([1e-4], ConvergenceStatus.CONVERGED, 1e-4),
        ([1e-2, 1e-2], ConvergenceStatus.NOT_CONVERGED, np.sqrt(2) * 1e-2),
    ],
)
def test_absolute_convergence_criterion_single(
    CriterionClass, key, value, expected_status, expected_info
):
    """Test of the general AbsoluteConvergenceCriterion with dict metric."""
    crit = CriterionClass(tol=1e-3, metric=lambda x: np.linalg.norm(x))
    status, info = crit.check(**{key: np.array(value)})
    assert status == expected_status
    assert np.isclose(info, expected_info)


@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (AbsoluteConvergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteCriterion, "increment"),
        (pp.ResidualBasedAbsoluteCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_status", "expected_info"),
    [
        ([1e-4], ConvergenceStatus.CONVERGED, 1e-4),
        ([1e-2, 1e-2], ConvergenceStatus.NOT_CONVERGED, np.sqrt(2) * 1e-2),
    ],
)
def test_absolute_convergence_criterion_dict(
    CriterionClass, key, value, expected_status, expected_info
):
    """Test of the general AbsoluteConvergenceCriterion with dict metric."""
    crit = CriterionClass(
        tol=1e-3,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )
    status, info = crit.check(**{key: np.array(value)})
    assert status == expected_status
    assert np.isclose(info["a"], expected_info)
    assert np.isclose(info["b"], 2 * expected_info)


@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (AbsoluteDivergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteDivergenceCriterion, "increment"),
        (pp.ResidualBasedAbsoluteDivergenceCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_status"),
    [
        ([1e-4], ConvergenceStatus.CONVERGED),
        ([1e-2, 1e-2], ConvergenceStatus.DIVERGED),
    ],
)
def test_absolute_divergence_criterion_single(
    CriterionClass, key, value, expected_status
):
    """Test of the general AbsoluteDivergenceCriterion with scalar metric."""
    crit = CriterionClass(tol=1e-3, metric=lambda x: np.linalg.norm(x))
    status = crit.check(**{key: np.array(value)})
    assert status == expected_status


@pytest.mark.parametrize(
    ("CriterionClass", "key"),
    [
        (AbsoluteDivergenceCriterion, "value"),
        (pp.IncrementBasedAbsoluteDivergenceCriterion, "increment"),
        (pp.ResidualBasedAbsoluteDivergenceCriterion, "residual"),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_status"),
    [
        ([1e-4], ConvergenceStatus.CONVERGED),
        ([1e-2, 1e-2], ConvergenceStatus.DIVERGED),
    ],
)
def test_absolute_divergence_criterion_dict(
    CriterionClass, key, value, expected_status
):
    """Test of the general AbsoluteDivergenceCriterion with dict metric."""
    crit = CriterionClass(
        tol=1e-3,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )
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
    ("value", "reference_value", "expected_status", "expected_info"),
    [
        (
            [1e-5],
            1e-2,
            ConvergenceStatus.CONVERGED,
            1e-3,
        ),  # rel = 1e-5/1e-2 = 1e-3 < tol
        (
            [1e-2, 1e-2],
            1e-1,
            ConvergenceStatus.NOT_CONVERGED,
            np.sqrt(2) * 1e-1,
        ),  # rel = sqrt(2)*1e-2/1e-1 = ~0.014 > tol
    ],
)
def test_relative_convergence_criterion_single(
    CriterionClass,
    key,
    reference_key,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the general RelativeConvergenceCriterion with scalar metric."""
    crit = CriterionClass(tol=1e-2, metric=lambda x: np.linalg.norm(x))
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
    ("value", "reference_value", "expected_status", "expected_info"),
    [
        ([1e-5], [1e-2], ConvergenceStatus.CONVERGED, 1e-3),
        (
            [1e-2, 1e-2],
            [1e-1, 1e-1],
            ConvergenceStatus.NOT_CONVERGED,
            1e-1,
        ),
    ],
)
def test_relative_convergence_criterion_dict(
    CriterionClass,
    key,
    reference_key,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the general RelativeConvergenceCriterion with dict metric."""
    crit = CriterionClass(
        tol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": np.linalg.norm(x) ** 2},
    )
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    print(status, info)
    assert status == expected_status
    assert np.isclose(info["a"], expected_info)
    assert np.isclose(info["b"], expected_info**2)


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
    ("value", "reference_value", "expected_status"),
    [
        ([1e-5], 1e-2, ConvergenceStatus.CONVERGED),  # rel = 0.001 < tol
        ([1e-2, 1e-2], 1e-1, ConvergenceStatus.DIVERGED),  # rel = ~0.014 > tol
    ],
)
def test_relative_divergence_criterion_single(
    CriterionClass, key, reference_key, value, reference_value, expected_status
):
    """Test of the general RelativeDivergenceCriterion with scalar metric."""
    crit = CriterionClass(tol=1e-2, metric=lambda x: np.linalg.norm(x))
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


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
    ("value", "reference_value", "expected_status"),
    [
        ([1e-5], [1e-2], ConvergenceStatus.CONVERGED),
        ([1e-2, 1e-2], [1e-1, 1e-1], ConvergenceStatus.DIVERGED),
    ],
)
def test_relative_divergence_criterion_dict(
    CriterionClass, key, reference_key, value, reference_value, expected_status
):
    """Test of the general RelativeDivergenceCriterion with dict metric."""
    crit = CriterionClass(
        tol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


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
def test_combined_convergence_criterion_single(
    CriterionClass,
    key,
    reference_key,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the CombinedConvergenceCriterion with scalar metric."""
    crit = CriterionClass(
        atol=1e-2,
        rtol=1e-2,
        metric=lambda x: np.linalg.norm(x),
    )
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status
    assert np.isclose(info, expected_info)


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
        ([1e-5], [1e-2], ConvergenceStatus.CONVERGED, 1e-5),
        (
            [1e-2, 1e-2],
            [1e-1, 1e-1],
            ConvergenceStatus.NOT_CONVERGED,
            np.sqrt(2) * 1e-2,
        ),
    ],
)
def test_combined_convergence_criterion_dict(
    CriterionClass,
    key,
    reference_key,
    value,
    reference_value,
    expected_status,
    expected_info,
):
    """Test of the CombinedConvergenceCriterion with dict metric."""
    crit = CriterionClass(
        atol=1e-2,
        rtol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )
    status, info = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status
    assert np.isclose(info["a"], expected_info)
    assert np.isclose(info["b"], 2 * expected_info)


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
def test_combined_divergence_criterion_single(
    CriterionClass, key, reference_key, value, reference_value, expected_status
):
    """Test of the CombinedDivergenceCriterion with scalar metric."""
    crit = CriterionClass(
        atol=1e-2,
        rtol=1e-2,
        metric=lambda x: np.linalg.norm(x),
    )
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


@pytest.mark.parametrize(
    ("CriterionClass", "key", "reference_key"),
    [
        # (CombinedDivergenceCriterion, "value", "reference"),
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
        ([1e-5], [1e-2], ConvergenceStatus.CONVERGED),
        ([1e-2, 1e-2], [1e-1, 1e-1], ConvergenceStatus.DIVERGED),
    ],
)
def test_combined_divergence_criterion_dict(
    CriterionClass, key, reference_key, value, reference_value, expected_status
):
    """Test of the CombinedDivergenceCriterion with dict metric."""
    crit = CriterionClass(
        atol=1e-2,
        rtol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )
    status = crit.check(**{key: np.array(value), reference_key: reference_value})
    assert status == expected_status


@pytest.mark.parametrize(
    ("iteration_index", "max_iterations", "expected_status"),
    [
        (-1, 3, ConvergenceStatus.CONVERGED),  # Before first iteration
        (0, 3, ConvergenceStatus.CONVERGED),  # First active iteration
        (1, 3, ConvergenceStatus.CONVERGED),  # Second active iteration
        (2, 3, ConvergenceStatus.DIVERGED),  # Third active iteration (max reached)
        (3, 3, ConvergenceStatus.DIVERGED),
    ],
)
def test_max_iterations_criterion(iteration_index, max_iterations, expected_status):
    """Test of the MaxIterationsCriterion."""
    crit = pp.MaxIterationsCriterion(max_iterations=max_iterations)
    status = crit.check(iteration_index=iteration_index)
    assert status == expected_status


# Test collection of criteria
@pytest.mark.parametrize(
    ("value", "expected_status_1", "expected_status_2", "expected_status"),
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
    value, expected_status_1, expected_status_2, expected_status
):
    """Test ConvergenceCriteria with multiple criteria."""
    # Create two simple absolute convergence criteria with different tolerances
    crit1 = AbsoluteConvergenceCriterion(tol=1e-3, metric=lambda x: np.linalg.norm(x))
    crit2 = AbsoluteConvergenceCriterion(
        tol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )

    # Create a collection
    criteria = pp.numerics.nonlinear.convergence_check.ConvergenceCriteria(
        {"crit1": crit1, "crit2": crit2}
    )

    status, info = criteria.check(value=np.array([value]))
    assert status["crit1"] == expected_status_1
    assert status["crit2"] == expected_status_2
    if expected_status == ConvergenceStatus.CONVERGED:
        assert status.is_converged()
    else:
        assert status.is_not_converged()
    assert np.isclose(info["crit1"], value)
    assert np.isclose(info["crit2"]["a"], value)
    assert np.isclose(info["crit2"]["b"], 2 * value)


@pytest.mark.parametrize(
    ("value", "expected_status_1", "expected_status_2", "expected_status"),
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
    value, expected_status_1, expected_status_2, expected_status
):
    """Test DivergenceCriteria with multiple criteria."""
    # Create two simple absolute divergence criteria with different tolerances
    crit1 = AbsoluteDivergenceCriterion(tol=1e-3, metric=lambda x: np.linalg.norm(x))
    crit2 = AbsoluteDivergenceCriterion(
        tol=1e-2,
        metric=lambda x: {"a": np.linalg.norm(x), "b": 2.0 * np.linalg.norm(x)},
    )

    # Create a collection
    criteria = pp.numerics.nonlinear.convergence_check.DivergenceCriteria(
        {"crit1": crit1, "crit2": crit2}
    )

    status = criteria.check(value=np.array([value]))
    assert status["crit1"] == expected_status_1
    assert status["crit2"] == expected_status_2
    if expected_status == ConvergenceStatus.CONVERGED:
        assert status.is_converged()
    else:
        assert status.is_failed()
