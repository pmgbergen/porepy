"""Unit tests for indexers used by the AD equation system."""

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.ad.indexers import (
    EquationIndexer,
    EquationOnDomain,
    VariableIndexer,
)


@pytest.fixture
def domains() -> tuple[pp.Grid, pp.Grid, pp.Grid]:
    """Three distinct domains for indexer tests."""
    return pp.CartGrid([1]), pp.CartGrid([2]), pp.CartGrid([3])


@pytest.fixture
def variables(
    domains: tuple[pp.Grid, pp.Grid, pp.Grid],
) -> tuple[pp.ad.Variable, pp.ad.Variable, pp.ad.Variable]:
    """Three distinct variables for indexer tests."""
    return (
        pp.ad.Variable("x", {"cells": 1}, domains[0]),
        pp.ad.Variable("y", {"cells": 1}, domains[1]),
        pp.ad.Variable("z", {"cells": 1}, domains[2]),
    )


def test_equation_on_domain_identity(
    domains: tuple[pp.Grid, pp.Grid, pp.Grid],
) -> None:
    """Equation identifiers compare and hash by equation name and domain."""
    identifier = EquationOnDomain("equation", domains[0])

    assert identifier == EquationOnDomain("equation", domains[0])
    assert identifier != EquationOnDomain("other_equation", domains[0])
    assert identifier != EquationOnDomain("equation", domains[1])
    assert {identifier: 1}[EquationOnDomain("equation", domains[0])] == 1


def test_variable_indexer_projection(
    variables: tuple[pp.ad.Variable, pp.ad.Variable, pp.ad.Variable],
) -> None:
    """Projection indices respect requested variable order."""
    x, y, _ = variables
    indexer = VariableIndexer(indices={x: np.array([0, 1]), y: np.array([2, 3, 4])})

    assert indexer.size == 5
    np.testing.assert_array_equal(
        indexer.projection_indices([y, x]), np.array([2, 3, 4, 0, 1])
    )
    empty_projection = indexer.projection_indices([])
    assert empty_projection.dtype == int
    assert empty_projection.size == 0


def test_variable_indexer_unknown_variable(
    variables: tuple[pp.ad.Variable, pp.ad.Variable, pp.ad.Variable],
) -> None:
    """Variable indexer operations reject variables outside the indexer."""
    x, _, unknown = variables
    indexer = VariableIndexer(indices={x: np.array([0])})

    with pytest.raises(ValueError, match="not known"):
        indexer.projection_indices([unknown])
    with pytest.raises(ValueError, match="not known"):
        indexer.construct_restricted_indexer([unknown])


def test_variable_indexer_restriction(
    variables: tuple[pp.ad.Variable, pp.ad.Variable, pp.ad.Variable],
) -> None:
    """Restricted indexers are contiguous and follow requested variable order."""
    x, y, _ = variables
    indexer = VariableIndexer(indices={x: np.array([4, 7]), y: np.array([10, 12, 13])})

    restricted = indexer.construct_restricted_indexer([y, x])

    assert list(restricted.indices) == [y, x]
    np.testing.assert_array_equal(restricted.indices[y], np.arange(3))
    np.testing.assert_array_equal(restricted.indices[x], np.arange(3, 5))
    assert restricted.size == 5


def test_variable_indexer_duplicate_restriction(
    variables: tuple[pp.ad.Variable, pp.ad.Variable, pp.ad.Variable],
) -> None:
    """A restricted indexer cannot represent the same variable more than once."""
    x, _, _ = variables
    indexer = VariableIndexer(indices={x: np.array([0, 1])})

    with pytest.raises(ValueError, match="duplicate"):
        indexer.construct_restricted_indexer([x, x])


def test_equation_indexer_offsets(
    domains: tuple[pp.Grid, pp.Grid, pp.Grid],
) -> None:
    """Equation indexers expose local and globally offset equation indices."""
    first, second, third = domains
    composition = {
        "equation_a": {
            first: np.array([0, 1]),
            second: np.array([2, 3, 4]),
        },
        "equation_b": {
            second: np.array([0]),
            third: np.array([1, 2]),
        },
    }

    indexer = EquationIndexer(equation_image_composition=composition)

    np.testing.assert_array_equal(
        indexer.equation_image_space_composition["equation_a"][first], [0, 1]
    )
    np.testing.assert_array_equal(
        indexer.equation_image_space_composition["equation_b"][third], [1, 2]
    )
    expected_keys = [
        EquationOnDomain("equation_a", first),
        EquationOnDomain("equation_a", second),
        EquationOnDomain("equation_b", second),
        EquationOnDomain("equation_b", third),
    ]
    assert list(indexer.indices) == expected_keys
    np.testing.assert_array_equal(indexer.indices[expected_keys[0]], [0, 1])
    np.testing.assert_array_equal(indexer.indices[expected_keys[1]], [2, 3, 4])
    np.testing.assert_array_equal(indexer.indices[expected_keys[2]], [5])
    np.testing.assert_array_equal(indexer.indices[expected_keys[3]], [6, 7])


def test_equation_indexer_restriction(
    domains: tuple[pp.Grid, pp.Grid, pp.Grid],
) -> None:
    """Restricted equation indices are local and follow the requested order."""
    first, second, third = domains
    equation_a_second = EquationOnDomain("equation_a", second)
    equation_b_second = EquationOnDomain("equation_b", second)
    equation_b_third = EquationOnDomain("equation_b", third)
    indexer = EquationIndexer(
        equation_image_composition={
            "equation_a": {first: np.arange(2), second: np.arange(2, 5)},
            "equation_b": {second: np.arange(1), third: np.arange(1, 3)},
        }
    )

    # Only equation b on sd 2, equation b on sd 3, equation a on sd 2.
    restricted = indexer.construct_restricted_indexer(
        [equation_b_second, equation_b_third, equation_a_second]
    )

    assert list(restricted.indices) == [
        equation_b_second,
        equation_b_third,
        equation_a_second,
    ]
    # Equation b on sd 2 should have a single dof.
    np.testing.assert_array_equal(restricted.indices[equation_b_second], np.arange(1))
    # Equation b on sd 3 should have two dofs: [1, 2].
    np.testing.assert_array_equal(restricted.indices[equation_b_third], np.arange(1, 3))
    # Equation a on sd 2 should have 3 dofs: [3, 4, 5].
    np.testing.assert_array_equal(
        restricted.indices[equation_a_second], np.arange(3, 6)
    )
    np.testing.assert_array_equal(
        restricted.equation_image_space_composition["equation_b"][third],
        np.arange(1, 3),
    )
    np.testing.assert_array_equal(
        restricted.equation_image_space_composition["equation_a"][second],
        np.arange(3),
    )


def test_empty_indexers() -> None:
    """Indexers support empty systems and restrictions."""
    variable_indexer = VariableIndexer(indices={})
    restricted_indexer = variable_indexer.construct_restricted_indexer([])
    equation_indexer = EquationIndexer(equation_image_composition={})

    assert variable_indexer.size == 0
    assert variable_indexer.indices == {}
    assert variable_indexer.projection_indices([]).size == 0
    assert restricted_indexer.size == 0
    assert restricted_indexer.indices == {}
    assert equation_indexer.indices == {}
    assert equation_indexer.equation_image_space_composition == {}
