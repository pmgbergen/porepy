"""Unit tests for indexers used by the AD equation system."""

from copy import copy
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import pytest

import porepy as pp


def get_domains() -> list[pp.Grid]:
    """Three distinct domains for indexer tests."""
    return [pp.CartGrid([3]), pp.CartGrid([2]), pp.CartGrid([4])]


def get_variables(domains: list[pp.Grid]) -> list[pp.ad.Variable]:
    """Four distinct variables for indexer tests."""
    return [
        pp.ad.Variable("x", pp.ad.GridEntities(cells=1), domains[0]),
        pp.ad.Variable("x", pp.ad.GridEntities(cells=1), domains[1]),
        pp.ad.Variable("y", pp.ad.GridEntities(cells=1), domains[1]),
        pp.ad.Variable("y", pp.ad.GridEntities(cells=1), domains[2]),
    ]


def get_equations(domains: list[pp.Grid]) -> list[pp.ad.EquationOnDomain]:
    """Four distinct equations for indexer tests."""
    return [
        pp.ad.EquationOnDomain("x", domains[0]),
        pp.ad.EquationOnDomain("x", domains[1]),
        pp.ad.EquationOnDomain("y", domains[1]),
        pp.ad.EquationOnDomain("y", domains[2]),
    ]


def get_equation_indexer(
    indices: Optional[dict[pp.ad.EquationOnDomain, np.ndarray]] = None,
) -> pp.ad.EquationIndexer:
    """Construct equation indexer for the tests."""
    if indices is None:
        x0, x1, y1, y2 = get_equations(get_domains())
        indices = {
            x0: np.array([0, 1, 2]),
            x1: np.array([3, 4]),
            y1: np.array([5, 6]),
            y2: np.array([7, 8, 9, 10]),
        }
    return pp.ad.EquationIndexer(indices=indices)


def get_variable_indexer(
    indices: Optional[dict[pp.ad.Variable, np.ndarray]] = None,
) -> pp.ad.VariableIndexer:
    """Construct variable indexer for the tests."""
    if indices is None:
        x0, x1, y1, y2 = get_variables(get_domains())
        indices = {
            x0: np.array([0, 1, 2]),
            x1: np.array([3, 4]),
            y1: np.array([5, 6]),
            y2: np.array([7, 8, 9, 10]),
        }
    return pp.ad.VariableIndexer(indices=indices)


def get_indexer(
    indexer_type: str,
    indices: Optional[dict] = None,
) -> pp.ad.Indexer:
    """Construct indexer for the tests."""
    if indexer_type == "equation_indexer":
        return get_equation_indexer(indices=indices)
    elif indexer_type == "variable_indexer":
        return get_variable_indexer(indices=indices)
    raise ValueError(indexer_type)


def get_unknown_key(indexer_type: str):
    """Constructs a key that is not present in `indexer.indices`."""
    if indexer_type == "equation_indexer":
        return pp.ad.EquationOnDomain(name="unknown", domain=get_domains()[1])
    elif indexer_type == "variable_indexer":
        return pp.ad.Variable(
            name="unknown", ndof=pp.ad.GridEntities(cells=1), domain=get_domains()[1]
        )
    raise ValueError(indexer_type)


def get_tag(
    indexer_type: str, name: str, defined_on: Optional[pp.solvers.DomainFilter] = None
):
    """Constructs either an `EquationTag` or a `VariableTag`."""
    if defined_on is None:
        defined_on = pp.solvers.Anywhere()
    if indexer_type == "equation_indexer":
        return pp.solvers.EquationTag(name=name, defined_on=defined_on)
    elif indexer_type == "variable_indexer":
        return pp.solvers.VariableTag(name=name, defined_on=defined_on)
    raise ValueError(indexer_type)


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_indexer_size(indexer_type: str):
    """Indexer should have the size attribute."""
    indexer = get_indexer(indexer_type)
    assert indexer.size == 11


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_comparison_by_name_and_domain(indexer_type: str):
    """Atomic equations and variables must be comparible and hashable by name and
    domain.

    """
    indexer = get_indexer(indexer_type)
    first, second = list(indexer.indices.keys())[:2]
    assert first != second
    if indexer_type == "equation_indexer":
        same_as_first = pp.ad.EquationOnDomain(name=first.name, domain=first.domain)
    elif indexer_type == "variable_indexer":
        same_as_first = pp.ad.Variable(
            name=first.name, ndof=pp.ad.GridEntities(cells=1), domain=first.domain
        )
    else:
        raise ValueError(indexer_type)
    assert first == same_as_first
    assert second != same_as_first
    assert same_as_first in indexer.indices


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_projection_indices(indexer_type: str) -> None:
    """Projection indices respect requested variable order, behave correctly on edge
    cases.

    """
    indexer = get_indexer(indexer_type)
    _, x, _, y = indexer.indices

    # Original order.
    np.testing.assert_array_equal(
        indexer.projection_indices([x, y]), np.array([3, 4, 7, 8, 9, 10])
    )

    # Reversed order.
    np.testing.assert_array_equal(
        indexer.projection_indices([y, x]), np.array([7, 8, 9, 10, 3, 4])
    )

    # Duplicated key. The method does not perform deduplication. If at some point we
    # realize that this causes more harm than good, this should be reconsider.
    np.testing.assert_array_equal(
        indexer.projection_indices([x, x]),
        np.array([3, 4, 3, 4]),
    )

    # Test behavior on empty subset.
    empty_projection = indexer.projection_indices([])
    assert empty_projection.dtype == int
    assert empty_projection.size == 0

    # Test behavior on unknown key.
    unknown = get_unknown_key(indexer_type)
    with pytest.raises(ValueError):
        _ = indexer.projection_indices([unknown])


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_construct_restricted_indexer(indexer_type: str) -> None:
    """Restricted indexers are contiguous and follow requested variable order."""
    indexer = get_indexer(indexer_type)
    _, x, _, y = indexer.indices

    # Original order.
    restricted = indexer.construct_restricted_indexer([x, y])
    assert restricted.size == 6
    assert list(restricted.indices) == [x, y]
    np.testing.assert_array_equal(restricted.indices[x], [0, 1])
    np.testing.assert_array_equal(restricted.indices[y], [2, 3, 4, 5])

    # Reversed order.
    restricted = indexer.construct_restricted_indexer([y, x])
    assert restricted.size == 6
    assert list(restricted.indices) == [y, x]
    np.testing.assert_array_equal(restricted.indices[y], [0, 1, 2, 3])
    np.testing.assert_array_equal(restricted.indices[x], [4, 5])

    # Duplicated key.
    with pytest.raises(ValueError):
        _ = indexer.construct_restricted_indexer([x, x])

    # Empty input.
    restricted = indexer.construct_restricted_indexer([])
    assert restricted.size == 0
    assert list(restricted.indices) == []

    # Unknown key.
    with pytest.raises(ValueError):
        _ = indexer.construct_restricted_indexer([get_unknown_key(indexer_type)])


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_filter_by_tags(indexer_type: str) -> None:
    """Tag filtering selects matching operators and preserves indexer order.

    Also tests `construct_restricted_indexer_from_tags`, since its current
    implementation is trivial, to avoid regression if its implementation changes later.

    """

    @dataclass(frozen=True)
    class OnDomain(pp.solvers.DomainFilter):
        """Tag with hard-coded domain."""

        domain: pp.GridLike

        def filter(self, domain: pp.GridLike, model: pp.PorePyModel) -> bool:
            return domain is self.domain

    def compare_restricted_indexers(tags, included):
        restricted = indexer.construct_restricted_indexer_from_tags(
            tags=tags, model=mock_model
        )
        restricted_expected = indexer.construct_restricted_indexer(included)
        # Keys must match.
        assert list(restricted.indices) == list(restricted_expected.indices)
        # Values must match.
        for key, expected in restricted_expected.indices.items():
            np.testing.assert_array_equal(restricted.indices[key], expected)

    indexer = get_indexer(indexer_type)
    mock_model = cast(pp.PorePyModel, "mock model")

    # Case 1: Equation / variable with name "y".
    tags = [get_tag(indexer_type=indexer_type, name="y")]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all(eq_or_var.name == "y" for eq_or_var in included)
    assert all(eq_or_var.name != "y" for eq_or_var in excluded)
    compare_restricted_indexers(tags=tags, included=included)

    # Case 2: Equations / variables with names "x" and "y" only on grid 1.
    grid_1 = list(indexer.indices)[1].domain
    on_grid_1 = OnDomain(domain=grid_1)
    tags = [
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_1),
        get_tag(indexer_type=indexer_type, name="y", defined_on=on_grid_1),
    ]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all(eq_or_var.domain == grid_1 for eq_or_var in included)
    assert all(eq_or_var.domain != grid_1 for eq_or_var in excluded)
    # "x" should go first.
    assert included[0].name == "x" and included[1].name == "y"
    compare_restricted_indexers(tags=tags, included=included)

    # Case 3: Same as Case 2 but reversed order.
    tags = [
        get_tag(indexer_type=indexer_type, name="y", defined_on=on_grid_1),
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_1),
    ]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all(eq_or_var.domain == grid_1 for eq_or_var in included)
    assert all(eq_or_var.domain != grid_1 for eq_or_var in excluded)
    # The reversed order is ignored. "x" should go first.
    assert included[0].name == "x" and included[1].name == "y"
    compare_restricted_indexers(tags=tags, included=included)

    # Case 4: Unknown name, should include nothing.
    tags = [get_tag(indexer_type=indexer_type, name="z")]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 0
    assert len(excluded) == 4
    compare_restricted_indexers(tags=tags, included=included)

    # Case 5: Filter does not match any known domain, should include nothing.
    unknown = pp.CartGrid([4])
    tags = [get_tag(indexer_type=indexer_type, name="x", defined_on=OnDomain(unknown))]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 0
    assert len(excluded) == 4
    compare_restricted_indexers(tags=tags, included=included)

    # Case 6: Disjoint tags with the same name.
    grid_0 = list(indexer.indices)[0].domain
    on_grid_0 = OnDomain(domain=grid_0)
    tags = [
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_0),
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_1),
    ]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all(eq_or_var.name == "x" for eq_or_var in included)
    assert all(eq_or_var.name != "x" for eq_or_var in excluded)
    # Grid 0 should go first.
    assert included[0].domain == grid_0 and included[1].domain == grid_1
    compare_restricted_indexers(tags=tags, included=included)

    # Case 7: Same as Case 6 but reversed oreder.
    tags = [
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_1),
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_0),
    ]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 2
    assert all(eq_or_var.name == "x" for eq_or_var in included)
    assert all(eq_or_var.name != "x" for eq_or_var in excluded)
    # Grid 1 should go first.
    assert included[1].domain == grid_1 and included[0].domain == grid_0

    # Case 8: Tags overlap. Should raise.
    tags = [
        get_tag(indexer_type=indexer_type, name="x"),
        get_tag(indexer_type=indexer_type, name="x", defined_on=on_grid_0),
    ]
    with pytest.raises(ValueError, match="Duplicated operators"):
        indexer.filter_by_tags(tags, model=mock_model)


def test_equation_system_indexer() -> None:
    """EquationSystemIndexer is a subclass with a helper property for EquationSystem:
    `equation_image_space_composition`. It stores indices by grid name, and indices are
    local to each equation. This test checks it.

    """
    grid_0, grid_1, grid_2 = get_domains()
    equation_image_space_composition = {
        "x": {
            grid_0: np.array([0, 1, 2]),
            grid_1: np.array([], dtype=int),  # Empty.
        },
        "y": {
            grid_1: np.array([0, 1]),
            grid_2: np.array([2, 3, 4, 5]),
        },
    }
    # Not using deepcopy because it will copy the grids, and the grid comparison won't
    # work correctly.
    indexer = pp.ad.EquationSystemIndexer(
        equation_image_space_composition=copy(equation_image_space_composition)
    )
    assert indexer.size == 9

    x0, x1, y1, y2 = get_equations([grid_0, grid_1, grid_2])
    expected = {
        x0: np.array([0, 1, 2]),
        x1: np.array([], dtype=int),
        y1: np.array([3, 4]),
        y2: np.array([5, 6, 7, 8]),
    }

    # Keys must be equal.
    assert list(indexer.indices) == list(expected)
    # Values must be equal.
    for key, expected_val in expected.items():
        np.testing.assert_array_equal(indexer.indices[key], expected_val)
    assert indexer.equation_image_space_composition == equation_image_space_composition


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_group_by_name(indexer_type: str) -> None:
    """Group indices by operator name and domain while preserving order."""
    indexer = get_indexer(indexer_type)
    grouped_dofs = indexer.group_by_name()
    grid_0 = list(indexer.indices)[0].domain
    grid_1 = list(indexer.indices)[1].domain
    grid_2 = list(indexer.indices)[3].domain
    expected = {
        "x": {
            grid_0: np.array([0, 1, 2]),
            grid_1: np.array([3, 4]),
        },
        "y": {
            grid_1: np.array([5, 6]),
            grid_2: np.array([7, 8, 9, 10]),
        },
    }
    # Names must match.
    assert list(grouped_dofs) == list(expected)
    for name, domains_to_dofs_expected in expected.items():
        domains_to_dofs = grouped_dofs[name]
        # Domains must match.
        assert list(domains_to_dofs) == list(domains_to_dofs_expected)
        for domain, dofs_expected in domains_to_dofs_expected.items():
            np.testing.assert_array_equal(domains_to_dofs[domain], dofs_expected)


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_identify_dof(indexer_type: str) -> None:
    """Identify the operator owning each index and reject out-of-range indices."""
    indexer = get_indexer(indexer_type)
    grid_0 = list(indexer.indices)[0].domain
    grid_1 = list(indexer.indices)[1].domain
    grid_2 = list(indexer.indices)[3].domain

    # Within range.
    for i in reversed(range(11)):
        eq_or_var = indexer.identify_dof(i)
        # Name.
        if i <= 4:
            assert eq_or_var.name == "x"
        else:
            assert eq_or_var.name == "y"
        # Domain.
        if i <= 2:
            assert eq_or_var.domain == grid_0
        elif i <= 6:
            assert eq_or_var.domain == grid_1
        else:
            assert eq_or_var.domain == grid_2

    # Out of range.
    with pytest.raises(KeyError):
        _ = indexer.identify_dof(-1)

    with pytest.raises(KeyError):
        _ = indexer.identify_dof(11)


@pytest.mark.parametrize("indexer_type", ["equation_indexer", "variable_indexer"])
def test_empty_indices(indexer_type: str):
    """Test indexer behavior if one key has an empty index array."""
    grid_0, grid_1, grid_2 = get_domains()
    if indexer_type == "equation_indexer":
        x0, x1, y1, _ = get_equations([grid_0, grid_1, grid_2])
    elif indexer_type == "variable_indexer":
        x0, x1, y1, _ = get_variables([grid_0, grid_1, grid_2])
    else:
        raise ValueError(indexer_type)

    indices = {
        x0: np.array([0, 1]),
        x1: np.zeros(0, dtype=int),
        y1: np.array([2, 3]),
    }
    indexer = get_indexer(indexer_type=indexer_type, indices=indices)
    assert indexer.size == 4

    # construct_restricted_indexer does not filter out empty indices.
    restricted = indexer.construct_restricted_indexer(operators=list(indexer.indices))
    # Keys must be equal.
    assert list(indexer.indices) == list(restricted.indices)
    # Values must be equal.
    for key, expected in indexer.indices.items():
        np.testing.assert_array_equal(restricted.indices[key], expected)

    # projection_indices ignores empty indices.
    projection = indexer.projection_indices(operators=list(indexer.indices))
    assert projection.dtype == np.int64
    np.testing.assert_array_equal(projection, [0, 1, 2, 3])

    # filter_by_tags does not filter out empty indices.
    mock_model = cast(pp.PorePyModel, "mock model")
    tags = [get_tag(indexer_type=indexer_type, name="x")]
    included, excluded = indexer.filter_by_tags(tags=tags, model=mock_model)
    assert len(included) == 2
    assert len(excluded) == 1

    # group_by_name FILTERS OUT empty indices.
    grouped = indexer.group_by_name()
    assert list(grouped) == ["x", "y"]
    assert len(grouped["x"]) == 1, "Only the non-empty domain."
    assert list(grouped["x"].keys())[0] == grid_0
    np.testing.assert_array_equal(list(grouped["x"].values())[0], [0, 1])
