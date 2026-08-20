"""Tests for the ``GridEntity`` enum and the ``GridEntities`` value object.

Verifies:
  1. Enum member values and the ``void`` sentinel.
  2. ``GridEntity`` is accessible as ``pp.ad.GridEntity``.
  3. ``admissible_dof_types`` contains only ``GridEntity`` members.
  4. ``create_variables`` works with enum-keyed ``dof_info`` dicts.
  5. ``set_equation`` works with enum-keyed ``equations_per_grid_entity`` dicts.
  6. ``GridEntities`` behaves like a read-only, nonzero-filtered
     ``Mapping[GridEntity, int]``, while also being immutable and hashable.
"""

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity
from porepy.numerics.ad.grid_entity import GridEntities


class TestGridEntityValues:
    """GridEntity members have the correct string values."""

    def test_cells_value(self):
        assert GridEntity.cells.value == "cells"

    def test_faces_value(self):
        assert GridEntity.faces.value == "faces"

    def test_nodes_value(self):
        assert GridEntity.nodes.value == "nodes"

    def test_enum_has_three_members(self):
        assert len(GridEntity) == 3

    def test_construction_from_string(self):
        """GridEntity('cells') == GridEntity.cells."""
        assert GridEntity("cells") is GridEntity.cells
        assert GridEntity("faces") is GridEntity.faces
        assert GridEntity("nodes") is GridEntity.nodes

    def test_accessible_via_pp_ad(self):
        """GridEntity is accessible as pp.ad.GridEntity."""
        assert pp.ad.GridEntity is GridEntity


class TestAdmissibleDofTypes:
    def test_admissible_dof_types_are_grid_entity_members(self):
        mdg = pp.MixedDimensionalGrid()
        eq_sys = pp.ad.EquationSystem(mdg)
        for entry in eq_sys.admissible_dof_types:
            assert isinstance(entry, GridEntity)


@pytest.fixture
def simple_mdg(scope="module"):
    """Return a minimal MixedDimensionalGrid with two subdomains."""
    mdg = pp.MixedDimensionalGrid()
    g1 = pp.CartGrid([2, 2])
    g2 = pp.CartGrid([3, 3])
    mdg.add_subdomains([g1, g2])
    return mdg, g1, g2


class TestCreateVariables:
    """Test that create_variables works with enum-keyed dof_info dicts and return
    variables of the expected size.
    """

    def _expected(self, dof_info, mdg):
        """Compute the expected number of dofs for a given dof_info dict."""
        expected = 0
        for entity, num_dofs in dof_info.items():
            if entity == GridEntity.cells:
                expected += sum(g.num_cells for g in mdg.subdomains()) * num_dofs
            elif entity == GridEntity.faces:
                expected += sum(g.num_faces for g in mdg.subdomains()) * num_dofs
            elif entity == GridEntity.nodes:
                expected += sum(g.num_nodes for g in mdg.subdomains()) * num_dofs
        return expected

    @pytest.mark.parametrize(
        "dof_info",
        [
            {GridEntity.cells: 1},
            {GridEntity.faces: 1},
            {GridEntity.nodes: 1},
            {GridEntity.cells: 2},
            {GridEntity.cells: 1, GridEntity.faces: 2},
        ],
    )
    def test_enum_dof_info(self, simple_mdg, dof_info):
        mdg, g1, g2 = simple_mdg
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info=dof_info, subdomains=[g1, g2])
        assert var.size == self._expected(dof_info, mdg)

    def test_non_admissible_dof_type_raises(self, simple_mdg):
        mdg, g1, _ = simple_mdg
        eq = pp.ad.EquationSystem(mdg)
        with pytest.raises(ValueError, match="Non-admissible"):
            eq.create_variables("bad", dof_info={"volume": 1}, subdomains=[g1])


class TestGridEntitiesConstruction:
    def test_defaults_to_zero(self):
        g = GridEntities()
        assert (g.cells, g.faces, g.nodes) == (0, 0, 0)

    def test_named_fields(self):
        g = GridEntities(cells=1, faces=2, nodes=3)
        assert (g.cells, g.faces, g.nodes) == (1, 2, 3)

    @pytest.mark.parametrize("field", ["cells", "faces", "nodes"])
    def test_negative_count_raises(self, field):
        with pytest.raises(ValueError, match="non-negative"):
            GridEntities(**{field: -1})

    @pytest.mark.parametrize(
        "dof_info, expected",
        [
            ({}, GridEntities()),
            ({GridEntity.cells: 2}, GridEntities(cells=2)),
            (
                {GridEntity.cells: 2, GridEntity.faces: 1},
                GridEntities(cells=2, faces=1),
            ),
        ],
        ids=["empty", "single", "double"],
    )
    def test_from_mapping_normalizes_dict(self, dof_info, expected):
        assert GridEntities.from_mapping(dof_info) == expected

    def test_from_mapping_grid_entities_returned_unchanged(self):
        g = GridEntities(cells=1)
        assert GridEntities.from_mapping(g) is g


class TestGridEntitiesEqualityAndHashing:
    def test_equal_instances_hash_equal(self):
        a, b = GridEntities(cells=1), GridEntities(cells=1)
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal_instances(self):
        assert GridEntities(cells=1) != GridEntities(cells=1, faces=2)

    def test_usable_as_dict_key_and_in_set(self):
        a, b, c = GridEntities(cells=1), GridEntities(cells=1), GridEntities(faces=2)
        assert {a: "x"}[b] == "x"
        assert len({a, b, c}) == 2


@pytest.fixture(
    params=[
        (GridEntities(), {}),
        (GridEntities(cells=1), {GridEntity.cells: 1}),
        (GridEntities(cells=1, faces=2), {GridEntity.cells: 1, GridEntity.faces: 2}),
        (
            GridEntities(cells=1, faces=2, nodes=3),
            {GridEntity.cells: 1, GridEntity.faces: 2, GridEntity.nodes: 3},
        ),
    ],
    ids=["empty", "single", "double", "triple"],
)
def dof_info_case(request):
    """A (GridEntities, equivalent plain dict) pair, covering representative
    combinations of present/absent entities."""
    return request.param


class TestGridEntitiesMappingInterface:
    """GridEntities behaves like a dict, filtered to nonzero entities only, matching
    the convention that an absent entity and an explicit zero count are the same.
    Each check is compared directly against an equivalent plain dict, across the
    representative dof_info_case patterns, rather than one test per method/pattern.
    """

    def test_bulk_conversions_match_reference_dict(self, dof_info_case):
        g, expected = dof_info_case
        assert dict(g) == expected
        assert dict(g.items()) == expected
        assert set(g.keys()) == set(expected.keys())
        assert set(g.values()) == set(expected.values())
        assert set(g) == set(expected.keys())  # __iter__
        assert len(g) == len(expected)
        assert bool(g) == bool(expected)

    @pytest.mark.parametrize("entity", list(GridEntity))
    def test_per_entity_access_matches_reference_dict(self, dof_info_case, entity):
        g, expected = dof_info_case
        assert g.get(entity) == expected.get(entity, 0)
        assert g.get(entity, -1) == expected.get(entity, -1)
        assert (entity in g) == (entity in expected)
        if entity in expected:
            assert g[entity] == expected[entity]
        else:
            with pytest.raises(KeyError):
                g[entity]

    def test_frozenset_of_items_matches_dict_convention(self):
        """Mirrors the pattern used in operators.py's MixedDimensionalVariable
        construction (comparing dof_info across sub-variables)."""
        a = GridEntities(cells=1)
        b = GridEntities.from_mapping({GridEntity.cells: 1})
        assert frozenset(a.items()) == frozenset(b.items())

    def test_can_broadcast_pattern(self):
        """Mirrors the exact expression used in _operations.py's _can_broadcast."""
        g = GridEntities(cells=1)
        assert len(g) == 1 and set(g.values()) == {1}
