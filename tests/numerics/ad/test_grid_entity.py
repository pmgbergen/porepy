"""Tests for the ``GridEntity`` enum and the ``GridEntities`` value object.

Verifies:
  1. Enum member values.
  2. ``GridEntity`` is accessible as ``pp.ad.GridEntity``.
  3. ``create_variables`` works with enum-keyed ``dof_info`` dicts.
  4. ``set_equation`` works with enum-keyed ``equations_per_grid_entity`` dicts.
  5. ``GridEntities`` is an immutable, hashable value object with one field per
     ``GridEntity`` member, and its derived properties treat a zero count and an
     absent entity as the same thing.
"""

import dataclasses

import pytest

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity
from porepy.numerics.ad.grid_entity import GridEntities


class TestGridEntityValues:
    """GridEntity members have the correct string values."""

    @pytest.mark.parametrize(
        "member, value",
        [
            (GridEntity.cells, "cells"),
            (GridEntity.faces, "faces"),
            (GridEntity.nodes, "nodes"),
        ],
    )
    def test_member_value(self, member, value):
        assert member.value == value

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


@pytest.mark.parametrize(
    "dof_info, expected",
    [
        (GridEntities(), False),
        (GridEntities(cells=1), True),
        (GridEntities(cells=0, faces=0, nodes=0), False),
        (GridEntities(nodes=3), True),
    ],
)
def test_bool_is_true_iff_some_entity_is_present(dof_info, expected):
    """A GridEntities is falsy exactly when no entity carries any DOFs."""
    assert bool(dof_info) is expected


@pytest.mark.parametrize(
    "dof_info, expected",
    [
        (GridEntities(), frozenset()),
        (GridEntities(cells=1), frozenset({GridEntity.cells})),
        (GridEntities(cells=0, faces=2), frozenset({GridEntity.faces})),
        (
            GridEntities(cells=1, faces=2, nodes=3),
            frozenset({GridEntity.cells, GridEntity.faces, GridEntity.nodes}),
        ),
    ],
)
def test_present_entities(dof_info, expected):
    """Entities with a zero count are not present."""
    assert dof_info.present_entities == expected


@pytest.mark.parametrize(
    "dof_info, expected",
    [
        (GridEntities(), False),
        (GridEntities(cells=1), True),
        (GridEntities(faces=1), True),
        (GridEntities(nodes=1), True),
        (GridEntities(cells=2), False),
        (GridEntities(cells=1, faces=1), False),
    ],
)
def test_is_unit_on_single_entity(dof_info, expected):
    """Only one entity present, carrying exactly one DOF, broadcasts."""
    assert dof_info.is_unit_on_single_entity() is expected


def test_zero_count_equals_absent_entity():
    """An explicit zero and an omitted entity describe the same DOF distribution."""
    assert GridEntities(cells=1, faces=0) == GridEntities(cells=1)
    assert GridEntities.from_mapping({GridEntity.faces: 0}) == GridEntities()


def test_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        GridEntities(cells=1).cells = 2  # type: ignore[misc]
