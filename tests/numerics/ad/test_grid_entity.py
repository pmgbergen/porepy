"""Tests for the ``GridEntity`` enum.

Verifies:
  1. Enum member values and the ``void`` sentinel.
  2. ``GridEntity`` is accessible as ``pp.ad.GridEntity``.
  3. ``admissible_dof_types`` contains only ``GridEntity`` members.
  4. ``create_variables`` works with enum-keyed ``dof_info`` dicts.
  5. ``set_equation`` works with enum-keyed ``equations_per_grid_entity`` dicts.
  6. ``SurrogateFactory`` works with enum-keyed ``dof_info`` dicts.
"""

import pytest
import numpy as np

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity


# ---------------------------------------------------------------------------
# Enum member values
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# admissible_dof_types uses enum members
# ---------------------------------------------------------------------------


class TestAdmissibleDofTypes:
    def test_admissible_dof_types_are_grid_entity_members(self):
        mdg = pp.MixedDimensionalGrid()
        eq_sys = pp.ad.EquationSystem(mdg)
        for entry in eq_sys.admissible_dof_types:
            assert isinstance(entry, GridEntity)


# ---------------------------------------------------------------------------
# create_variables with enum-keyed dof_info
# ---------------------------------------------------------------------------


def _simple_mdg():
    """Return a minimal MixedDimensionalGrid with two subdomains."""
    mdg = pp.MixedDimensionalGrid()
    g1 = pp.CartGrid([2, 2])
    g2 = pp.CartGrid([3, 3])
    mdg.add_subdomains([g1, g2])
    return mdg, g1, g2


class TestCreateVariables:
    def test_enum_dof_info_cells(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "p", dof_info={GridEntity.cells: 1}, subdomains=[g1, g2]
        )
        assert var is not None
        assert len(eq.variables) == 2

    def test_enum_dof_info_faces(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("u", dof_info={GridEntity.faces: 1}, subdomains=[g1])
        assert var is not None

    def test_enum_dof_info_nodes(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("v", dof_info={GridEntity.nodes: 1}, subdomains=[g1])
        assert var is not None

    def test_mixed_dof_info_enum_keys(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "w", dof_info={GridEntity.cells: 1, GridEntity.faces: 2}, subdomains=[g1]
        )
        assert var is not None

    def test_non_admissible_dof_type_raises(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        with pytest.raises(ValueError, match="Non-admissible"):
            eq.create_variables("bad", dof_info={"volume": 1}, subdomains=[g1])

    def test_num_dofs_correct(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        eq.create_variables("p", dof_info={GridEntity.cells: 2}, subdomains=[g1])
        expected = g1.num_cells * 2
        assert eq.num_dofs() == expected


# ---------------------------------------------------------------------------
# set_equation with enum-keyed equations_per_grid_entity
# ---------------------------------------------------------------------------


class TestSetEquation:
    def test_enum_equations_per_grid_entity(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "p", dof_info={GridEntity.cells: 1}, subdomains=[g1, g2]
        )
        operator = var + var
        operator.set_name("test_eq")
        eq.set_equation(
            operator,
            grids=[g1, g2],
            equations_per_grid_entity={GridEntity.cells: 1},
        )
        assert "test_eq" in eq.equations


# ---------------------------------------------------------------------------
# SurrogateFactory with enum-keyed dof_info
# ---------------------------------------------------------------------------


class TestSurrogateFactory:
    def test_enum_dof_info(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "p", dof_info={GridEntity.cells: 1}, subdomains=[g1, g2]
        )
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        assert factory is not None
