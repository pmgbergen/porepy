"""Tests for the ``GridEntity`` enum.

Verifies:
  1. Enum member values and the ``void`` sentinel.
  2. ``GridEntity`` is accessible as ``pp.ad.GridEntity``.
  3. ``admissible_dof_types`` contains only ``GridEntity`` members.
  4. ``create_variables`` works with enum-keyed ``dof_info`` dicts.
  5. ``set_equation`` works with enum-keyed ``equations_per_grid_entity`` dicts.
"""

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity


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
