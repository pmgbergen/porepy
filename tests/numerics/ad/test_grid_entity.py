"""Tests for the ``GridEntity`` enum introduced in Stage 1 of the AD
operator domains/ranges implementation (see ``copilot/ad_operator_domains_and_ranges.md``).

The tests verify:
  1. Enum member values and the ``void`` sentinel.
  2. Backward compatibility: ``GridEntity`` members compare equal to plain strings,
     share the same hash, and are therefore interchangeable as dict keys or set members.
  3. ``create_variables`` still accepts legacy string-keyed ``dof_info`` dicts.
  4. ``set_equation`` still accepts legacy string-keyed ``equations_per_grid_entity`` dicts.
  5. ``SurrogateFactory`` still accepts legacy string-keyed ``dof_info`` dicts.
"""

import pytest
import numpy as np

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity


# ---------------------------------------------------------------------------
# 1. Enum member values
# ---------------------------------------------------------------------------


class TestGridEntityValues:
    """GridEntity members have the correct string values."""

    def test_cells_value(self):
        assert GridEntity.cells.value == "cells"

    def test_faces_value(self):
        assert GridEntity.faces.value == "faces"

    def test_nodes_value(self):
        assert GridEntity.nodes.value == "nodes"

    def test_void_value(self):
        assert GridEntity.void.value == "void"

    def test_enum_has_four_members(self):
        assert len(GridEntity) == 4

    def test_construction_from_string(self):
        """GridEntity('cells') == GridEntity.cells."""
        assert GridEntity("cells") is GridEntity.cells
        assert GridEntity("faces") is GridEntity.faces
        assert GridEntity("nodes") is GridEntity.nodes
        assert GridEntity("void") is GridEntity.void

    def test_accessible_via_pp_ad(self):
        """GridEntity is accessible as pp.ad.GridEntity."""
        assert pp.ad.GridEntity is GridEntity


# ---------------------------------------------------------------------------
# 2. Backward compatibility: str equality and hash
# ---------------------------------------------------------------------------


class TestGridEntityBackwardCompatibility:
    """GridEntity members are interchangeable with plain strings."""

    def test_enum_equals_string(self):
        assert GridEntity.cells == "cells"
        assert GridEntity.faces == "faces"
        assert GridEntity.nodes == "nodes"
        assert GridEntity.void == "void"

    def test_string_equals_enum(self):
        assert "cells" == GridEntity.cells
        assert "faces" == GridEntity.faces
        assert "nodes" == GridEntity.nodes

    def test_hash_equality(self):
        assert hash(GridEntity.cells) == hash("cells")
        assert hash(GridEntity.faces) == hash("faces")
        assert hash(GridEntity.nodes) == hash("nodes")

    def test_enum_key_lookup_in_string_keyed_dict(self):
        """Enum member can look up a value in a string-keyed dict."""
        d = {"cells": 1, "faces": 2, "nodes": 3}
        assert d[GridEntity.cells] == 1
        assert d[GridEntity.faces] == 2
        assert d[GridEntity.nodes] == 3

    def test_string_key_lookup_in_enum_keyed_dict(self):
        """String can look up a value in an enum-keyed dict."""
        d = {GridEntity.cells: 1, GridEntity.faces: 2, GridEntity.nodes: 3}
        assert d["cells"] == 1
        assert d["faces"] == 2
        assert d["nodes"] == 3

    def test_set_membership_string_in_enum_set(self):
        """Plain strings are members of sets built from GridEntity values."""
        entity_set = {GridEntity.cells, GridEntity.faces, GridEntity.nodes}
        assert "cells" in entity_set
        assert "faces" in entity_set
        assert "nodes" in entity_set

    def test_set_membership_enum_in_string_set(self):
        """GridEntity members are found in string sets."""
        string_set = {"cells", "faces", "nodes"}
        assert GridEntity.cells in string_set
        assert GridEntity.faces in string_set
        assert GridEntity.nodes in string_set

    def test_dict_equality_string_vs_enum_keys(self):
        """A dict with string keys equals a dict with GridEntity keys (same values)."""
        string_dict = {"cells": 1, "faces": 2}
        enum_dict = {GridEntity.cells: 1, GridEntity.faces: 2}
        assert string_dict == enum_dict

    def test_str_concatenation(self):
        """GridEntity members can be concatenated like plain strings."""
        assert "num_" + GridEntity.cells == "num_cells"
        assert "num_" + GridEntity.faces == "num_faces"
        assert "num_" + GridEntity.nodes == "num_nodes"


# ---------------------------------------------------------------------------
# 3. admissible_dof_types uses enum members
# ---------------------------------------------------------------------------


class TestAdmissibleDofTypes:
    def test_admissible_dof_types_are_grid_entity_members(self):
        mdg = pp.MixedDimensionalGrid()
        eq_sys = pp.ad.EquationSystem(mdg)
        for entry in eq_sys.admissible_dof_types:
            assert isinstance(entry, GridEntity)

    def test_string_key_is_admissible(self):
        """Passing a plain string like 'cells' is still considered admissible."""
        mdg = pp.MixedDimensionalGrid()
        g = pp.CartGrid([2, 2])
        mdg.add_subdomains([g])
        eq_sys = pp.ad.EquationSystem(mdg)
        # Should not raise even though we pass a string key.
        eq_sys.create_variables("p", dof_info={"cells": 1}, subdomains=[g])


# ---------------------------------------------------------------------------
# 4. create_variables accepts legacy string-keyed dof_info
# ---------------------------------------------------------------------------


def _simple_mdg():
    """Return a minimal MixedDimensionalGrid with two subdomains."""
    mdg = pp.MixedDimensionalGrid()
    g1 = pp.CartGrid([2, 2])
    g2 = pp.CartGrid([3, 3])
    mdg.add_subdomains([g1, g2])
    return mdg, g1, g2


class TestCreateVariablesBackwardCompatibility:
    def test_string_dof_info_cells(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info={"cells": 1}, subdomains=[g1, g2])
        # Variable should be created without error.
        assert var is not None
        assert len(eq.variables) == 2

    def test_string_dof_info_faces(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("u", dof_info={"faces": 1}, subdomains=[g1])
        assert var is not None

    def test_string_dof_info_nodes(self):
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("v", dof_info={"nodes": 1}, subdomains=[g1])
        assert var is not None

    def test_mixed_dof_info_string_keys(self):
        """Multi-entity dof_info with plain string keys still works."""
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "w", dof_info={"cells": 1, "faces": 2}, subdomains=[g1]
        )
        assert var is not None

    def test_enum_dof_info_cells(self):
        """Enum-keyed dof_info works too."""
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables(
            "p", dof_info={GridEntity.cells: 1}, subdomains=[g1, g2]
        )
        assert var is not None
        assert len(eq.variables) == 2

    def test_non_admissible_dof_type_raises(self):
        """Passing an invalid dof type should still raise ValueError."""
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        with pytest.raises(ValueError, match="Non-admissible"):
            eq.create_variables("bad", dof_info={"volume": 1}, subdomains=[g1])

    def test_num_dofs_correct_with_string_key(self):
        """Number of DOFs computed correctly when string keys are used."""
        mdg, g1, _ = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        eq.create_variables("p", dof_info={"cells": 2}, subdomains=[g1])
        expected = g1.num_cells * 2
        assert eq.num_dofs() == expected


# ---------------------------------------------------------------------------
# 5. set_equation accepts legacy string-keyed equations_per_grid_entity
# ---------------------------------------------------------------------------


class TestSetEquationBackwardCompatibility:
    def test_string_equations_per_grid_entity(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info={"cells": 1}, subdomains=[g1, g2])
        operator = var + var
        operator.set_name("test_eq")
        # Should not raise with plain string key.
        eq.set_equation(
            operator,
            grids=[g1, g2],
            equations_per_grid_entity={"cells": 1},
        )
        assert "test_eq" in eq.equations

    def test_enum_equations_per_grid_entity(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info={"cells": 1}, subdomains=[g1, g2])
        operator = var + var
        operator.set_name("test_eq2")
        eq.set_equation(
            operator,
            grids=[g1, g2],
            equations_per_grid_entity={GridEntity.cells: 1},
        )
        assert "test_eq2" in eq.equations


# ---------------------------------------------------------------------------
# 6. SurrogateFactory accepts legacy string-keyed dof_info
# ---------------------------------------------------------------------------


class TestSurrogateFactoryBackwardCompatibility:
    def test_string_dof_info(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info={"cells": 1}, subdomains=[g1, g2])
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={"cells": 1},
        )
        assert factory is not None

    def test_enum_dof_info(self):
        mdg, g1, g2 = _simple_mdg()
        eq = pp.ad.EquationSystem(mdg)
        var = eq.create_variables("p", dof_info={"cells": 1}, subdomains=[g1, g2])
        factory = pp.ad.SurrogateFactory(
            name="f2",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        assert factory is not None
