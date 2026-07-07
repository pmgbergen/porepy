"""
Tests of the well class. In particular, functionality for constructing the
well network and the resulting updates to the mixed-dimensional grid are tested.

Content:
  * Addition of one well to mdgs with one or two 2d fractures.
  * Addition of two wells to mdgs with one or three 2d fractures.
Both tests check for number of grids, number of edges and three types of face
tags. Grid node ordering is tacitly assumed - if the assumption is broken, the
well implementation should also be revisited.

"""

import numbers
from typing import List

import numpy as np
import pytest

import porepy as pp


@pytest.mark.xfail(reason="Well-matrix functionality has not been updated")
def test_add_one_well_with_matrix(get_mdg) -> None:
    """Compute intersection between one well and the rock matrix mesh."""
    mdg = get_mdg([], [1])
    # Add the coupling between the rock matrix and the well.
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    # Check the number of subdomains and interfaces.
    assert mdg.num_subdomains() == 2
    assert mdg.num_interfaces() == 1

    # Check the well grid.
    for well_grid in mdg.subdomains(dim=1):
        np.testing.assert_allclose(well_grid.nodes[2].max(), 1)
        np.testing.assert_allclose(well_grid.nodes[2].min(), 0.2)

    # EK: The idea behind the test is sound, but the known numbers must be revisited
    # after the well-matrix functionality has been updated/brought back to life.
    # for intf in mdg.interfaces():
    #     assert intf.num_sides() == 1
    #     assert intf.num_cells == 1
    #     assert np.allclose(intf.mortar_to_secondary_int().todense(), 1)

    #     known = np.zeros(24)
    #     known[0] = 0.175
    #     known[3] = 0.29166667
    #     known[11] = 0.25
    #     known[22] = 0.08333333
    #     known[23] = 0.2

    #     # Since the generation of .msh files is platform-dependent, only norm values
    #     # are compared.
    #     assert np.isclose(
    #         np.linalg.norm(known),
    #         np.linalg.norm(intf.mortar_to_primary_int().toarray().flatten()),
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )

    # Adding a well also adds a new boundary grid. Check that new boundary grid is
    # initialized.
    well_boundaries = mdg.boundaries(dim=0)
    assert len(well_boundaries) == 1
    for well_bg in well_boundaries:
        # num_cells is one of the attributes that are initialized lazily.
        assert isinstance(well_bg.num_cells, numbers.Integral)
