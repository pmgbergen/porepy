"""
Tests of the well class and well-matrix intersection functionality.

Content:
  - TestWellClass: Simple tests for the well class, mainly covering construction.
  - test_compute_well_rock_matrix_intersections: Test the computation of intersections
    between a well and the rock matrix mesh. The tested method is not in active use and
    the test is marked as expected to fail pending updates to code and test.

"""

import numbers
from typing import List

import numpy as np
import pytest

import porepy as pp


class TestWellClass:
    @pytest.mark.parametrize(
        "coords",
        [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 1], [1, 2], [1, 3]]),
        ],
    )
    def test_single_well(self, coords) -> None:
        """Test the creation of a well object."""
        # Define the well coordinates.

        # Create a well object.
        well = pp.Well(coords, index=0)
        assert well.index == 0
        # Check that the well object has the correct attributes.
        assert isinstance(well, pp.Well)
        assert np.allclose(well.pts, coords)
        assert well.num_segments() == coords.shape[1] - 1

        for seg_ind, seg_coord in well.segments():
            # Check that the segment coordinates are correct.
            assert np.allclose(
                seg_coord,
                coords[:, seg_ind[0] : seg_ind[1] + 2],
            )

    def test_multiple_wells(self) -> None:
        """Test the creation of multiple well objects. Nothing special should happen."""
        # Define the well coordinates.
        coords1 = np.array([[0, 0], [0, 1]])
        coords2 = np.array([[1, 1], [1, 2], [1, 3]])

        # Create multiple well objects.
        well1 = pp.Well(coords1, index=0)
        well2 = pp.Well(coords2, index=1)

        # Check that the well objects have the correct attributes.
        assert np.allclose(well1.pts, coords1)
        assert np.allclose(well2.pts, coords2)


@pytest.mark.xfail(reason="Well-matrix functionality has not been updated")
def test_compute_well_rock_matrix_intersections(get_mdg) -> None:
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
