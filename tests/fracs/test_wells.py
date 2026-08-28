"""
Tests of the well class and well-matrix intersection functionality.

Content:
  - TestWellClass: Simple tests for the well class, mainly covering construction.

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
