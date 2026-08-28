"""
Tests of the well class and well-matrix intersection functionality.

Content:
  - TestWellClass: Simple tests for the well class, mainly covering construction.
  - TestSegmentCellInterval: Unit tests of the segment-cell clipping,
    including the degenerate configurations in which a well touches the
    boundary of a cell.
  - TestDistributeSharedIntervals: Unit tests of the rule that shares a
    segment between cells without losing or double counting length.

The central invariant, used throughout, is that the lengths attributed to the cells a
well segment passes through must sum to the length of that segment (or, for a segment
that leaves the domain, to the part inside it). A failure of that invariant means the
search is silently losing or double counting well-matrix contact, which shows up
downstream as lost or duplicated mass transfer rather than as a crash.

"""

import numbers
from typing import List

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.wells_3d import _distribute_shared_intervals, _segment_cell_interval


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


def _unit_cube_half_spaces() -> tuple[np.ndarray, np.ndarray]:
    """Half-space representation of the unit cube ``[0, 1]^3``."""
    normals = np.hstack((-np.eye(3), np.eye(3)))
    offsets = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return normals, offsets


def _clip_length(start: np.ndarray, end: np.ndarray, tol: float = 1e-5) -> float:
    """Length of the part of a segment inside the unit cube."""
    normals, offsets = _unit_cube_half_spaces()
    interval = _segment_cell_interval(start, end, normals, offsets, tol)
    if interval is None:
        return 0.0
    return (interval[1] - interval[0]) * float(np.linalg.norm(end - start))


class TestSegmentCellInterval:
    """Clipping of a segment against a single convex cell, here the unit cube.

    The degenerate cases are the point of this class: a well is not conforming to the
    rock matrix mesh, so it may run along a face, an edge, or through a vertex of a cell
    rather than through its interior. A failure here means such configurations are
    mis-measured, which the invariant tests further down will then also catch, but less
    specifically.
    """

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            # Wholly inside.
            ([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 0.6),
            # Start inside, end outside.
            ([0.5, 0.5, 0.5], [0.5, 0.5, 2.0], 0.5),
            # Both endpoints outside, crossing the cell.
            ([0.5, 0.5, -1.0], [0.5, 0.5, 2.0], 1.0),
            # Wholly outside.
            ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0], 0.0),
            # Touching a single face from outside: zero length, not a crossing.
            ([0.5, 0.5, 1.0], [0.5, 0.5, 2.0], 0.0),
            # Exactly spanning the cell.
            ([0.5, 0.5, 0.0], [0.5, 0.5, 1.0], 1.0),
        ],
    )
    def test_generic_positions(self, start, end, expected) -> None:
        """A segment in general position relative to the cell."""
        length = _clip_length(np.array(start, dtype=float), np.array(end, dtype=float))
        assert np.isclose(length, expected)

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            # Lying in a face of the cell: the full length is claimed.
            ([0.2, 0.5, 0.0], [0.8, 0.5, 0.0], 0.6),
            # Running along an edge of the cell.
            ([0.0, 0.0, 0.2], [0.0, 0.0, 0.8], 0.6),
            # Passing exactly through a vertex, otherwise outside.
            ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], np.sqrt(3.0)),
            # Grazing a vertex from outside: zero length.
            ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 0.0),
            # Parallel to a face, just outside it.
            ([0.2, 0.5, -1e-3], [0.8, 0.5, -1e-3], 0.0),
        ],
    )
    def test_boundary_positions(self, start, end, expected) -> None:
        """A segment touching the boundary of the cell.

        These are the configurations that the previous, intersection-point based
        implementation silently reported as zero length.
        """
        length = _clip_length(np.array(start, dtype=float), np.array(end, dtype=float))
        assert np.isclose(length, expected)

    def test_degenerate_segment(self) -> None:
        """A segment of zero length has no intersection."""
        p = np.array([0.5, 0.5, 0.5])
        normals, offsets = _unit_cube_half_spaces()
        assert _segment_cell_interval(p, p, normals, offsets, 1e-5) is None


class TestDistributeSharedIntervals:
    """Distribution of a segment over cells, without loss or double counting."""

    def test_disjoint_intervals_are_kept(self) -> None:
        """Cells covering different parts of the segment keep their own lengths."""
        fractions = _distribute_shared_intervals({0: (0.0, 0.4), 1: (0.4, 1.0)}, 1e-10)
        assert np.isclose(fractions[0], 0.4)
        assert np.isclose(fractions[1], 0.6)

    def test_fully_shared_interval_is_split_equally(self) -> None:
        """A segment in a shared face is claimed by both cells, and split evenly."""
        fractions = _distribute_shared_intervals({3: (0.0, 1.0), 7: (0.0, 1.0)}, 1e-10)
        assert np.isclose(fractions[3], 0.5)
        assert np.isclose(fractions[7], 0.5)
        assert np.isclose(sum(fractions.values()), 1.0)

    def test_edge_shared_by_many_cells(self) -> None:
        """A segment along an edge is split equally between all cells sharing it."""
        cells = {c: (0.0, 1.0) for c in range(6)}
        fractions = _distribute_shared_intervals(cells, 1e-10)
        assert np.allclose(list(fractions.values()), 1.0 / 6.0)
        assert np.isclose(sum(fractions.values()), 1.0)

    def test_partial_overlap(self) -> None:
        """Only the overlapping part is shared; the rest is attributed in full."""
        fractions = _distribute_shared_intervals({0: (0.0, 0.6), 1: (0.4, 1.0)}, 1e-10)
        # [0, 0.4] to cell 0, [0.4, 0.6] shared, [0.6, 1.0] to cell 1.
        assert np.isclose(fractions[0], 0.4 + 0.1)
        assert np.isclose(fractions[1], 0.4 + 0.1)
        assert np.isclose(sum(fractions.values()), 1.0)

    def test_segment_partly_outside_all_cells(self) -> None:
        """Length not covered by any cell is not attributed to one."""
        fractions = _distribute_shared_intervals({0: (0.25, 0.75)}, 1e-10)
        assert np.isclose(sum(fractions.values()), 0.5)

    def test_no_candidates(self) -> None:
        """A segment outside the grid produces no connections."""
        assert _distribute_shared_intervals({}, 1e-10) == {}
