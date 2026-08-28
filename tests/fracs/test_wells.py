"""
Tests of the well class and well-matrix intersection functionality.

Content:
  - TestWellClass: Simple tests for the well class, mainly covering construction.
  - TestSegmentCellInterval: Unit tests of the segment-cell clipping,
    including the degenerate configurations in which a well touches the
    boundary of a cell.
  - TestDistributeSharedIntervals: Unit tests of the rule that shares a
    segment between cells without losing or double counting length.
  - Tests of compute_well_rock_matrix_intersections, covering agreement with a
    brute-force oracle and conservation of well length.
  - TestValidateConvexCell: Unit tests of the convexity and planarity guard.
  - test_search_examines_a_shrinking_fraction_of_the_grid: a scaling
    benchmark, run on the weekly schedule rather than per pull request.

The central invariant, used throughout, is that the lengths attributed to the cells a
well segment passes through must sum to the length of that segment (or, for a segment
that leaves the domain, to the part inside it). A failure of that invariant means the
search is silently losing or double counting well-matrix contact, which shows up
downstream as lost or duplicated mass transfer rather than as a crash.

"""

import numbers
import time
from typing import List

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.wells_3d import (
    _cell_half_spaces,
    _distribute_shared_intervals,
    _segment_cell_interval,
    _validate_convex_cell,
)


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

        A well does not conform to the rock matrix mesh, so these configurations arise
        routinely rather than exceptionally, and each must be measured rather than
        discarded as degenerate.
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


def _well_mdg(matrix: pp.Grid, points: np.ndarray) -> pp.MixedDimensionalGrid:
    """Build a mixed-dimensional grid with one rock matrix grid and one well.

    Parameters:
        matrix: The three-dimensional rock matrix grid.
        points: ``shape=(3, num_points)``

            Vertices of the well polyline, in order.

    Returns:
        A mixed-dimensional grid holding the two subdomains, without any interface.

    """
    well = pp.TensorGrid(np.linspace(0, 1, points.shape[1]))
    well.nodes = points
    well.compute_geometry()
    # Tag the grid as a well, which is how the search identifies well subdomains.
    well.well_num = 0

    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([matrix, well])
    return mdg


def _connection_matrix(mdg: pp.MixedDimensionalGrid) -> np.ndarray:
    """Dense well-to-matrix connection map of the single interface in ``mdg``."""
    interfaces = list(mdg.interfaces())
    assert len(interfaces) == 1
    return interfaces[0]._primary_to_mortar_int.toarray()


def _brute_force_connections(
    matrix: pp.Grid, points: np.ndarray, tol: float = 1e-5
) -> np.ndarray:
    """Reference implementation of the well-matrix search.

    Every well segment is tested against every cell of the rock matrix, bypassing the
    ``ADTree`` entirely. This is far too slow for production use but is obviously
    correct, which makes it the natural oracle for the tree-based search.
    """
    cell_faces = matrix.cell_faces.tocsc()
    connections = np.zeros((points.shape[1] - 1, matrix.num_cells))
    for seg in range(points.shape[1] - 1):
        start, end = points[:, seg], points[:, seg + 1]
        intervals = {}
        for c in range(matrix.num_cells):
            normals, offsets = _cell_half_spaces(matrix, c, cell_faces)
            interval = _segment_cell_interval(start, end, normals, offsets, tol)
            if interval is not None:
                intervals[c] = interval
        for c, fraction in _distribute_shared_intervals(intervals, 1e-10).items():
            connections[seg, c] = fraction
    return connections


# Well trajectories exercising the configurations that a non-conforming well may take
# relative to the rock matrix mesh. Ids are used to make failures readable.
WELL_TRAJECTORIES = {
    "interior_vertical": np.array(
        [[0.31, 0.31, 0.31], [0.27, 0.27, 0.27], [0.1, 0.5, 0.9]]
    ),
    "on_cell_faces": np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.1, 0.5, 0.9]]),
    "slanted": np.array([[0.13, 0.5, 0.87], [0.17, 0.45, 0.81], [0.08, 0.5, 0.92]]),
    "kinked": np.array([[0.3, 0.6, 0.3], [0.3, 0.3, 0.6], [0.2, 0.5, 0.8]]),
    "along_grid_line": np.array(
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.05, 0.5, 0.95]]
    ),
}


@pytest.mark.parametrize(
    "trajectory", list(WELL_TRAJECTORIES), ids=list(WELL_TRAJECTORIES)
)
@pytest.mark.parametrize("grid_type", ["simplex", "cartesian"])
def test_search_matches_brute_force_oracle(trajectory, grid_type) -> None:
    """The tree-based search must agree exactly with an exhaustive search.

    A failure means the ``ADTree`` broad phase is discarding cells that the well
    genuinely intersects, since both paths share the same narrow phase.
    """
    if grid_type == "simplex":
        matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
    else:
        matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
    matrix.compute_geometry()

    points = WELL_TRAJECTORIES[trajectory]
    mdg = _well_mdg(matrix, points)
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    np.testing.assert_allclose(
        _connection_matrix(mdg), _brute_force_connections(matrix, points), atol=1e-10
    )


@pytest.mark.parametrize(
    "trajectory", list(WELL_TRAJECTORIES), ids=list(WELL_TRAJECTORIES)
)
@pytest.mark.parametrize("num_cells", [2, 3, 5])
def test_well_length_is_conserved(trajectory, num_cells) -> None:
    """All of a well segment inside the domain must be attributed to some cell.

    A failure here means well-matrix mass transfer is being lost or double counted.
    """
    matrix = pp.StructuredTetrahedralGrid([num_cells] * 3, [1, 1, 1])
    matrix.compute_geometry()

    points = WELL_TRAJECTORIES[trajectory]
    mdg = _well_mdg(matrix, points)
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    # All trajectories lie inside the unit cube, so every segment is fully covered.
    row_sums = _connection_matrix(mdg).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_well_partially_outside_domain() -> None:
    """A well leaving the domain is attributed only its interior part."""
    matrix = pp.CartGrid([2, 2, 2], [1, 1, 1])
    matrix.compute_geometry()

    # The single segment runs from below the domain to its mid-height, so exactly one
    # third of it is inside.
    points = np.array([[0.3, 0.3], [0.3, 0.3], [-1.0, 0.5]])
    mdg = _well_mdg(matrix, points)
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    np.testing.assert_allclose(_connection_matrix(mdg).sum(axis=1), [1.0 / 3.0])


def test_well_in_shared_face_is_split_between_neighbours(caplog) -> None:
    """A well lying in a face shared by two cells is split equally between them.

    The well runs along the interior face of a two-cell grid, so it belongs to neither
    cell's interior. Splitting equally keeps the total contact length correct and is
    symmetric in the two cells; see the note in ``_distribute_shared_intervals``.
    """
    matrix = pp.CartGrid([2, 1, 1], [1, 1, 1])
    matrix.compute_geometry()

    points = np.array([[0.5, 0.5], [0.5, 0.5], [0.25, 0.75]])
    mdg = _well_mdg(matrix, points)
    with caplog.at_level("WARNING"):
        pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    connections = _connection_matrix(mdg)
    np.testing.assert_allclose(np.sort(connections[0]), [0.5, 0.5])
    assert "split equally" in caplog.text


class TestValidateConvexCell:
    """Rejection of cells the half-space representation cannot describe.

    Representing a cell as an intersection of half-spaces is exact only for a convex
    cell with planar faces. Such a cell must be refused explicitly, because the clipping
    would otherwise return a plausible but wrong contact length rather than fail.
    """

    def test_convex_cell_accepted(self) -> None:
        """The unit cube is convex with planar faces."""
        normals, offsets = _unit_cube_half_spaces()
        vertices = np.array(
            [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
        ).T
        _validate_convex_cell(0, vertices, normals, offsets, 1e-5)

    def test_reflex_vertex_rejected(self) -> None:
        """A vertex pushed outside a face plane makes the cell non-convex."""
        normals, offsets = _unit_cube_half_spaces()
        vertices = np.array(
            [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
        ).T
        # Move one vertex outside the half-space of the top face.
        vertices[:, -1] = np.array([1.0, 1.0, 1.3])
        with pytest.raises(ValueError, match="not convex"):
            _validate_convex_cell(3, vertices, normals, offsets, 1e-5)

    def test_non_planar_face_rejected(self) -> None:
        """A face whose vertices do not share a plane is rejected."""
        normals, offsets = _unit_cube_half_spaces()
        vertices = np.array(
            [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
        ).T
        # Lift a single vertex of the top face off its plane.
        vertices[2, -1] = 1.05
        with pytest.raises(ValueError, match="not convex"):
            _validate_convex_cell(7, vertices, normals, offsets, 1e-5)

    def test_tolerance_is_relative_to_cell_size(self) -> None:
        """A perturbation below tolerance is accepted at any cell scale."""
        scale = 1e-3
        normals, offsets = _unit_cube_half_spaces()
        offsets = offsets * scale
        vertices = (
            np.array(
                [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
            ).T
            * scale
        )
        # A perturbation well below the relative tolerance times the cell diameter.
        vertices[2, -1] += 1e-9 * scale
        _validate_convex_cell(0, vertices, normals, offsets, 1e-5)


@pytest.mark.skipped
def test_search_examines_a_shrinking_fraction_of_the_grid() -> None:
    """Locating a well must not require looking at the whole rock matrix.

    Restricting the expensive exact intersection to a small set of candidate cells is
    what the ``ADTree`` exists for. The assertion is on the fraction of cells the tree
    offers as candidates, which is deterministic and therefore free of timing noise: as
    the grid is refined, a well of fixed trajectory must touch a steadily smaller share
    of it. Wall-clock timings of the individual phases are reported alongside, so that
    the test doubles as a benchmark record when the search is changed.

    """
    points = np.array([[0.31, 0.31, 0.31], [0.27, 0.27, 0.27], [0.05, 0.5, 0.95]])

    def measure(divisions: int) -> tuple[int, float, float, float]:
        matrix = pp.StructuredTetrahedralGrid([divisions] * 3, [1, 1, 1])
        matrix.compute_geometry()
        mdg = _well_mdg(matrix, points)

        build_start = time.perf_counter()
        tree = pp.adtree.ADTree(2 * matrix.dim, matrix.dim)
        tree.from_grid(matrix)
        build = time.perf_counter() - build_start

        # Candidates the tree offers for the segments of this well.
        candidates = 0
        for seg in range(points.shape[1] - 1):
            box = np.sort(
                np.vstack((points[:, seg], points[:, seg + 1])), axis=0
            ).ravel()
            candidates += tree.search(pp.adtree.ADTNode("query", box)).size

        total_start = time.perf_counter()
        pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)
        total = time.perf_counter() - total_start

        return matrix.num_cells, candidates / matrix.num_cells, build, total - build

    coarse_cells, coarse_share, coarse_build, coarse_query = measure(8)
    fine_cells, fine_share, fine_build, fine_query = measure(20)

    print(
        f"\ncells {coarse_cells} -> {fine_cells}"
        f"\ncandidate share {coarse_share:.4f} -> {fine_share:.4f}"
        f"\nADTree build    {coarse_build:.3f} s -> {fine_build:.3f} s"
        f"\nquery + clip    {coarse_query:.3f} s -> {fine_query:.3f} s"
    )

    # Refining the grid must not increase the share of it that has to be examined.
    assert fine_share < coarse_share
    # The well is a one-dimensional object in a three-dimensional grid, so the share of
    # cells it can touch has to become small. A loose bound is enough to catch a search
    # that has degenerated towards inspecting everything.
    assert fine_share < 0.05
