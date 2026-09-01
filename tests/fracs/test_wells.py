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
    _connection_projections,
    _connection_side_grid,
    _distribute_shared_intervals,
    _equivalent_radius,
    _mortar_cell_directions,
    _perpendicular_section,
    _polygon_principal_extents,
    _segment_cell_interval,
    _validate_convex_cell,
    _well_connections,
    well_equivalent_radii,
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

    @staticmethod
    def _lengths(pieces: dict) -> dict:
        """Total length attributed to each cell."""
        return {c: sum(b - a for a, b in iv) for c, iv in pieces.items()}

    def test_disjoint_intervals_are_kept(self) -> None:
        """Cells covering different parts of the segment keep their own lengths."""
        out = _distribute_shared_intervals({0: (0.0, 0.4), 1: (0.4, 1.0)}, 1e-10)
        assert np.isclose(self._lengths(out)[0], 0.4)
        assert np.isclose(self._lengths(out)[1], 0.6)

    def test_fully_shared_interval_is_split_equally(self) -> None:
        """A segment in a shared face is claimed by both cells, and split evenly."""
        out = _distribute_shared_intervals({3: (0.0, 1.0), 7: (0.0, 1.0)}, 1e-10)
        lengths = self._lengths(out)
        assert np.isclose(lengths[3], 0.5)
        assert np.isclose(lengths[7], 0.5)
        assert np.isclose(sum(lengths.values()), 1.0)

    def test_shared_pieces_do_not_overlap(self) -> None:
        """The pieces handed to the sharing cells are disjoint.

        The division has to be geometric rather than a weight, because the mortar grid
        built from these intervals recomputes its cell volumes from its own geometry. A
        failure here means the shared length would be counted once per sharing cell.
        """
        out = _distribute_shared_intervals({3: (0.2, 0.8), 7: (0.2, 0.8)}, 1e-10)
        all_pieces = sorted(iv for pieces in out.values() for iv in pieces)
        for (_, upper), (lower, _) in zip(all_pieces[:-1], all_pieces[1:]):
            assert lower >= upper - 1e-12
        assert np.isclose(sum(b - a for a, b in all_pieces), 0.6)

    def test_edge_shared_by_many_cells(self) -> None:
        """A segment along an edge is split equally between all cells sharing it."""
        out = _distribute_shared_intervals({c: (0.0, 1.0) for c in range(6)}, 1e-10)
        lengths = self._lengths(out)
        assert np.allclose(list(lengths.values()), 1.0 / 6.0)
        assert np.isclose(sum(lengths.values()), 1.0)

    def test_partial_overlap(self) -> None:
        """Only the overlapping part is shared; the rest is attributed in full."""
        out = _distribute_shared_intervals({0: (0.0, 0.6), 1: (0.4, 1.0)}, 1e-10)
        lengths = self._lengths(out)
        # [0, 0.4] to cell 0, [0.4, 0.6] shared, [0.6, 1.0] to cell 1.
        assert np.isclose(lengths[0], 0.4 + 0.1)
        assert np.isclose(lengths[1], 0.4 + 0.1)
        assert np.isclose(sum(lengths.values()), 1.0)

    def test_contiguous_contact_is_one_interval(self) -> None:
        """A cell touching the segment along one stretch is given a single interval.

        Each interval becomes one mortar cell, so gratuitous splitting would inflate the
        number of interface unknowns without changing the physics.
        """
        out = _distribute_shared_intervals({0: (0.0, 0.5), 1: (0.5, 1.0)}, 1e-10)
        assert all(len(pieces) == 1 for pieces in out.values())

    def test_segment_partly_outside_all_cells(self) -> None:
        """Length not covered by any cell is not attributed to one."""
        out = _distribute_shared_intervals({0: (0.25, 0.75)}, 1e-10)
        assert np.isclose(sum(self._lengths(out).values()), 0.5)

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


def _contact_lengths(mdg: pp.MixedDimensionalGrid) -> np.ndarray:
    """Length of contact between each well cell and each rock matrix cell.

    Read back from the interface of ``mdg``, whose mortar cells are the individual
    contacts: each mortar cell lies in one well cell and one rock matrix cell, and its
    volume is the contact length.
    """
    interfaces = list(mdg.interfaces())
    assert len(interfaces) == 1
    intf = interfaces[0]
    well_cells = intf._secondary_to_mortar_avg.tocsr().indices
    matrix_cells = intf._primary_to_mortar_avg.tocsr().indices

    _, well = mdg.interface_to_subdomain_pair(intf)
    matrix, _ = mdg.interface_to_subdomain_pair(intf)
    lengths = np.zeros((well.num_cells, matrix.num_cells))
    np.add.at(lengths, (well_cells, matrix_cells), intf.cell_volumes)
    return lengths


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
        for c, pieces in _distribute_shared_intervals(intervals, 1e-10).items():
            connections[seg, c] = sum(upper - lower for lower, upper in pieces)
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
    well_lengths = mdg.subdomains(dim=1)[0].cell_volumes
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    expected = _brute_force_connections(matrix, points) * well_lengths[:, None]
    np.testing.assert_allclose(_contact_lengths(mdg), expected, atol=1e-10)


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
    well_lengths = mdg.subdomains(dim=1)[0].cell_volumes
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    # All trajectories lie inside the unit cube, so every segment is fully covered.
    np.testing.assert_allclose(
        _contact_lengths(mdg).sum(axis=1), well_lengths, atol=1e-10
    )


def test_well_partially_outside_domain() -> None:
    """A well leaving the domain is attributed only its interior part."""
    matrix = pp.CartGrid([2, 2, 2], [1, 1, 1])
    matrix.compute_geometry()

    # The single segment runs from below the domain to its mid-height, so exactly one
    # third of it is inside.
    points = np.array([[0.3, 0.3], [0.3, 0.3], [-1.0, 0.5]])
    mdg = _well_mdg(matrix, points)
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    # The segment is 1.5 long, of which the third above z = 0 lies in the domain.
    np.testing.assert_allclose(_contact_lengths(mdg).sum(axis=1), [0.5])


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

    lengths = _contact_lengths(mdg)
    # The segment is 0.5 long and lies in the face between the two cells.
    np.testing.assert_allclose(np.sort(lengths[0])[-2:], [0.25, 0.25])
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


class TestWellConnections:
    """The contacts between a well and the rock matrix, as a list of records.

    Each connection becomes one cell of the mortar grid, so a failure here changes the
    number of interface unknowns as well as the geometry they are attached to.
    """

    @staticmethod
    def _connections(matrix: pp.Grid, points: np.ndarray):
        well = _well_mdg(matrix, points).subdomains(dim=1)[0]
        tree = pp.adtree.ADTree(2 * matrix.dim, matrix.dim)
        tree.from_grid(matrix)
        return _well_connections(matrix, well, tree, 1e-10, 1e-5)

    def test_contacts_cover_the_well(self) -> None:
        """The contacts of a well cell account for its whole length."""
        matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
        matrix.compute_geometry()
        points = WELL_TRAJECTORIES["slanted"]
        connections = self._connections(matrix, points)

        well = _well_mdg(matrix, points).subdomains(dim=1)[0]
        for cell in range(well.num_cells):
            covered = sum(c.length for c in connections if c.well_cell == cell)
            assert np.isclose(covered, well.cell_volumes[cell], atol=1e-10)

    def test_contacts_lie_on_the_well(self) -> None:
        """Every contact is a sub-segment of the well cell it belongs to."""
        matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
        matrix.compute_geometry()
        points = WELL_TRAJECTORIES["interior_vertical"]
        connections = self._connections(matrix, points)

        for c in connections:
            for point in (c.start, c.end):
                # Distance from the point to the line through the well cell endpoints.
                a, b = points[:, c.well_cell], points[:, c.well_cell + 1]
                direction = (b - a) / np.linalg.norm(b - a)
                offset = point - a
                assert np.isclose(
                    np.linalg.norm(offset - np.dot(offset, direction) * direction),
                    0.0,
                    atol=1e-10,
                )

    def test_shared_face_gives_one_contact_per_cell(self) -> None:
        """A well in a shared face contacts both neighbours, over disjoint stretches."""
        matrix = pp.CartGrid([2, 1, 1], [1, 1, 1])
        matrix.compute_geometry()
        points = np.array([[0.5, 0.5], [0.5, 0.5], [0.25, 0.75]])
        connections = self._connections(matrix, points)

        assert len(connections) == 2
        assert {c.matrix_cell for c in connections} == {0, 1}
        assert np.allclose([c.length for c in connections], 0.25)


class TestConnectionMortarPieces:
    """The mortar side grid and projections built from the connections.

    These are the objects the interface unknowns live on, so a failure here shows up
    downstream as a miscount of degrees of freedom or as a coupling that does not
    conserve mass.
    """

    @staticmethod
    def _setup(matrix: pp.Grid, points: np.ndarray):
        well = _well_mdg(matrix, points).subdomains(dim=1)[0]
        tree = pp.adtree.ADTree(2 * matrix.dim, matrix.dim)
        tree.from_grid(matrix)
        connections = _well_connections(matrix, well, tree, 1e-10, 1e-5)
        return matrix, well, connections

    @pytest.mark.parametrize(
        "trajectory", list(WELL_TRAJECTORIES), ids=list(WELL_TRAJECTORIES)
    )
    def test_side_grid_matches_the_connections(self, trajectory) -> None:
        """One cell per contact, of the right length, adding up to the well length."""
        matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
        matrix.compute_geometry()
        matrix, well, connections = self._setup(matrix, WELL_TRAJECTORIES[trajectory])
        side = _connection_side_grid(connections)

        assert side.num_cells == len(connections)
        np.testing.assert_allclose(
            side.cell_volumes, [c.length for c in connections], atol=1e-12
        )
        # The trajectories lie inside the domain, so the contacts tile the whole well.
        np.testing.assert_allclose(
            side.cell_volumes.sum(), well.cell_volumes.sum(), atol=1e-10
        )

    @pytest.mark.parametrize(
        "trajectory", list(WELL_TRAJECTORIES), ids=list(WELL_TRAJECTORIES)
    )
    def test_projection_normalisations(self, trajectory) -> None:
        """The intensive maps average, the extensive maps distribute.

        PorePy requires unit row sums of the intensive projections and unit column sums
        of the extensive ones. Getting this wrong does not raise: it silently scales the
        quantity being projected.
        """
        matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
        matrix.compute_geometry()
        matrix, well, connections = self._setup(matrix, WELL_TRAJECTORIES[trajectory])
        proj = _connection_projections(matrix, well, connections)

        for name in ("_primary_to_mortar_avg", "_secondary_to_mortar_avg"):
            row_sums = np.asarray(proj[name].sum(axis=1)).ravel()
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-12, err_msg=name)

        for name in ("_primary_to_mortar_int", "_secondary_to_mortar_int"):
            col_sums = np.asarray(proj[name].sum(axis=0)).ravel()
            touched = col_sums > 0
            np.testing.assert_allclose(col_sums[touched], 1.0, atol=1e-12, err_msg=name)

    def test_each_connection_sees_one_cell_on_each_side(self) -> None:
        """A connection lies in exactly one cell of either neighbour.

        This is what makes the two normalisations separately definable, and is the
        property the per-well-cell arrangement lacked.
        """
        matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
        matrix.compute_geometry()
        matrix, well, connections = self._setup(matrix, WELL_TRAJECTORIES["kinked"])
        proj = _connection_projections(matrix, well, connections)

        for name in proj:
            assert np.all(np.diff(proj[name].tocsr().indptr) == 1), name


def test_interface_has_one_cell_per_contact() -> None:
    """The mortar grid resolves each contact separately.

    This is the property the coupling is built on: a well cell crossing several rock
    matrix cells gets one flux per crossing, so it can take fluid from one cell and
    deliver it to another. A regression to one mortar cell per well cell would leave the
    total contact length right but make such a state unrepresentable.
    """
    matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
    matrix.compute_geometry()
    points = WELL_TRAJECTORIES["slanted"]
    mdg = _well_mdg(matrix, points)
    well = mdg.subdomains(dim=1)[0]
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    intf = list(mdg.interfaces())[0]
    lengths = _contact_lengths(mdg)
    assert intf.num_cells == np.count_nonzero(lengths)
    # The well is genuinely non-conforming, so there is more than one contact per cell.
    assert intf.num_cells > well.num_cells


def test_flux_distributed_to_the_matrix_is_conserved() -> None:
    """Interface fluxes reach the rock matrix without loss or duplication.

    ``mortar_to_primary_int`` is what carries the well flux into the rock matrix mass
    balance. A failure here is lost or invented mass rather than a misplaced one.
    """
    matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
    matrix.compute_geometry()
    mdg = _well_mdg(matrix, WELL_TRAJECTORIES["kinked"])
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    intf = list(mdg.interfaces())[0]
    unit_flux = np.ones(intf.num_cells)
    to_matrix = intf.mortar_to_primary_int() @ unit_flux
    to_well = intf.mortar_to_secondary_int() @ unit_flux

    assert np.isclose(to_matrix.sum(), intf.num_cells)
    assert np.isclose(to_well.sum(), intf.num_cells)


def test_pressure_projected_to_the_interface_is_not_rescaled() -> None:
    """A constant rock matrix pressure reaches the interface unchanged.

    ``primary_to_mortar_avg`` is intensive, so it must average rather than sum. Under
    the previous arrangement the same matrix served both roles, and projecting an
    extensive quantity silently rescaled it.
    """
    matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
    matrix.compute_geometry()
    mdg = _well_mdg(matrix, WELL_TRAJECTORIES["interior_vertical"])
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)

    intf = list(mdg.interfaces())[0]
    constant = np.full(matrix.num_cells, 3.5)
    np.testing.assert_allclose(intf.primary_to_mortar_avg() @ constant, 3.5)


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


def _box_vertices(dx: float, dy: float, dz: float) -> np.ndarray:
    """Vertices of an axis-aligned box with one corner at the origin."""
    return np.array(
        [[x, y, z] for x in (0, dx) for y in (0, dy) for z in (0, dz)], dtype=float
    ).T


def _random_rotation(seed: int) -> np.ndarray:
    """A proper rotation matrix, for testing invariance under change of frame."""
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


class TestPerpendicularSection:
    """Projection of a cell onto the plane perpendicular to the well."""

    @pytest.mark.parametrize(
        "direction, expected",
        [
            ([0, 0, 1], (4.0, 2.0)),
            ([0, 1, 0], (4.0, 3.0)),
            ([1, 0, 0], (3.0, 2.0)),
        ],
    )
    def test_box_projects_to_the_perpendicular_face(self, direction, expected) -> None:
        """Along an axis, the shadow of a box is the face perpendicular to that axis.

        A failure means the projection plane is not the one perpendicular to the given
        direction, since the two side lengths would then be mixed.
        """
        section = _perpendicular_section(_box_vertices(4, 2, 3), np.array(direction))
        extents = _polygon_principal_extents(section)
        np.testing.assert_allclose(extents, expected, atol=1e-12)

    def test_section_is_independent_of_the_frame(self) -> None:
        """Rotating cell and well together must not change the measured shape.

        The projection is expressed in an arbitrary orthonormal frame of the plane, so
        a failure here means some quantity read off it is frame dependent after all.
        """
        vertices = _box_vertices(4, 2, 3)
        direction = np.array([0.3, -0.7, 1.0])
        reference = _polygon_principal_extents(
            _perpendicular_section(vertices, direction)
        )
        for seed in range(5):
            rotation = _random_rotation(seed)
            rotated = _polygon_principal_extents(
                _perpendicular_section(rotation @ vertices, rotation @ direction)
            )
            np.testing.assert_allclose(rotated, reference, atol=1e-12)

    def test_direction_scale_and_sign_are_immaterial(self) -> None:
        """Only the line along the well matters, not its parametrisation."""
        vertices = _box_vertices(4, 2, 3)
        direction = np.array([0.3, -0.7, 1.0])
        reference = _polygon_principal_extents(
            _perpendicular_section(vertices, direction)
        )
        for factor in (-1.0, 0.01, 250.0):
            np.testing.assert_allclose(
                _polygon_principal_extents(
                    _perpendicular_section(vertices, factor * direction)
                ),
                reference,
                atol=1e-12,
            )


class TestPolygonPrincipalExtents:
    """The two lengths summarising a convex polygon."""

    @pytest.mark.parametrize("sides", [(1.0, 1.0), (3.0, 1.0), (0.3, 1.7)])
    def test_rectangle_is_reproduced_exactly(self, sides) -> None:
        """On a rectangle the equivalent rectangle must be the rectangle itself.

        This exactness is what makes the equivalent well radius reduce to Peaceman's
        expression on a Cartesian cell, so a failure here breaks that reduction.
        """
        width, height = sides
        polygon = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=float
        )
        np.testing.assert_allclose(
            _polygon_principal_extents(polygon), sorted(sides, reverse=True), atol=1e-12
        )

    def test_extents_scale_with_the_polygon(self) -> None:
        """Both lengths are homogeneous of degree one in the polygon."""
        polygon = np.array([[0, 0], [2, 0], [2.5, 1.5], [0.5, 1.0]], dtype=float)
        reference = np.array(_polygon_principal_extents(polygon))
        np.testing.assert_allclose(
            _polygon_principal_extents(3.0 * polygon), 3.0 * reference, atol=1e-12
        )

    def test_extra_corners_on_an_edge_do_not_change_the_shape(self) -> None:
        """Subdividing an edge must not alter the measured extents.

        Second moments of area are used precisely so that the result depends on the
        shape rather than on how its boundary is discretised. A failure indicates a
        reversion to a corner-based measure, which would misjudge a cell whose
        projection has nearly coincident corners.
        """
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        subdivided = np.array(
            [[0, 0], [1.5, 0], [1.99, 0], [2, 0], [2, 2], [0, 2]], dtype=float
        )
        np.testing.assert_allclose(
            _polygon_principal_extents(subdivided),
            _polygon_principal_extents(square),
            atol=1e-12,
        )


class TestEquivalentRadius:
    """The equivalent well radius of a cell traversed by a well."""

    @pytest.mark.parametrize(
        "sides", [(1, 1, 1), (2, 1, 1), (4, 1, 3), (0.3, 1.7, 1.0)]
    )
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_cartesian_cell_reproduces_peaceman(self, sides, axis) -> None:
        """A well along an axis of a Cartesian cell must give Peaceman's radius.

        This is the reference case the whole expression generalises, so it is asserted
        to machine precision rather than to a tolerance. A failure means the
        generalisation no longer contains the case it generalises.
        """
        perpendicular = [side for index, side in enumerate(sides) if index != axis]
        expected = 0.14 * np.sqrt(perpendicular[0] ** 2 + perpendicular[1] ** 2)
        radius = _equivalent_radius(_box_vertices(*sides), np.eye(3)[axis])
        assert radius == pytest.approx(expected, abs=1e-12)

    def test_radius_is_independent_of_the_frame(self) -> None:
        """Rotating cell and well together must leave the radius unchanged."""
        vertices = _box_vertices(4, 2, 3)
        direction = np.array([0.3, -0.7, 1.0])
        reference = _equivalent_radius(vertices, direction)
        for seed in range(5):
            rotation = _random_rotation(seed)
            rotated = _equivalent_radius(rotation @ vertices, rotation @ direction)
            assert rotated == pytest.approx(reference, abs=1e-12)

    def test_radius_scales_with_the_cell(self) -> None:
        """The radius is a length, so it scales linearly with the cell."""
        vertices = _box_vertices(4, 2, 3)
        direction = np.array([0.3, -0.7, 1.0])
        reference = _equivalent_radius(vertices, direction)
        assert _equivalent_radius(3.0 * vertices, direction) == pytest.approx(
            3.0 * reference, abs=1e-12
        )

    def test_radius_depends_on_the_well_direction(self) -> None:
        """A cell crossed broadside and one crossed lengthwise must differ.

        This is the property the volume-based radius lacks, and the reason for
        replacing it: a cell long in one direction has a much smaller cross-section
        when the well runs along that direction than when it runs across it.
        """
        vertices = _box_vertices(0.05, 0.05, 10.0)
        along = _equivalent_radius(vertices, np.array([0.0, 0.0, 1.0]))
        across = _equivalent_radius(vertices, np.array([1.0, 0.0, 0.0]))
        assert along < 0.1 * across

    def test_prism_reduces_to_its_triangular_cross_section(self) -> None:
        """For a vertical well through a prism, the section is the base triangle.

        Prismatic extrusions of two-dimensional simplex grids are one of the target
        grid types, and are the only unstructured case with an unambiguous expected
        cross-section, so they pin down the projection where tetrahedra cannot.
        """
        triangle = np.array([[0.0, 1.3, 0.4], [0.0, 0.0, 0.9]])
        vertices = np.vstack(
            [np.tile(triangle, 2), np.repeat([0.0, 2.5], 3)],
        )
        extruded = _polygon_principal_extents(
            _perpendicular_section(vertices, np.array([0.0, 0.0, 1.0]))
        )
        flat = _polygon_principal_extents(triangle.T[[0, 1, 2]])
        np.testing.assert_allclose(extruded, flat, atol=1e-12)


def _prism_grid(num_cells: int, height: float) -> pp.Grid:
    """A grid of triangular prisms, from extruding a two-dimensional simplex grid."""
    base = pp.StructuredTriangleGrid([num_cells, num_cells], [1, 1])
    base.compute_geometry()
    grid, _, _ = pp.grid_extrusion.extrude_grid(
        base, np.linspace(0, height, num_cells + 1)
    )
    grid.compute_geometry()
    return grid


class TestWellEquivalentRadii:
    """Equivalent radii read off a well-matrix interface."""

    @pytest.mark.parametrize("grid_type", ["cartesian", "simplex", "prism"])
    def test_radii_match_the_standalone_expression(self, grid_type) -> None:
        """Every mortar cell must carry the radius of the cell it lies in.

        A failure means the mortar cells and the rock matrix cells, or the mortar cells
        and the contact directions, are being paired up in different orders.
        """
        if grid_type == "cartesian":
            matrix = pp.CartGrid([3, 3, 3], [1, 1, 1])
            matrix.compute_geometry()
        elif grid_type == "simplex":
            matrix = pp.StructuredTetrahedralGrid([3, 3, 3], [1, 1, 1])
            matrix.compute_geometry()
        else:
            matrix = _prism_grid(3, 1.0)

        mdg = _well_mdg(matrix, WELL_TRAJECTORIES["slanted"])
        pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)
        intf = list(mdg.interfaces())[0]

        radii = well_equivalent_radii(mdg, intf)
        assert radii.shape == (intf.num_cells,)
        assert np.all(radii > 0)

        cell_nodes = matrix.cell_nodes().tocsc()
        matrix_cells = intf._primary_to_mortar_avg.tocsr().indices
        directions = _mortar_cell_directions(intf)
        for mortar_cell, cell in enumerate(matrix_cells):
            loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
            expected = _equivalent_radius(
                matrix.nodes[:, cell_nodes.indices[loc]], directions[:, mortar_cell]
            )
            assert radii[mortar_cell] == pytest.approx(expected, abs=1e-12)

    def test_vertical_well_in_a_cartesian_grid_gives_peaceman(self) -> None:
        """The one case with a closed form: every contact must reproduce it.

        A vertical well through an axis-aligned Cartesian grid only ever crosses cells
        along their third axis, so every mortar cell must carry the classical Peaceman
        radius of the cell cross-section.
        """
        matrix = pp.CartGrid([4, 4, 4], [2.0, 1.0, 3.0])
        matrix.compute_geometry()
        points = np.array([[0.31, 0.31], [0.27, 0.27], [0.1, 2.9]])
        mdg = _well_mdg(matrix, points)
        pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)
        intf = list(mdg.interfaces())[0]

        expected = 0.14 * np.sqrt((2.0 / 4) ** 2 + (1.0 / 4) ** 2)
        np.testing.assert_allclose(
            well_equivalent_radii(mdg, intf), expected, atol=1e-12
        )

    def test_a_cell_crossed_twice_gets_two_radii(self) -> None:
        """Two well cells crossing one rock matrix cell differently must differ.

        This is the reason the radius belongs to the mortar grid rather than to the
        rock matrix grid; a failure means it has collapsed back to one value per rock
        matrix cell.
        """
        matrix = pp.CartGrid([1, 1, 1], [1.0, 4.0, 8.0])
        matrix.compute_geometry()
        # A kinked well, both legs of which stay inside the single cell.
        points = np.array([[0.5, 0.5, 0.5], [0.5, 3.5, 3.5], [1.0, 1.0, 7.0]])
        mdg = _well_mdg(matrix, points)
        pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg)
        intf = list(mdg.interfaces())[0]

        matrix_cells = intf._primary_to_mortar_avg.tocsr().indices
        assert np.all(matrix_cells == 0)
        radii = well_equivalent_radii(mdg, intf)
        assert radii.size == 2
        assert not np.isclose(radii[0], radii[1])
