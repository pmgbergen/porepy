"""Module for well representation in :class:`~Well` and :class:`~WellNetworks`.

A well is a polyline of at least ``num_segments = 1`` segment defined through a list of
``num_points = num_segments + 1`` points. Wells are connected in a network.

After defining and meshing a fracture network, the wells may be added to the
mixed-dimensional grid by

    .. code:: python3

        compute_well_fracture_intersections(well_network, fracture_network)
        well_network.mesh(mdg)

"""

from __future__ import annotations

import logging
from typing import Callable, Iterator, NamedTuple, Optional

import gmsh
import numpy as np
import scipy.sparse as sps
from scipy.spatial import ConvexHull

import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

# Module-wide logger
logger = logging.getLogger(__name__)

_WELL_RADIUS_RESOLUTION_WARNING = 10
"""Ratio of equivalent to physical well radius below which the Peaceman well index is
considered too inaccurate to be used without warning."""


class Well:
    """Class representing a single well as a polyline embedded in 3D space.

    The fracture is defined by its vertices. It contains various utility methods,
    mainly intended for use together with the :class:`WellNetwork3d` class.

    Parameters:
        points: ``shape=(3, num_points)``

            Endpoints of each of ``num_points - 1`` line segments of the new well.
        index: ``default=None``

            Index of the well. if not given, the well will be assigned the index ``-1``.
        tags: ``default=None``

            Dictionary of tags, identifying the different types of points.

    """

    def __init__(
        self,
        points: np.ndarray,
        index: int = -1,
        tags: Optional[dict] = None,
    ) -> None:
        self.pts: np.ndarray = np.asarray(points, dtype=float)
        """``shape = (3, num_points)``

        Endpoints of each of the ``num_points - 1`` line segments of the new well."""
        self.dim: int = 1
        """Wells modelled as lines have always dimension 1."""

        self.tags: dict
        """Dictionary of tags, e.g., to identify different types of points.

        In particular, ``tags["intersecting_fractures"]`` has length ``num_points``
        and will be used to identify which fracture(s) intersects each of the points
        in ``pts``.

        """
        # Initialize tag dictionary.
        if tags is None:
            self.tags = {}
        else:
            self.tags = tags

        self._index: int = index
        """Index attribute."""

    @property
    def index(self) -> int:
        """Get the index of the well."""
        return self._index

    def segments(self) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
        """Iterate over the segments defined through segment indices and endpoints.

        Yields:
            Tuple with two elements

            :obj:`tuple`: ``len=2``

                2-tuple of integers, containing the segment indices.
            :obj:`numpy.ndarray`:

                Coordinates of the endpoints of the segment indices.

        """
        for i in range(self.num_segments()):
            segment_inds = (i, i + 1)
            endpoints = self.pts[:, segment_inds]
            yield segment_inds, endpoints

    def num_points(self) -> int:
        """Get number of points of the well.

        Returns:
            The number of points on the polyline well.
        """
        return self.pts.shape[1]

    def num_segments(self) -> int:
        """Get number of segments.

        Returns:
            Number of segments, i.e., ``num_points - 1``.
        """
        return self.num_points() - 1

    def to_gmsh(self) -> list[int]:
        """Export the well to GMSH as a polyline.

        Returns:
            List of GMSH line tags corresponding to the segments of the well.
        """
        num_points = self.pts.shape[1]
        point_tags = []
        for i in range(num_points):
            point_tag = gmsh.model.occ.addPoint(*self.pts[:, i], 0)
            point_tags.append(point_tag)

        segment_inds = []

        for seg_ind in range(num_points - 1):
            si = gmsh.model.occ.addLine(point_tags[seg_ind], point_tags[seg_ind + 1])
            segment_inds.append(si)

        return segment_inds

    def copy(self) -> Well:
        """Create a deep copy of the well.

        Warning:
            The original points (as given when the fracture was initialized) will
            *not* be preserved.

        Returns:
            A deep copy of the well with the same points and tags.

        """
        p = np.copy(self.pts)
        t = self.tags.copy()
        return Well(p, tags=t)

    def __repr__(self) -> str:
        """Get a string representation of the well properties.

        Returns:
            A string representation of the well properties.

        """
        s = f"Well consisting of {self.num_segments()} segments.\n"

        # If the well consists of only a few segments (5 here is somewhat randomly
        # chosen), list all the coordinates. If not, we limit the representation to
        # (effectively) the bounding box, which usually coincides with well endpoints.
        if self.num_points() < 5:
            s += "Coorditates of well points (x, y, z):\n"
            for i in range(self.num_points()):
                s += f"({self.pts[0, i]}, {self.pts[1, i]}, {self.pts[2, i]})\n"
        else:
            s += f"Maximum coordinates: {self.pts.max(axis=1)}\n"
            s += f"Minimum coordinates: {self.pts.min(axis=1)}\n"

        return s


class _WellMatrixConnection(NamedTuple):
    """One stretch of contact between a well cell and a rock matrix cell.

    A well is not conforming to the rock matrix mesh, so a well cell generally crosses
    several rock matrix cells and a rock matrix cell is generally crossed by several
    well cells. A connection is one such contact, and becomes one cell of the mortar
    grid coupling the two.

    """

    well_cell: int
    """Index of the well cell."""
    matrix_cell: int
    """Index of the rock matrix cell."""
    start: np.ndarray
    """``shape=(3,)`` Start point of the contact."""
    end: np.ndarray
    """``shape=(3,)`` End point of the contact."""

    @property
    def length(self) -> float:
        """Length of the contact."""
        return float(np.linalg.norm(self.end - self.start))


def _cell_half_spaces(
    sd: pp.Grid, cell: int, cell_faces_csc: sps.csc_matrix
) -> tuple[np.ndarray, np.ndarray]:
    """Represent a cell as an intersection of half-spaces.

    Parameters:
        sd: The grid the cell belongs to.
        cell: Index of the cell.
        cell_faces_csc: ``sd.cell_faces`` in CSC format, passed in to avoid repeated
            conversion when looping over cells.

    Returns:
        A tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(3, num_faces_of_cell)``

            Outward unit normal of each face of the cell.

        :obj:`~numpy.ndarray`: ``shape=(num_faces_of_cell,)``

            Plane constant of each face, so that the cell interior is the set of points
            ``x`` with ``normals.T @ x <= offsets`` for every face.

    """
    loc = slice(cell_faces_csc.indptr[cell], cell_faces_csc.indptr[cell + 1])
    faces = cell_faces_csc.indices[loc]
    signs = cell_faces_csc.data[loc]

    normals = signs * sd.face_normals[:, faces]
    normals = normals / np.linalg.norm(normals, axis=0)
    offsets = np.einsum("ij,ij->j", normals, sd.face_centers[:, faces])
    return normals, offsets


def _validate_convex_cell(
    cell: int,
    vertices: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tol: float,
) -> None:
    """Verify that a cell is convex and has planar faces.

    The segment-cell intersection in :func:`_segment_cell_interval` represents the cell
    as an intersection of half-spaces, which is exact only for a convex cell with planar
    faces. Both properties hold for simplices, Cartesian cells and Voronoi cells, but
    not for agglomerated cells, and planarity may fail for perturbed polyhedral grids.

    Parameters:
        cell: Index of the cell, used in the error message.
        vertices: ``shape=(3, num_vertices_of_cell)``

            Vertices of the cell.
        normals: ``shape=(3, num_faces_of_cell)``

            Outward unit normals of the faces of the cell.
        offsets: ``shape=(num_faces_of_cell,)``

            Plane constants of the faces of the cell.
        tol: Relative geometric tolerance, scaled by the diameter of the cell.

    Raises:
        ValueError: If the cell is not convex, or if one of its faces is not planar.

    """
    diameter = float(np.linalg.norm(vertices.max(axis=1) - vertices.min(axis=1)))
    abs_tol = tol * diameter

    # Signed distance from every vertex to every face plane. For a convex cell with
    # planar faces all vertices lie on the inner side of, or on, every face plane.
    signed_distance = normals.T @ vertices - offsets[:, None]

    if np.any(signed_distance > abs_tol):
        raise ValueError(
            f"Cell {cell} of the rock matrix grid is not convex, or has non-planar "
            "faces. The well-matrix intersection computation represents cells as "
            "intersections of half-spaces and is valid only for convex cells with "
            "planar faces."
        )


def _segment_cell_interval(
    start: np.ndarray,
    end: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tol: float,
) -> Optional[tuple[float, float]]:
    """Intersect a line segment with a convex cell.

    The cell is represented as an intersection of half-spaces, and the parameter
    interval ``[0, 1]`` of the segment is clipped against each of them in turn.

    A segment running along a face or an edge of the cell is reported with its full
    length, and a segment merely grazing a vertex is reported with zero length. A
    segment running along a shared face or edge is therefore claimed by every cell
    sharing it, which :func:`_distribute_shared_intervals` resolves.

    Parameters:
        start: ``shape=(3,)``

            Start point of the segment.
        end: ``shape=(3,)``

            End point of the segment.
        normals: ``shape=(3, num_faces_of_cell)``

            Outward unit normals of the faces of the cell.
        offsets: ``shape=(num_faces_of_cell,)``

            Plane constants of the faces of the cell.
        tol: Relative geometric tolerance, scaled by the length of the segment.

    Returns:
        The parameter interval ``(t_enter, t_exit)``, with ``0 <= t_enter <= t_exit <=
        1``, of the part of the segment inside the cell, or ``None`` if the segment does
        not intersect the cell.

    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0.0:
        return None
    abs_tol = tol * length

    # Signed distance from the start point to each face plane, positive outside, and the
    # rate at which that distance changes along the segment.
    distance = normals.T @ start - offsets
    rate = normals.T @ direction

    t_enter, t_exit = 0.0, 1.0
    for dist, dt in zip(distance, rate):
        if np.abs(dt) <= abs_tol:
            # The segment is parallel to this face. It is either entirely inside the
            # half-space (possibly lying in the face itself) or entirely outside it.
            if dist > abs_tol:
                return None
            continue
        t = -dist / dt
        if dt > 0:
            t_exit = min(t_exit, t)
        else:
            t_enter = max(t_enter, t)
        if t_enter > t_exit:
            return None

    return t_enter, t_exit


def _distribute_shared_intervals(
    intervals: dict[int, tuple[float, float]], tol: float
) -> dict[int, list[tuple[float, float]]]:
    """Distribute a segment over the cells it passes through, without double counting.

    A segment that runs along a face or an edge of the grid lies on the boundary of
    every cell sharing that face or edge, and each of them claims the same part of the
    segment. The shared part is divided equally between them, by splitting it into
    consecutive pieces of equal length rather than by attaching a weight to a shared
    piece. Dividing it geometrically keeps the division intact when the mortar grid
    built from these intervals recomputes its cell volumes from its own geometry.

    Note:
        Splitting equally is a modelling choice. The physical well lies on the interface
        between the cells, so no cell contains it, and the assignment is discontinuous
        in the well position regardless of the rule chosen: a well displaced
        infinitesimally to one side belongs entirely to the cell on that side. An equal
        split is the average of the two one-sided limits.

    Parameters:
        intervals: For each cell, the parameter interval of the segment inside it, as
            returned by :func:`_segment_cell_interval`.
        tol: Relative tolerance below which a piece is treated as empty.

    Returns:
        For each cell the segment touches, the parameter intervals attributed to it.
        Pieces that meet are merged, so a cell is normally given a single interval; it
        is given more than one only where its contact with the segment is genuinely
        disconnected, as when the segment leaves the cell and re-enters it.

    """
    if len(intervals) == 0:
        return {}

    cells = sorted(intervals)
    breakpoints = np.unique(np.array([t for cell in cells for t in intervals[cell]]))

    pieces: dict[int, list[tuple[float, float]]] = {cell: [] for cell in cells}
    shared = False
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        width = upper - lower
        if width <= tol:
            continue
        midpoint = 0.5 * (lower + upper)
        covering = [
            cell
            for cell in cells
            if intervals[cell][0] <= midpoint <= intervals[cell][1]
        ]
        if len(covering) == 0:
            continue
        if len(covering) > 1:
            shared = True
        # Hand each claiming cell a consecutive piece of equal length.
        share = width / len(covering)
        for position, cell in enumerate(covering):
            pieces[cell].append(
                (lower + position * share, lower + (position + 1) * share)
            )

    if shared:
        logger.warning(
            "A well segment runs along a face or an edge of the rock matrix grid. It "
            "lies on the boundary of several cells, and the shared length has been "
            "split equally between them."
        )

    return {
        cell: merged
        for cell, cell_pieces in pieces.items()
        if (merged := _merge_touching_intervals(cell_pieces, tol))
    }


def _merge_touching_intervals(
    pieces: list[tuple[float, float]], tol: float
) -> list[tuple[float, float]]:
    """Join intervals that meet, and drop those shorter than the tolerance.

    Parameters:
        pieces: Intervals along a segment, in increasing order.
        tol: Length below which an interval is dropped, and gap below which two
            intervals are considered to meet.

    Returns:
        The merged intervals.

    """
    merged: list[tuple[float, float]] = []
    for lower, upper in pieces:
        if merged and lower - merged[-1][1] <= tol:
            merged[-1] = (merged[-1][0], upper)
        else:
            merged.append((lower, upper))
    return [(a, b) for a, b in merged if b - a > tol]


def _well_segments(sd_w: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of the segments a well grid consists of.

    Parameters:
        sd_w: The well grid, of dimension one.

    Returns:
        A tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(3, sd_w.num_cells)``

            Start point of each segment.

        :obj:`~numpy.ndarray`: ``shape=(3, sd_w.num_cells)``

            End point of each segment.

    """
    cell_nodes = sd_w.cell_nodes()
    cells = np.arange(sd_w.num_cells)
    node_pairs = (
        cell_nodes.indices[
            pp.array_operations.expand_index_pointers(
                cell_nodes.indptr[cells], cell_nodes.indptr[cells + 1]
            )
        ]
        .reshape((-1, 2))
        .T
    )
    return sd_w.nodes[:, node_pairs[0]], sd_w.nodes[:, node_pairs[1]]


def _segment_connections(
    well_cell: int,
    start: np.ndarray,
    end: np.ndarray,
    tree: pp.adtree.ADTree,
    half_spaces: Callable[[int], tuple[np.ndarray, np.ndarray]],
    min_length: float,
    tol: float,
) -> list[_WellMatrixConnection]:
    """Find the contacts between one well cell and the rock matrix.

    The tree provides the cells whose bounding box the segment may enter, which are then
    intersected exactly.

    Parameters:
        well_cell: Index of the well cell the segment represents.
        start: ``shape=(3,)``

            Start point of the segment.
        end: ``shape=(3,)``

            End point of the segment.
        tree: Search tree over the cells of the rock matrix grid.
        half_spaces: Half space representation of a rock matrix cell, given its index.
        min_length: Minimum fraction of the segment for a contact to be kept.
        tol: Relative geometric tolerance.

    Returns:
        The contacts between this well cell and the rock matrix cells it passes through.

    """
    bounding_box = np.sort(np.vstack((start, end)), axis=0).ravel()
    candidates = tree.search(pp.adtree.ADTNode("well segment", bounding_box))

    intervals = {}
    for candidate in candidates:
        cell = tree.nodes[candidate].key
        interval = _segment_cell_interval(start, end, *half_spaces(cell), tol)
        if interval is not None:
            intervals[cell] = interval

    direction = end - start
    return [
        _WellMatrixConnection(
            well_cell, cell, start + lower * direction, start + upper * direction
        )
        for cell, pieces in _distribute_shared_intervals(intervals, min_length).items()
        for lower, upper in pieces
    ]


def _well_connections(
    sd_max: pp.Grid,
    sd_w: pp.Grid,
    tree: pp.adtree.ADTree,
    min_length: float,
    tol: float,
) -> list[_WellMatrixConnection]:
    """Find every contact between a well and the rock matrix.

    Parameters:
        sd_max: The rock matrix grid.
        sd_w: The well grid.
        tree: Search tree over the cells of the rock matrix grid.
        min_length: Minimum fraction of a well cell for a contact to be kept.
        tol: Relative geometric tolerance.

    Returns:
        The contacts, ordered by well cell.

    Raises:
        ValueError: If a rock matrix cell traversed by the well is not convex, or has
            non-planar faces.

    """
    cell_faces = sd_max.cell_faces.tocsc()
    cell_nodes = sd_max.cell_nodes().tocsc()

    def validated_half_spaces(cell: int) -> tuple[np.ndarray, np.ndarray]:
        """Half space representation of a rock matrix cell, checked for convexity."""
        normals, offsets = _cell_half_spaces(sd_max, cell, cell_faces)
        loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
        vertices = sd_max.nodes[:, cell_nodes.indices[loc]]
        _validate_convex_cell(cell, vertices, normals, offsets, tol)
        return normals, offsets

    start, end = _well_segments(sd_w)
    return [
        connection
        for well_cell, (segment_start, segment_end) in enumerate(zip(start.T, end.T))
        for connection in _segment_connections(
            well_cell,
            segment_start,
            segment_end,
            tree,
            validated_half_spaces,
            min_length,
            tol,
        )
    ]


def _connection_side_grid(connections: list[_WellMatrixConnection]) -> pp.Grid:
    """Build the grid whose cells are the well-matrix contacts.

    This becomes the side grid of the mortar grid, and therefore carries one set of
    interface unknowns per contact. Its cells are the contacts themselves, as disjoint
    segments: two contacts that meet along the well are still separate cells, since they
    belong to different rock matrix cells.

    Parameters:
        connections: The contacts between a well and the rock matrix.

    Returns:
        A one-dimensional grid with one cell per contact, whose cell volumes are the
        contact lengths.

    """
    num = len(connections)
    nodes = np.empty((3, 2 * num))
    nodes[:, 0::2] = np.array([c.start for c in connections]).T
    nodes[:, 1::2] = np.array([c.end for c in connections]).T

    # Each face of a one-dimensional grid is a single node, and each cell has the two
    # faces bounding it.
    face_nodes = sps.csc_matrix(
        (np.ones(2 * num, dtype=bool), (np.arange(2 * num), np.arange(2 * num))),
        shape=(2 * num, 2 * num),
    )
    # In case it becomes important, the sign convention assigned here is essentially
    # random.
    cell_faces = sps.csc_matrix(
        (
            np.tile([-1.0, 1.0], num),
            (np.arange(2 * num), np.repeat(np.arange(num), 2)),
        ),
        shape=(2 * num, num),
    )

    grid = pp.Grid(1, nodes, face_nodes, cell_faces, "well-matrix connections")
    grid.compute_geometry()
    return grid


def _connection_projections(
    sd_max: pp.Grid, sd_w: pp.Grid, connections: list[_WellMatrixConnection]
) -> dict[str, sps.csc_matrix]:
    """Build the projections between the neighbouring grids and the connections.

    Each connection touches exactly one cell on the rock matrix and will sides, and both
    the integrating (extensive) and non-integrating (intensive) maps are constructed.

    Note:
        The intensive maps come out as matrices of ones, which reads like a sum rather
        than an average. It is both: a connection lies inside a single cell on either
        side, so the average over the cells overlapping it is an average of one value.
        The extensive maps are the ones carrying a weight, because a rock matrix (or
        well) cell may host several connections and its share has to be divided between
        them.

    Parameters:
        sd_max: The rock matrix grid. sd_w: The well grid. connections: The contacts
        between the well and the rock matrix.

    Returns:
        The four projection matrices, keyed by the attribute of
        :class:`~porepy.grids.mortar_grid.MortarGrid` they are assigned to.

    """
    rows = np.arange(len(connections))
    lengths = np.array([c.length for c in connections])
    well_cells = np.array([c.well_cell for c in connections], dtype=int)
    matrix_cells = np.array([c.matrix_cell for c in connections], dtype=int)

    def matrix(data: np.ndarray, cols: np.ndarray, num_cols: int) -> sps.csc_matrix:
        return sps.csc_matrix((data, (rows, cols)), shape=(len(connections), num_cols))

    # Contact length relative to the total contact of the cell it is measured against.
    # A rock matrix cell may host contacts from several well cells; a well cell is
    # divided between the rock matrix cells it crosses.
    per_matrix_cell = np.bincount(
        matrix_cells, weights=lengths, minlength=sd_max.num_cells
    )
    per_well_cell = np.bincount(well_cells, weights=lengths, minlength=sd_w.num_cells)

    return {
        "_primary_to_mortar_avg": matrix(
            np.ones(len(connections)), matrix_cells, sd_max.num_cells
        ),
        "_primary_to_mortar_int": matrix(
            lengths / per_matrix_cell[matrix_cells], matrix_cells, sd_max.num_cells
        ),
        "_secondary_to_mortar_avg": matrix(
            np.ones(len(connections)), well_cells, sd_w.num_cells
        ),
        "_secondary_to_mortar_int": matrix(
            lengths / per_well_cell[well_cells], well_cells, sd_w.num_cells
        ),
    }


def _add_well_matrix_interface(
    mdg: pp.MixedDimensionalGrid,
    sd_max: pp.Grid,
    sd_w: pp.Grid,
    connections: list[_WellMatrixConnection],
) -> None:
    """Couple a well to the rock matrix through a new interface.

    The mortar grid has one cell per contact, so each contact carries its own set of
    interface unknowns. A well cell crossing several rock matrix cells therefore has one
    flux per crossing, each driven by the pressure of the cell it lies in, rather than a
    single flux driven by an average over them.

    Parameters:
        mdg: The mixed-dimensional grid the interface is added to.
        sd_max: The rock matrix grid.
        sd_w: The well grid.
        connections: The contacts between the well and the rock matrix.

    """
    side_grid = {
        pp.grids.mortar_grid.MortarSides.LEFT_SIDE: _connection_side_grid(connections)
    }
    mg = pp.MortarGrid(sd_w.dim, side_grid, codim=sd_max.dim - sd_w.dim)

    for name, projection in _connection_projections(sd_max, sd_w, connections).items():
        setattr(mg, name, projection)
    # _set_projections() will set the mortar-to-{primary, secondary} maps.
    mg._set_projections()
    mg.compute_geometry()

    mdg.add_interface(mg, (sd_max, sd_w), mg._primary_to_mortar_int)


def compute_well_rock_matrix_intersections(
    mdg: pp.MixedDimensionalGrid,
    cells: Optional[np.ndarray] = None,
    min_length: float = 1e-10,
    tol: float = 1e-5,
) -> None:
    """Compute intersections and add edge coupling between the well and the rock matrix.

    To be called after the well grids are constructed. The rock matrix cells are assumed
    to be convex with planar faces, which is verified for the cells the wells pass
    through; see :func:`_validate_convex_cell`. A single grid of highest dimension is
    assumed.

    Parameters:
        mdg: The mixed-dimensional grid containing all the elements.
        cells: ``default=None``

            A set of cells that might be considered to construct the ADTree. If it is
            not given the tree is constructed by using all the higher dimensional grid
            cells.
        min_length: ``default=1e-10``

            Minimum length of the part of a well segment inside a cell for the pair
            to be included in the mapping. Relative to the length of the well segment.
        tol: ``default=1e-5``

            Relative geometric tolerance used in the computations, scaled by a local
            length scale, the segment length or the cell diameter as appropriate.

    Raises:
        ValueError: If a rock matrix cell traversed by a well is not convex, or has
            non-planar faces.

    """
    # The rock matrix is assumed to be the single grid of highest dimension.
    sd_max: pp.Grid = mdg.subdomains(dim=mdg.dim_max())[0]
    tree = pp.adtree.ADTree(2 * sd_max.dim, sd_max.dim)
    tree.from_grid(sd_max, cells)

    # Wells are of co-dimension two relative to the rock matrix.
    well_subdomains = [
        sd for sd in mdg.subdomains(dim=sd_max.dim - 2) if hasattr(sd, "well_num")
    ]
    for sd_w in well_subdomains:
        connections = _well_connections(sd_max, sd_w, tree, min_length, tol)
        _add_well_matrix_interface(mdg, sd_max, sd_w, connections)


def _perpendicular_section(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project a cell onto the plane perpendicular to a direction.

    The projection is the shadow the cell casts along ``direction``, which for a convex
    cell is a convex polygon. Its shape is what the flow around a well sees, and is
    therefore the geometry entering the equivalent well radius.

    Parameters:
        vertices: ``shape=(3, num_vertices)``

            Vertices of the cell.
        direction: ``shape=(3,)``

            Direction of the well through the cell. Need not be a unit vector, and its
            sign is immaterial.

    Returns:
        ``shape=(num_corners, 2)``

        Corners of the projected polygon, in counter-clockwise order, in an arbitrary
        but orthonormal coordinate system of the plane. Only quantities invariant under
        rotation of that system should be read off it.

    """
    axis = direction / np.linalg.norm(direction)
    # Any vector not parallel to the axis spans a plane with it; the coordinate
    # direction in which the axis is smallest is guaranteed not to be parallel.
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)

    projected = np.column_stack([vertices.T @ first, vertices.T @ second])
    hull = ConvexHull(projected)
    return projected[hull.vertices]


def _polygon_principal_extents(polygon: np.ndarray) -> tuple[float, float]:
    """Measure a convex polygon by the sides of the rectangle equivalent to it.

    The polygon is summarised by two lengths: the rectangle with the same area and the
    same ratio between the principal second moments of area. This reproduces the sides
    exactly when the polygon is itself a rectangle, which is what makes the equivalent
    well radius reduce to Peaceman's on a Cartesian cell.

    Second moments of area are used rather than the spread of the corners, because the
    corners weigh a shape by how its boundary happens to be discretised: two nearly
    coincident corners, as arise when a well runs nearly parallel to a face of a
    tetrahedron, would count double.

    Note:
        The two lengths are equal whenever the second moments are isotropic, that is
        for any polygon with three-fold or higher symmetry: a square, an equilateral
        triangle, and hence the section of a regular tetrahedron cut along a symmetry
        axis. Such a shape distinguishes no direction, so no direction-dependent pair of
        lengths should be assigned to it, and both come out as the square root of the
        area. These are common enough sections that the equal case is easily mistaken
        for a degenerate one.

    Parameters:
        polygon: ``shape=(num_corners, 2)``

            Corners of a convex polygon, in counter-clockwise order.

    Returns:
        A tuple consisting of

        :obj:`float`: The longer side of the equivalent rectangle.

        :obj:`float`: The shorter side of the equivalent rectangle.

    """
    x, y = polygon[:, 0], polygon[:, 1]
    next_x, next_y = np.roll(x, -1), np.roll(y, -1)
    # Shoelace terms, from which area, centroid and second moments all follow.
    cross = x * next_y - next_x * y

    area = cross.sum() / 2
    centroid_x = ((x + next_x) * cross).sum() / (6 * area)
    centroid_y = ((y + next_y) * cross).sum() / (6 * area)

    # Second moments of area about the centroid, by the parallel axis theorem.
    second_xx = ((x**2 + x * next_x + next_x**2) * cross).sum() / 12
    second_yy = ((y**2 + y * next_y + next_y**2) * cross).sum() / 12
    second_xy = (
        (x * next_y + 2 * x * y + 2 * next_x * next_y + next_x * y) * cross
    ).sum() / 24
    covariance = (
        np.array(
            [
                [
                    second_xx - area * centroid_x**2,
                    second_xy - area * centroid_x * centroid_y,
                ],
                [
                    second_xy - area * centroid_x * centroid_y,
                    second_yy - area * centroid_y**2,
                ],
            ]
        )
        / area
    )

    # Eigenvalues are the variances along the principal axes, ascending.
    across_variance, along_variance = np.linalg.eigvalsh(covariance)
    aspect_ratio = np.sqrt(along_variance / across_variance)

    # The equivalent rectangle has the area of the polygon and this aspect ratio, so
    # the product of its sides is the area and their quotient the aspect ratio.
    longer_side = np.sqrt(area * aspect_ratio)
    shorter_side = np.sqrt(area / aspect_ratio)
    return float(longer_side), float(shorter_side)


def _equivalent_radius(vertices: np.ndarray, direction: np.ndarray) -> float:
    """Equivalent well radius of a cell traversed by a well.

    The equivalent radius is the distance from the well at which the analytical radial
    pressure equals the numerical cell average, and is what makes the Peaceman well
    index consistent with the discretisation. Peaceman derived it for a well running
    perpendicular to a rectangular block, where it is ``0.14`` times the diagonal of the
    block. The cell is measured here in the plane perpendicular to the well, which
    reproduces that expression on a Cartesian cell traversed along an axis and
    generalises it to arbitrary cell shapes and well orientations.

    The permeability is assumed isotropic. Anisotropy enters by weighting the two
    extents against each other, and the principal directions are fixed here by geometry
    alone, so introducing it does not require the permeability tensor to be
    decomposed.

    Note:
        Peaceman's derivation places the well at the centre of the block. A well is not
        conforming to the rock matrix mesh, so it generically passes through a cell off
        centre, and the radius is correspondingly approximate.

    Parameters:
        vertices: ``shape=(3, num_vertices)``

            Vertices of the cell.
        direction: ``shape=(3,)``

            Direction of the well through the cell. Need not be a unit vector, and its
            sign is immaterial.

    Returns:
        The equivalent well radius of the cell.

    """
    longer, shorter = _polygon_principal_extents(
        _perpendicular_section(vertices, direction)
    )
    return 0.14 * float(np.sqrt(longer**2 + shorter**2))


def _mortar_cell_directions(interface: pp.MortarGrid) -> np.ndarray:
    """Direction of the well along each cell of a well-matrix mortar grid.

    Each mortar cell is one contact, and is a straight segment, so its direction is the
    vector between the two nodes bounding it. The direction is recovered from the
    geometry of the mortar grid rather than stored, so that it cannot fall out of step
    with the grid it describes.

    Parameters:
        interface: A mortar grid built by
            :func:`compute_well_rock_matrix_intersections`.

    Returns:
        ``shape=(3, interface.num_cells)``

        Direction of each contact. Not normalised, and of arbitrary sign.

    """
    directions = []
    for side_grid in interface.side_grids.values():
        cell_nodes = side_grid.cell_nodes().tocsc()
        # A cell of a one-dimensional grid is bounded by exactly two nodes, and in
        # column order these come in pairs, one pair per cell. The coordinates
        # therefore split into an array whose three axes are the coordinate, the cell,
        # and which of the cell's two ends.
        endpoints = side_grid.nodes[:, cell_nodes.indices].reshape(3, -1, 2)
        # Subtracting the two ends gives the direction of every contact at once.
        directions.append(endpoints[:, :, 1] - endpoints[:, :, 0])
    return np.hstack(directions)


def well_equivalent_radii(
    mdg: pp.MixedDimensionalGrid, interface: pp.MortarGrid
) -> np.ndarray:
    """Equivalent well radius of each contact between a well and the rock matrix.

    The radius depends on the direction of the well through the cell, so a rock matrix
    cell crossed by two differently oriented well cells has two different radii. It is
    therefore a quantity of the mortar grid, not of the rock matrix grid.

    Parameters:
        mdg: The mixed-dimensional grid the interface belongs to.
        interface: A mortar grid built by
            :func:`compute_well_rock_matrix_intersections`.

    Returns:
        ``shape=(interface.num_cells,)``

        Equivalent well radius of each contact, ordered as the mortar cells.

    """
    sd_max, _ = mdg.interface_to_subdomain_pair(interface)
    cell_nodes = sd_max.cell_nodes().tocsc()

    def cell_vertices(cell: int) -> np.ndarray:
        """Vertices of a cell of the rock matrix grid."""
        loc = slice(cell_nodes.indptr[cell], cell_nodes.indptr[cell + 1])
        return sd_max.nodes[:, cell_nodes.indices[loc]]

    # Each mortar cell lies in a single rock matrix cell, so the intensive projection
    # has exactly one entry per row.
    matrix_cells = interface._primary_to_mortar_avg.tocsr().indices
    directions = _mortar_cell_directions(interface)
    return np.array(
        [
            _equivalent_radius(cell_vertices(cell), direction)
            for cell, direction in zip(matrix_cells, directions.T)
        ]
    )


def check_well_radius_resolution(
    radii: np.ndarray,
    path_lengths: np.ndarray,
    well_radius: float,
    well_num: int,
) -> None:
    """Verify that the mesh resolves the region around a well.

    The Peaceman well index is inversely proportional to ``log(r_e / r_w)``, which is
    singular when the equivalent radius equals the well radius and negative below it. A
    mesh fine enough for that to happen has resolved the near-well region to the point
    where the well index is no longer the right way to represent the well, and the model
    it belongs to is outside its range of validity.

    Refusing rather than clamping to a finite value is deliberate: a clamped index
    returns a plausible number that is merely wrong, which is far harder to notice than
    a failure.

    The condition is on the cross-section of the cell perpendicular to the well, not on
    its volume. For cells of moderate aspect ratio the two agree and only genuinely
    small cells are affected, but a cell elongated along the well can have a large
    volume and still be too narrow across.

    Parameters:
        radii: ``shape=(num_contacts,)``

            Equivalent well radius of each contact, as returned by
            :func:`well_equivalent_radii`.
        path_lengths: ``shape=(2, num_contacts)``

            Path length of the two ends of each contact, used to report where along the
            well the mesh is too fine.
        well_radius: Physical radius of the well.
        well_num: Index of the well, used to report which well is affected.

    Raises:
        ValueError: If some cell is so small that its equivalent radius does not exceed
            the well radius.

    """

    def stretch(offending: np.ndarray) -> str:
        """Where along the well the offending contacts lie."""
        lower = float(path_lengths[:, offending].min())
        upper = float(path_lengths[:, offending].max())
        return f"between {lower:.4g} and {upper:.4g} along it"

    too_small = radii <= well_radius
    if np.any(too_small):
        raise ValueError(
            f"Well {well_num} passes through rock matrix cells too fine for the "
            f"Peaceman well index. Over {radii[too_small].size} of its "
            f"{radii.size} contacts, {stretch(too_small)}, the equivalent well radius "
            f"falls to {float(radii[too_small].min()):.3e}, which does not exceed the "
            f"well radius {well_radius:.3e}. The well index is not defined in this "
            "regime. Coarsen the grid there, close the well over that stretch, or "
            "represent the well by a resolved geometry rather than a well index."
        )

    marginal = radii < _WELL_RADIUS_RESOLUTION_WARNING * well_radius
    if np.any(marginal):
        logger.warning(
            "The rock matrix is barely coarse enough around well %d: %d of %d "
            "contacts, %s, have an equivalent well radius below %d times the well "
            "radius, the smallest being %.3e. The Peaceman well index is increasingly "
            "inaccurate as this ratio approaches unity.",
            well_num,
            int(marginal.sum()),
            radii.size,
            stretch(marginal),
            _WELL_RADIUS_RESOLUTION_WARNING,
            float(radii[marginal].min()),
        )


def _ordered_well_cells(sd_w: pp.Grid, well_head: np.ndarray) -> np.ndarray:
    """Order the cells of a well grid along the well.

    The cells of a well grid are not stored in the order they appear along the well, so
    the order has to be recovered from the connectivity before path length can be
    accumulated.

    A well grid is generally not a single path. Meshing splits a well at every point
    where it meets a fracture, leaving one segment per stretch between intersections,
    all in the same grid. The segments are therefore walked in turn, each entered at
    whichever of its free ends lies closest to where the previous one ended.

    Parameters:
        sd_w: The well grid.
        well_head: ``shape=(3,)``

            A point identifying the end of the well from which path length is measured,
            normally the first vertex of the well trajectory. The free end nearest it
            is used, so the point need not lie on the well.

    Returns:
        ``shape=(sd_w.num_cells,)``

        Indices of the well cells, ordered from the end nearest ``well_head``.

    """
    cell_nodes = sd_w.cell_nodes().tocsc()
    nodes_of_cell = [
        cell_nodes.indices[cell_nodes.indptr[cell] : cell_nodes.indptr[cell + 1]]
        for cell in range(sd_w.num_cells)
    ]

    cells_of_node: dict[int, list[int]] = {}
    for cell, nodes in enumerate(nodes_of_cell):
        for node in nodes:
            cells_of_node.setdefault(int(node), []).append(cell)

    free_ends = [node for node, cells in cells_of_node.items() if len(cells) == 1]
    if len(free_ends) % 2 != 0:
        raise ValueError(
            f"The well grid of well {getattr(sd_w, 'well_num', '?')} has "
            f"{len(free_ends)} free ends. A well is a union of open segments and must "
            "have an even number, so path length along it is not defined."
        )

    order: list[int] = []
    visited: set[int] = set()
    unentered = set(free_ends)
    position = np.asarray(well_head, dtype=float)

    while unentered:
        # Enter the next segment at the free end nearest where the previous one ended.
        node = min(
            unentered, key=lambda end: np.linalg.norm(sd_w.nodes[:, end] - position)
        )
        unentered.discard(node)
        while True:
            candidates = [c for c in cells_of_node[node] if c not in visited]
            if not candidates:
                break
            cell = candidates[0]
            order.append(cell)
            visited.add(cell)
            node = int(nodes_of_cell[cell][nodes_of_cell[cell] != node][0])
        unentered.discard(node)
        position = sd_w.nodes[:, node]

    return np.array(order, dtype=int)


def well_cell_path_offsets(
    sd_w: pp.Grid, well_head: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Locate each well cell along the well.

    Path length, measured from the head of the well, is the natural coordinate for a
    well completion: it increases monotonically along the well whatever its
    orientation, which the elevation does not for a horizontal or undulating
    trajectory.

    Parameters:
        sd_w: The well grid.
        well_head: ``shape=(3,)``

            A point identifying the end of the well from which path length is measured.

    Returns:
        A tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(sd_w.num_cells,)``

            Path length from the head of the well to the near end of each cell.

        :obj:`~numpy.ndarray`: ``shape=(3, sd_w.num_cells)``

            The end of each cell nearer the head of the well.

    """
    order = _ordered_well_cells(sd_w, well_head)
    start, end = _well_segments(sd_w)

    offsets = np.empty(sd_w.num_cells)
    near_end = np.empty((3, sd_w.num_cells))
    travelled = 0.0
    position = np.asarray(well_head, dtype=float)
    for cell in order:
        # The stored orientation of a cell is arbitrary, so the near end is whichever
        # of its two ends lies closer to where the previous cell left off.
        to_start = np.linalg.norm(start[:, cell] - position)
        to_end = np.linalg.norm(end[:, cell] - position)
        entering_at_start = to_start <= to_end

        near_end[:, cell] = start[:, cell] if entering_at_start else end[:, cell]
        # Consecutive cells meet, so this is zero within a segment and between two
        # segments split apart at a fracture. It is the length of the unmeshed stretch
        # where a well leaves the domain and returns to it.
        travelled += float(min(to_start, to_end))
        offsets[cell] = travelled
        travelled += sd_w.cell_volumes[cell]
        position = end[:, cell] if entering_at_start else start[:, cell]

    return offsets, near_end


def well_contact_path_spans(
    mdg: pp.MixedDimensionalGrid, interface: pp.MortarGrid, well_head: np.ndarray
) -> np.ndarray:
    """Locate each well-matrix contact along the well.

    Parameters:
        mdg: The mixed-dimensional grid the interface belongs to.
        interface: A mortar grid built by
            :func:`compute_well_rock_matrix_intersections`.
        well_head: ``shape=(3,)``

            A point identifying the end of the well from which path length is measured.

    Returns:
        ``shape=(2, interface.num_cells)``

        Path length of the two ends of each contact, the nearer end first.

    """
    _, sd_w = mdg.interface_to_subdomain_pair(interface)
    offsets, near_end = well_cell_path_offsets(sd_w, well_head)

    # Each mortar cell lies in a single well cell, so the intensive projection has
    # exactly one entry per row.
    well_cells = interface._secondary_to_mortar_avg.tocsr().indices

    spans = np.empty((2, interface.num_cells))
    for side_grid in interface.side_grids.values():
        cell_nodes = side_grid.cell_nodes().tocsc()
        endpoints = side_grid.nodes[:, cell_nodes.indices].reshape(3, -1, 2)
        for index, well_cell in enumerate(well_cells):
            reference = near_end[:, well_cell, None]
            along = offsets[well_cell] + np.linalg.norm(
                endpoints[:, index, :] - reference, axis=0
            )
            spans[:, index] = np.sort(along)
    return spans


def _merged_open_intervals(
    open_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Reduce completion intervals to a disjoint, increasing list.

    Overlapping intervals would otherwise contribute twice to the open length of a
    contact and could make it appear more than fully open.

    Parameters:
        open_intervals: Intervals of path length along the well that are open to flow,
            in any order and possibly overlapping.

    Returns:
        The same set of points, as disjoint intervals in increasing order.

    """
    merged: list[tuple[float, float]] = []
    for lower, upper in sorted((min(a, b), max(a, b)) for a, b in open_intervals):
        if merged and lower <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return merged


def open_fractions(
    spans: np.ndarray, open_intervals: list[tuple[float, float]]
) -> np.ndarray:
    """Fraction of each contact that is open to flow.

    A well is cased over part of its length, and only the open parts exchange fluid
    with the surrounding rock. The boundary of an open interval generally falls inside
    a contact rather than between two of them, so the result is a fraction rather than
    a flag: were it rounded to one or zero, the well index would jump discontinuously
    as the rock matrix mesh is refined.

    Parameters:
        spans: ``shape=(2, num_contacts)``

            Path length of the two ends of each contact, as returned by
            :func:`well_contact_path_spans`.
        open_intervals: Intervals of path length along the well that are open to flow.
            An empty list closes the well along its whole length.

    Returns:
        ``shape=(num_contacts,)``

        Fraction of each contact that is open, between zero and one.

    """
    lower, upper = np.min(spans, axis=0), np.max(spans, axis=0)
    length = upper - lower

    open_length = np.zeros_like(length)
    for start, end in _merged_open_intervals(open_intervals):
        open_length += np.maximum(
            0.0, np.minimum(upper, end) - np.maximum(lower, start)
        )

    return np.divide(open_length, length, out=np.zeros_like(length), where=length > 0)


def well_number_of_interface(
    mdg: pp.MixedDimensionalGrid, interface: pp.MortarGrid
) -> int:
    """Identify the well coupled by a well interface.

    A well-matrix interface names its well directly, as the well grid is one of its
    two neighbours. A well-fracture interface does not: it couples the fracture to the
    point where the well meets it, and that point does not record which well it belongs
    to. The well is one step further, through the interface between the point and the
    well grid.

    Parameters:
        mdg: The mixed-dimensional grid the interface belongs to.
        interface: An interface of codimension two coupling a well to its surroundings.

    Returns:
        Index of the well, as carried by ``well_num`` of the well grid.

    Raises:
        ValueError: If no well can be reached from the interface.

    """
    _, sd_secondary = mdg.interface_to_subdomain_pair(interface)
    if getattr(sd_secondary, "well_num", -1) >= 0:
        return int(sd_secondary.well_num)

    for neighbour in mdg.subdomain_to_interfaces(sd_secondary):
        sd_primary, _ = mdg.interface_to_subdomain_pair(neighbour)
        if getattr(sd_primary, "well_num", -1) >= 0:
            return int(sd_primary.well_num)

    raise ValueError(
        "No well grid is reachable from this interface. A well-fracture interface is "
        "expected to couple a fracture to a point that is itself coupled to the well "
        "grid, and that second interface is missing."
    )
