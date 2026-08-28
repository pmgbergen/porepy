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
from typing import Iterator, Optional

import gmsh
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

# Module-wide logger
logger = logging.getLogger(__name__)


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


def _segment_cell_interval(
    start: np.ndarray,
    end: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tol: float,
) -> Optional[tuple[float, float]]:
    """Intersect a line segment with a convex cell.

    The cell is represented as an intersection of half-spaces, and the parameter
    interval ``[0, 1]`` of the segment is clipped against each of them in turn. Unlike
    an approach based on explicit segment-face intersections, this treats a segment that
    touches the boundary of the cell as an ordinary case rather than a special one: a
    segment running along a face or an edge of the cell is reported with its full
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
) -> dict[int, float]:
    """Distribute a segment over the cells it passes through, without double counting.

    A segment that runs along a face or an edge of the grid lies on the boundary of
    every cell sharing that face or edge, and each of them claims the same part of the
    segment. The overlapping part is split equally between them, which is symmetric in
    the sharing cells and preserves the total: the distributed fractions sum to the
    fraction of the segment covered by at least one cell.

    Note:
        Splitting equally is a modelling choice. The physical well lies on the interface
        between the cells, so no cell contains it, and the assignment is discontinuous
        in the well position regardless of the rule chosen: a well displaced
        infinitesimally to one side belongs entirely to the cell on that side. An equal
        split is the average of the two one-sided limits.

    Parameters:
        intervals: For each cell, the parameter interval of the segment inside it, as
            returned by :func:`_segment_cell_interval`.
        tol: Relative tolerance below which a sub-interval is treated as empty.

    Returns:
        For each cell, the fraction of the segment length attributed to it. Cells whose
        attributed fraction is zero are omitted.

    """
    if len(intervals) == 0:
        return {}

    cells = list(intervals)
    breakpoints = np.unique(np.array([t for cell in cells for t in intervals[cell]]))

    fractions: dict[int, float] = {cell: 0.0 for cell in cells}
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
        for cell in covering:
            fractions[cell] += width / len(covering)

    if shared:
        logger.warning(
            "A well segment runs along a face or an edge of the rock matrix grid. It "
            "lies on the boundary of several cells, and the shared length has been "
            "split equally between them."
        )

    return {cell: f for cell, f in fractions.items() if f > tol}


def compute_well_rock_matrix_intersections(
    mdg: pp.MixedDimensionalGrid,
    cells: Optional[np.ndarray] = None,
    min_length: float = 1e-10,
    tol: float = 1e-5,
) -> None:
    """Compute intersections and add edge coupling between the well and the rock matrix.

    To be called after the well grids are constructed. We are assuming convex cells
    and a single high dimensional grid. To speed up the geometrical computation we
    construct an ``ADTree``.

    Parameters:
        mdg: The mixed-dimensional grid containing all the elements.
        cells: ``default=None``

            A set of cells that might be considered to construct the ADTree. If it is
            not given the tree is constructed by using all the higher dimensional grid
            cells.
        min_length: ``default=1e-10``

            Minimum length a segment that intersect a cell needs to have to be
            considered in the mapping.
        tol: ``default=1e-5``

            Geometric tolerance used in the computations.

    """
    # Extract the dimension of the rock matrix, assumed to be of highest dimension.
    dim_max: int = mdg.dim_max()
    # We assume only one single higher dimensional grid, needed for the ADTree.
    sd_max: pp.Grid = mdg.subdomains(dim=dim_max)[0]
    # Construct an ADTree for fast computation.
    tree = pp.adtree.ADTree(2 * sd_max.dim, sd_max.dim)
    tree.from_grid(sd_max, cells)

    # Extract the grids of the wells of co-dimension 2.
    well_subdomains: list[pp.Grid] = [
        g for g in mdg.subdomains(dim=dim_max - 2) if hasattr(g, "well_num")
    ]

    # Pre-compute some well information.
    nodes_w = []
    for sd_w in well_subdomains:
        sd_w_cn = sd_w.cell_nodes()
        sd_w_cells = np.arange(sd_w.num_cells)
        # Get the cells of the 0d as segments (start, end).
        first = sd_w_cn.indptr[sd_w_cells]
        second = sd_w_cn.indptr[sd_w_cells + 1]

        nodes_w.append(
            sd_w_cn.indices[pp.array_operations.expand_index_pointers(first, second)]
            .reshape((-1, 2))
            .T
        )

    # Operate on the rock matrix grid.
    faces, cells, _ = sparse_array_to_row_col_data(sd_max.cell_faces.tocsc())
    cells_order = np.argsort(cells)  # type: ignore
    faces = faces[cells_order]

    nodes, *_ = sparse_array_to_row_col_data(sd_max.face_nodes)
    indptr = sd_max.face_nodes.indptr

    # Loop on all the well grids.
    for sd_w, n_w in zip(well_subdomains, nodes_w):
        # Extract the start and end point of the segments.
        start = sd_w.nodes[:, n_w[0]]
        end = sd_w.nodes[:, n_w[1]]

        # Lists for the cell_cell_map.
        primary_secondary_I, primary_secondary_J, primary_secondary_data = [], [], []

        # Operate on the segments.
        for seg_id, (seg_start, seg_end) in enumerate(zip(start.T, end.T)):
            # Create the box for the segment by ordering its start and end.
            box = np.sort(np.vstack((seg_start, seg_end)), axis=0).ravel()
            # Extract the id of the ad nodes.
            seg_adnodes = tree.search(pp.adtree.ADTNode("dummy_node", box))
            # Extract the key of the ad nodes which is the cell id.
            seg_cells = [tree.nodes[n].key for n in seg_adnodes]
            # Loop on all the higher dimensional cells.
            for c in seg_cells:
                # For the current cell retrieve its faces.
                loc = slice(
                    sd_max.cell_faces.indptr[c], sd_max.cell_faces.indptr[c + 1]
                )
                faces_loc = faces[loc]
                # Get the local nodes, face based.
                poly = np.array(
                    [
                        sd_max.nodes[:, nodes[indptr[f] : indptr[f + 1]]]
                        for f in faces_loc
                    ]
                )
                # Compute the intersections between the segment and the current higher-
                # dimensional cell.
                _, _, _, ratio = pp.intersections.segments_polyhedron(
                    seg_start, seg_end, poly, tol
                )
                # Store the requested information to build the projection operator.
                if ratio > min_length:
                    primary_secondary_I += [seg_id]
                    primary_secondary_J += [c]
                    primary_secondary_data += ratio.tolist()

        # Primary to secondary map.
        primary_secondary_map = sps.csc_matrix(
            (primary_secondary_data, (primary_secondary_I, primary_secondary_J)),
            shape=(sd_w.num_cells, sd_max.num_cells),
        )

        # Add a new edge to the mixed-dimensional grid.

        # Create the mortar grid.
        side_g = {pp.grids.mortar_grid.MortarSides.LEFT_SIDE: sd_w.copy()}
        mg = pp.MortarGrid(sd_w.dim, side_g, codim=sd_max.dim - sd_w.dim)
        # Set the maps.
        mg._primary_to_mortar_int = primary_secondary_map
        mg._primary_to_mortar_avg = primary_secondary_map.copy()
        mg._secondary_to_mortar_int = sps.diags(np.ones(sd_w.num_cells), format="csc")
        mg._secondary_to_mortar_avg = sps.diags(np.ones(sd_w.num_cells), format="csc")
        mg._set_projections()
        # Compute the geometry and save the mortar grid.
        mg.compute_geometry()

        mdg.add_interface(mg, (sd_max, sd_w), primary_secondary_map)
