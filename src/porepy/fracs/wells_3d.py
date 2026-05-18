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
from typing import Iterator, Optional, NamedTuple

from dataclasses import dataclass
from pathlib import Path
import gmsh
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

from .gmsh_interface import PhysicalNames

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
        index: Optional[int] = None,
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

        self._index: int = -1
        """Private index attribute. To be accessed by the property."""

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
        """
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
        """
        Warning:
            The original points (as given when the fracture was initialized) will
            *not* be preserved.

        Returns:
            A deep copy of the well with the same points and tags.

        """
        p = np.copy(self.pts)
        t = self.tags.copy()
        return Well(p, tags=t)

    def __str__(self) -> str:
        """Return a string representation of the well.

        Returns:
            A string representation of the well.
        """
        s = f"Well \n"  #  consisting of {self.num_segments()} segments.\n"
        # s += f"Well index: {self.index}"
        return s

    def __repr__(self) -> str:
        """Get a string representation of the well properties.

        Returns:
            A string representation of the well properties.

        """
        s = f"Well \n"  #  consisting of {self.num_segments()} segments.\n"
        # s += f"Well index: {self.index}\n"

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


def compute_well_fracture_intersections(
    well_network: WellNetwork3d, fracture_network: FractureNetwork3d
) -> None:
    """Compute well-fracture intersections.

    Store tags identifying which fracture and well segments each intersection
    corresponds.

    Note:
        A new set of points will be computed for each well, with original points and
        new intersection points. Note that original points may also correspond to an
        intersection with a fracture. Each well's tags are updated with the list
        "intersecting_fractures", with one list for each point in the new set. The
        entries of the inner list are the indices of the fractures intersecting the
        well at the corresponding point. Multiple fractures may intersect in any
        given point, but this might require special treatment elsewhere. The tags are
        crucial to the meshing of the well network.

    Parameters:
        well_network: Network of wells. Dimension 2 or 3 must match that of the
            fracture network.
        fracture_network: Three-dimensional fracture network.

    """
    nd = fracture_network.nd
    gmsh.initialize()
    fracture_tags = fracture_network.fractures_to_gmsh()
    segment_inds = well_network.to_gmsh()
    flattened_segment_inds = [item for sublist in segment_inds for item in sublist]

    domain_tag = fracture_network.domain_to_gmsh()

    gmsh.model.occ.synchronize()

    def process_well(w):
        segment_inds = w.to_gmsh()
        isect = well_fracture_intersection(segment_inds)

    def points_of_segments(segment_inds):
        return [
            gmsh.model.get_boundary([(nd - 2, t)], oriented=False) for t in segment_inds
        ]

    def parents_of_point(p_tag):
        parents = gmsh.model.get_adjacencies(0, p_tag, dim=-1)
        return parents

    split_wells = out_dim_tag_map[: len(flattened_segment_inds)]

    breakpoint()

    # NEXT STEPS:
    # 1) identify intersections, somehow store this information.
    #    This should be fragment first fractures (to get intersections), then fragment
    #    wells with fractures to get these intersections. We need a way to tag the
    #    well-something intersection points, using physical names.
    # 2) construct grids for the wells.

    nd = fracture_network.nd

    for well in well_network.wells:
        well_pts = np.empty((3, 0))
        well_tags = []
        for seg_ind, segment in well.segments():
            # Special treatment of endpoint of the segment, which should not be added
            # to the point array nor have its tag updated unless we are at the
            # endpoint of the well.
            ignore_endpoint_tag = seg_ind[1] < well.num_segments()
            # Keep track of information for this segment
            pts_seg = segment.copy()

            assert pts_seg.shape == (3, 2)
            pi = [gmsh.model.occ.addPoint(*segment[:, i], 0) for i in range(2)]
            l = gmsh.model.occ.addLine(pi[0], pi[1])
            gmsh.model.occ.synchronize()

            # Do a fragmentation to compute intersections.
            if len(fracture_tags) > 0:
                _, out_dim_tag_map = gmsh.model.occ.fragment(
                    [(nd - 2, l)],
                    [(nd - 1, t) for t in fracture_tags],
                    removeObject=False,
                    removeTool=False,
                )
            else:
                # No fractures in the network. Gmsh in this case returns empty output so
                # we manually set the output to be the input segment.
                out_dim_tag_map = [[(nd - 2, l)]]

            # The output dimension-tag map contains all output entities, with the first
            # entry representing the line segment (possibly fragmented into
            # sub-segments). The other entries represent the fractures, which we do not
            # need here.
            gmsh.model.occ.synchronize()

            # EK: This should not happen, but make the assertion to cover any unexpected
            # (to me) behavior from Gmsh's side.
            assert len(out_dim_tag_map[0]) > 0, (
                "Is both the fracture and well list empty?"
            )
            # If the first intersected object is not a segment, something is wrong,
            # likely on a technical (EK's assumptions on Gmsh?) level. Continuing makes
            # no sense.
            assert out_dim_tag_map[0][0][0] == nd - 2

            # To get the boundary points of the sub-segments, we extract the boundary of
            # each sub-segment. Some work is needed to actually extract the point tags.
            segment_points_dims = [
                gmsh.model.get_boundary([split_segment], oriented=False)
                for split_segment in out_dim_tag_map[0]
            ]
            segment_points = []
            for sub_segment in segment_points_dims:
                for p in sub_segment:
                    if p[0] == 0:  # point
                        segment_points.append(p[1])
            # Uniquify the points of this segment.
            unique_points = np.asarray(list(set(segment_points)))
            point_coordinates = []
            for p_tag in unique_points:
                x, y, z = gmsh.model.get_bounding_box(0, p_tag)[:3]
                point_coordinates.append(np.array([[x], [y], [z]]))
            sort_inds, sorted_pts = _argsort_points_along_line_segment(
                np.hstack(point_coordinates)
            )
            # For all points, find which fractures they are close to and take note.
            tags_seg = []
            for p_tag in unique_points[sort_inds]:
                frac_tag_log = []
                for fi, f_tag in enumerate(fracture_tags):
                    dist = gmsh.model.occ.get_distance(0, p_tag, nd - 1, f_tag)[0]
                    if dist < well_network.tol:
                        frac_tag_log.append(fracture_network.fractures[fi].index)
                tags_seg.append(np.array(frac_tag_log, dtype=int))

            stop_ind = sort_inds.size - ignore_endpoint_tag
            well_pts = np.hstack((well_pts, sorted_pts[:, :stop_ind]))
            # The last tag might change when it is used for the start point of the
            # next segment. Store remaining tags.
            for tag in tags_seg[:stop_ind]:
                well_tags.append(tag)
        # Overwrite old points and tags for this well
        well.pts = well_pts
        well.tags["intersecting_fractures"] = well_tags

    gmsh.finalize()


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
    # Extract the dimension of the rock matrix, assumed to be of highest dimension
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
        # get the cells of the 0d as segments (start, end)
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

        # primary to secondary map
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


def _argsort_points_along_line_segment(
    seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort point lying along a segment.

    Note:
        The sorting is done so that
        ``seg[d, inds[0]], seg[d,inds[1]], ..., seg[d, inds[-2]], seg[d,inds[-1]]``
        is monotone for at least one dimension ``d``. Ascending or descending order is
        determined by the values of the two end points.

    Parameters:
        seg: ``shape=(3, num_points)``

            Coordinates of the points to be sorted, assumed to lie on a straight line.

    Returns:
        Tuple with two elements.

        :obj:`numpy.ndarray`: ``shape=(num_points, 1)``

            Indices of the sorting.

        :obj:`numpy.ndarray`: ``shape=(3, num_points)``

            Sorted points.

    """
    # Find a dimension along which the points may be sorted (coordinates are not
    # constant):
    for dim in range(3):
        if not np.isclose(seg[dim, 0] - seg[dim, 1], 0):
            break
    # Perform sorting
    inds = np.argsort(seg[dim])
    # Invert if the original segment was in decreasing order
    if seg[dim, 0] > seg[dim, 1]:
        inds = inds[::-1]
    return inds, seg[:, inds]


def _intersection_subdomain(
    point: np.ndarray, mdg: pp.MixedDimensionalGrid
) -> pp.PointGrid:
    """Make a point subdomain and add to mdg.

    Parameters:
        point: ``shape=(3, 1)``:

            Intersection coordinates.
        mdg: The mixed-dimensional grid.

    Returns:
        Grid representing the subdomain at the intersection point.

    """
    sd = pp.PointGrid(point)
    sd.history.append("Well-fracture intersection grid")
    sd.compute_geometry()
    mdg.add_subdomains(sd)
    return sd


def _add_fracture_2_intersection_interface(
    sd_secondary: pp.Grid, frac_num: int, mdg: pp.MixedDimensionalGrid
) -> None:
    """Add an interface between a fracture and an intersection point.

    Does not check that the well lies *inside* a fracture cell and not on the face
    between two cells.

    Parameters:
        sd_secondary: Secondary subdomain grid, e.g., the (intersection) point grid.
        frac_num: Index of the fracture.
        mdg: Mixed-dimensional grid.

    """
    for sd in mdg.subdomains():
        if sd.frac_num == frac_num:
            sd_primary = sd
            break  # EK, is there a preferred method?
    cell_primary = sd_primary.closest_cell(sd_secondary.cell_centers)
    cell_secondary = np.array([0], dtype=int)

    cell_cell_map = sps.coo_matrix(
        (np.ones(1, dtype=bool), (cell_secondary, cell_primary)),
        shape=(sd_secondary.num_cells, sd_primary.num_cells),
    )
    _add_interface(0, sd_primary, sd_secondary, mdg, cell_cell_map)


def _add_well_2_intersection_interface(
    sd_primary: pp.Grid, sd_secondary: pp.Grid, mdg: pp.MixedDimensionalGrid
) -> None:
    """Add an interface between a well and an intersection subdomain.

    Parameters:
        sd_primary: Primary subdomain grid, e.g., the well grid.
        sd_secondary: Secondary subdomain grid, e.g., the intersection point grid.
        mdg: The mixed-dimensional grid.

    """
    cell_l = np.array([0], dtype=int)
    vec = sd_primary.face_centers - sd_secondary.cell_centers
    face_h = np.array([np.argmin(np.sum(np.power(vec, 2), axis=0))], dtype=int)
    face_cell_map = sps.coo_matrix(
        (np.ones(1, dtype=bool), (cell_l, face_h)),
        shape=(sd_secondary.num_cells, sd_primary.num_faces),
    )
    _add_interface(0, sd_primary, sd_secondary, mdg, face_cell_map)
