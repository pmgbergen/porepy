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

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d
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
        index: Optional[int] = None,
        tags: Optional[dict] = None,
    ) -> None:
        self.pts: np.ndarray = np.asarray(points, dtype=float)
        """``shape = (3, num_points)``

        Endpoints of each of the ``num_points - 1`` line segments of the new well."""
        self.orig_pts: np.ndarray = self.pts.copy()
        """``shape = (nd, num_points)``

        Original endpoints of the well. Copy kept in case the well geometry is modified.

        """
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

        # Set well index
        if index is not None:
            self.index = index

    @property
    def index(self) -> int:
        """
        Parameters:
            i: An integer representing the assigned index.

        Returns:
            The index of the well.

        """
        return self._index

    @index.setter
    def index(self, i: Optional[int] = None) -> None:
        if i is None:
            self._index = -1
        else:
            self._index = i

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

    def add_point(self, point: np.ndarray, ind: Optional[int] = None) -> None:
        """Add new point to ``self.pts``.

        Parameters:
            point: ``shape=(nd, 1)``

                Point to be added.
            ind: ``default=None``

                Index. If ``ind`` is not specified, the point is appended at the end of
                ``self.pts``. Otherwise, it is inserted between the old points ``ind``
                and ``ind + 1``.

        """
        if ind is None:
            self.pts = np.hstack((self.pts, point))
        else:
            self.pts = np.hstack((self.pts[:, :ind], point, self.pts[:, ind:]))

    def _mesh_size(
        self, segment_ind: Optional[tuple[int, int]] = None
    ) -> Optional[float]:
        """Return the mesh size for a well or one of its segments.

        Note:
            Can be overwritten to yield well specific, or segment specific values.
            Meshing of :class:`~WellNetwork` defaults to ``WellNetwork._mesh_size``
            if ``None`` is returned by this method.

        Parameters:
            segment_ind: ``default=None``

             Indices defining the segment, i.e., indices of the endpoints of the
             segment. If ``None``, the mesh size for the entire well is returned.

        Returns:
            ``None`` if the method is not overridden, otherwise it should return a
            ``float`` with the corresponding mesh size.

        """
        return None

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
        s = f"Well consisting of {self.num_segments()} segments.\n"
        s += f"Well index: {self.index}"
        return s

    def __repr__(self) -> str:
        """Get a string representation of the well properties.

        Returns:
            A string representation of the well properties.

        """
        s = f"Well consisting of {self.num_segments()} segments.\n"
        s += f"Well index: {self.index}\n"

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


class WellNetwork3d:
    """Collection of :class:`~Well` classes with geometrical information.

    Facilitates meshing of all wells in the network and their addition to a
    mixed-dimensional grid, see e.g., :meth:`~mesh` method.

    Parameters:
        domain: Domain specification.
        wells: ``default=None``

            List of wells in the network. If not empty, the constructor assigns indices
            to the wells corresponding to the order in this list.
        tol: ``default=1e-8``

            Geometric tolerance used in computations.
        parameters: ``default=None``

            Dictionary of parameters, e.g., for the meshing process.

    """

    def __init__(
        self,
        domain: pp.Domain,
        wells: Optional[list[Well]] = None,
        tol: float = 1e-8,
        parameters: Optional[dict] = None,
    ) -> None:
        self.domain: pp.Domain = domain
        """Domain specification."""

        self.well_dim: int = 1
        """All polyline wells have dimension 1."""

        self.wells: list[Well] = wells if wells is not None else []
        """List of wells in the network."""

        for i, w in enumerate(self.wells):
            w.index = i

        self.parameters: dict = {}
        """Dictionary of parameters, e.g. for the meshing process passed at
        instantiation. """
        if parameters is not None:
            self.parameters = parameters

        self.tol: float = tol
        """Geometric tolerance used in computations."""

        # Assign an empty tag dictionary
        self.tags: dict[str, list[bool]] = dict()

    def add(self, well: Well) -> None:
        """Add a well to the network.

        The well will be assigned a new index, higher than the maximum value
        currently found in the network.

        Parameters:
            well: Well to be added.

        """
        ind = np.array([w.index for w in self.wells])

        if ind.size > 0:
            well.index = np.max(ind) + 1
        else:
            well.index = 0
        self.wells.append(well)

    def _mesh_size(
        self, well: Well, segment_ind: Optional[tuple[int, int]] = None
    ) -> float:
        """Return the mesh size for a well or one of its segments.

        Parameters:
            well: Well for which to access mesh size.
            segment_ind: ``default=None``

                Indices defining the segment, i.e., indices of the endpoints of the
                segment. If ``None``, the mesh size for the entire well is returned.

        Returns:
            Mesh size for the :attr:`well` or one of its segments.

        """
        # TODO: IS: I have not concluded whether mesh size should be global (Network)
        #  or local (individual wells/segments). To be decided, with possible
        #  consequences for this function.
        size = well._mesh_size(segment_ind)
        if size is None:
            size = self.parameters["mesh_size"]
        return size

    def mesh(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Produce grids for the network's wells and add to existing ``mdg``.

        One grid is constructed for each sub-line extending between two fracture
        intersections. In the simplest case, the well is a (poly)-line with two end
        points, and a single grid is produced.

        Intersection grids are added for all intersection points between wells and
        fractures. Finally, edges are added between intersection points and both
        fractures and well segments.

        Example:
            Topology for well intersecting two fractures, terminating at the
            lowermost

            .. code:: python3

                            |
                sd_well_0    |
                            |
                            * e(sd_isec_0, sd_well_0)
                sd_isec_0    . * e(sd_isec_0, sd_frac_0)  ___________ sd_frac_0 (2d)
                            * e(sd_isec_0, sd_well_1)
                            |
                sd_well_1    |
                            |
                            * e(sd_isec_1, sd_well_1)
                sd_isec_1    . * e(sd_isec_1, sd_frac_1)  ___________ sd_frac_1 (2d)

            Note that all edge grids ``*`` are zero-dimensional, and that those
            connected with the fracture have co-dimension 2.

        Each point defining the well polyline is assumed to have a tag list stored in
        ``well.tags["intersecting_fractures"]``. An empty tag means the point does
        not correspond to a fracture intersection. An integer entry ``i`` identifies
        an intersection with the fracture with ``g.frac_num = i``. If the list
        contains multiple tags, the interpretation would be an intersection between
        the well and a fracture intersection line or point. This is not implemented.
        Points not corresponding to a fracture intersection, but merely representing
        a kink in the polyline, will not be represented by a zero-dimensional grid.
        Instead, the two neighboring segments are joined and a single *piecewise*
        linear grid is produced.

        This function may be split/restructured in the future. One possibility is to
        let gmsh do the actual meshing as done in the ``FractureNetwork`` classes.
        For now, this simplified approach is deemed sufficient.

        Parameters:
            mdg: Mixed-dimensional grid.

        """
        # Will be added as g.well_num for the well grids.
        well_num = 0
        for w in self.wells:
            tags_w = w.tags.get(
                "intersecting_fractures", [np.empty(0)] * w.pts.shape[1]
            )
            for t in tags_w:
                if t.size > 1:
                    raise NotImplementedError(
                        """Meshing of wells intersecting multiple fractures at
                        the same point is not implemented."""
                    )

            subline_endpoint_inds = [0]
            points_subline = np.empty((3, 0))
            # Tags for the endpoint faces of the well grid.
            endp_tip_tags = np.zeros(2, dtype=bool)
            endp_frac_tags = np.zeros(2, dtype=bool)
            # Special treatment of first end point. We need to check whether it
            # corresponds to an intersection. If so, we make an intersection grid and
            # add an edge between that grid and the fracture in question. Note that
            # the edge with the first well segment is added below.
            if tags_w[0].size > 0:
                sd_isec = _intersection_subdomain(w.pts[:, 0], mdg)
                _add_fracture_2_intersection_interface(sd_isec, tags_w[0], mdg)
                endp_frac_tags[0] = True

            for inds_seg, seg in w.segments():
                tags_seg = [tags_w[i] for i in inds_seg]
                length = pp.geometry.distances.point_pointset(seg[:, 0], seg[:, 1])
                num_pts = int(length / self._mesh_size(w, inds_seg))
                num_pts = max(num_pts, 2)
                points_loc = np.linspace(seg[:, 0], seg[:, 1], num_pts).T
                points_subline = np.hstack((points_subline, points_loc))

                # Check if the second end point is a fracture intersection. If not,
                # proceed to next segment unless we're at the well's second endpoint.
                if tags_seg[1].size == 0:
                    if inds_seg[1] == w.num_points() - 1:
                        # We're at the well end, and it corresponds to an internal tip.
                        endp_tip_tags[1] = True
                        endp_frac_tags[1] = False
                    else:
                        # Remove last point, since it is included in next iteration.
                        points_subline = np.reshape(points_subline[:, :-1], (3, -1))
                        continue

                # The end point is an intersection. Thus, we make a grid for the
                # subdomain consisting of the segments from endpoint_inds_loc[0] to
                # this end point and for the intersection point.
                subline_endpoint_inds.append(inds_seg[1])
                sd_w = pp.TensorGrid(np.arange(points_subline.shape[1]))
                sd_w.nodes = points_subline.copy()
                sd_w.compute_geometry()
                mdg.add_subdomains(sd_w)

                sd_w.well_num = well_num
                sd_w.name += " well " + str(well_num)
                sd_w.tags["parent_well_index"] = w.index
                well_num += 1

                # Add intersection grid and interfaces if the second segment point is
                # not a tip.
                if not endp_tip_tags[1]:
                    endp_frac_tags[1] = True
                    sd_isec = _intersection_subdomain(seg[:, 1], mdg)
                    sd_isec.tags["parent_well_index"] = w.index

                    # Add interfaces between intersection grid and both fracture and
                    # well grid.
                    _add_well_2_intersection_interface(sd_w, sd_isec, mdg)
                    _add_fracture_2_intersection_interface(
                        sd_isec, tags_w[inds_seg[1]], mdg
                    )

                # Further, if the new segment's first end point corresponds to an
                # intersection (as opposed to a global boundary or internal tip),
                # add the interface between this segment and that intersection.
                if endp_frac_tags[0]:
                    # Index for the intersection grid corresponding to the first
                    # endpoint of this subline. Last one if we have not added for the
                    # second endpoint, in which case it's the penultimate 0d grid
                    previous_ind = -1 - endp_frac_tags[1]
                    previous_g_isec = mdg.subdomains(dim=self.well_dim - 1)[
                        previous_ind
                    ]  # EK, is there a preferred method?
                    _add_well_2_intersection_interface(sd_w, previous_g_isec, mdg)

                # Finally, update tags for the well's faces (boundary, tip, fracture).
                bounding_planes = self.domain.polytope_from_bounding_box()
                boundary = np.zeros(2, dtype=bool)
                endp_inds = [0, -1]
                endpts = sd_w.face_centers[:, endp_inds]
                for plane in bounding_planes:
                    dist, _, _ = pp.geometry.distances.points_polygon(endpts, plane)
                    boundary = np.logical_or(boundary, np.isclose(dist, 0))

                endp_tip_tags[0] = np.logical_not(
                    np.logical_or(boundary[0], endp_frac_tags[0])
                )
                sd_w.tags["domain_boundary_faces"][endp_inds] = boundary
                sd_w.tags["tip_faces"][endp_inds] = endp_tip_tags
                sd_w.tags["fracture_faces"][endp_inds] = endp_frac_tags

                # Properly initalize the newly generated boundary grid.
                if (bg_w := mdg.subdomain_to_boundary_grid(sd_w)) is not None:
                    # Overwrite number of cells. This was initialized wrongly before
                    # sd_w.tags["domain_boundary_faces"] was set.
                    bg_w.num_cells = np.sum(boundary)
                    bg_w.set_projections()
                    bg_w.compute_geometry()
                # Reset the points for next iteration/subline.
                points_subline = np.empty((3, 0))
                subline_endpoint_inds = [inds_seg[1]]
                endp_tip_tags = np.zeros(2, dtype=bool)
                endp_frac_tags = np.array([1, 0], dtype=bool)
        for t in ["domain_boundary", "tip", "fracture"]:
            pp.utils.tags.add_node_tags_from_face_tags(mdg, t)

    def __repr__(self) -> str:
        """Return a string representation of the well network.

        Returns:
            A string representation of the well network.

        """
        # At the moment, it is unclear what more information should be included in
        # the string representation. We therefore implement only __repr__ (calls to
        # __str__ will be forwarded to __repr__).
        s = f"Well network consisting of {len(self.wells)} wells.\n"
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
            # Initiate tags for this segment, with empty elements for the endpoints
            tags_seg = [np.empty(0), np.empty(0)]
            for fracture, tag in zip(
                fracture_network.fractures, fracture_network.tags["boundary"]
            ):
                if tag:
                    continue
                pts_seg, tags_seg = _intersection_segment_fracture(
                    pts_seg, fracture, tags_seg, ignore_endpoint_tag
                )
            # Sort points of this segment
            sort_inds, sorted_pts = _argsort_points_along_line_segment(pts_seg)

            stop_ind = sort_inds.size - ignore_endpoint_tag
            well_pts = np.hstack((well_pts, sorted_pts[:, :stop_ind]))
            # The last tag might change when it is used for the start point of the
            # next segment. Store remaining tags in correct order
            for i in sort_inds[:stop_ind]:
                well_tags.append(tags_seg[i])
        # Overwrite old points and tags for this well
        well.pts = well_pts
        well.tags["intersecting_fractures"] = well_tags


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
            sd_w_cn.indices[pp.utils.mcolon.mcolon(first, second)].reshape((-1, 2)).T
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


def _intersection_segment_fracture(
    segment_points: np.ndarray,
    fracture: pp.PlaneFracture,
    tags: list[np.ndarray],
    ignore_endpoint_tag: bool,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute intersection between a single line segment and fracture.

    If no intersection exists (distance > 0), no updates are done to ``points`` or
    ``tags``. If the intersection is internal (distance between intersection and both
    endpoints > 0), the point is appended to ``segment_points`` and a tag appended to
    ``tags``. If the intersection is on one of the existing points, that point's tag
    is updated, unless ``ignore_endpoint_tag`` tag is ``True`` (see below).

    Parameters:
        segment_points: ``shape=(3, num_points)``

            Coordinates of the points on the line segment, sorted as
            ``[start, end, *any interior points]``.
        fracture: The plane fracture to be checked for intersections with the line
            segment.
        tags: ``len = num_points``

            Identify fractures (by ``fracture.index``) intersecting at each of the
            points in ``segment_points``.
        ignore_endpoint_tag: Whether to update the tag of the second endpoint. To be
            used when looping over a polyline. The last endpoint of this segment will be
            treated as the first endpoint of the next segment.
        tol: ``default=1e-8``

            Tolerance used to determine whether there is an intersection between the
            segment and the fracture.

    Returns:
        Tuple with two elements.

        :obj:`~numpy.ndarray`: ``shape=(3, num_points)``

            Updated coordinates of the points on the line segment, sorted as
            ``[start, end, *any interior points]``. Any new points have been appended.
        :obj:`list`: ``len = num_points``

            Updated tags.

    """
    distance, isec_pt = pp.geometry.distances.segments_polygon(
        segment_points[:, 0], segment_points[:, 1], fracture.pts
    )
    if distance > tol:
        # No intersection exists
        return segment_points, tags
    dist_endpt_isec = pp.geometry.distances.point_pointset(isec_pt, segment_points)
    ind_point_at_node = np.isclose(dist_endpt_isec, 0)

    if ignore_endpoint_tag and ind_point_at_node[1]:
        # No updates wanted, see parameter description of ignore_endpoint_tag
        return segment_points, tags
    elif np.any(ind_point_at_node):
        # The new intersection point already exists on the segment (endpoint or
        # internal). Point is not added, but tags are updated with the fracture index.
        ind_loc = ind_point_at_node.nonzero()[0][0]  # type: ignore
        if fracture.index is not None:
            tags[ind_loc] = np.append(tags[ind_loc], fracture.index)
    else:
        # New (internal) point. Store point and tag
        segment_points = np.hstack((segment_points, isec_pt))
        tags.append(np.array(fracture.index))
    return segment_points, tags


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


def _add_interface(
    dim: int,
    sd_primary: pp.Grid,
    sd_secondary: pp.Grid,
    mdg: pp.MixedDimensionalGrid,
    primary_secondary_map: sps.coo_matrix,
) -> None:
    """Utility method to add an interface to the mdg.

    Both grids should already be present in the mixed-dimensional grid.

    Parameters:
        sd_primary: Primary subdomain grid. In the context of this module, it represents
            a fracture or well.
        sd_secondary: Secondary subdomain grid. In the context of this module, it
            typically represents an intersection point.
        mdg: MixedDimensionalGrid to which the interface will be added.
        primary_secondary_map: Map between ``cells_l`` and either ``faces_h`` (codim=1)
            or ``cells_h`` (codim=2).

    """
    codim = sd_primary.dim - sd_secondary.dim
    subdomain_pair = (sd_primary, sd_secondary)
    side_g = {pp.grids.mortar_grid.MortarSides.LEFT_SIDE: sd_secondary.copy()}
    mg = pp.MortarGrid(dim, side_g, primary_secondary_map, codim=codim)
    mg._primary_to_mortar_int = primary_secondary_map
    mg.compute_geometry()
    mdg.add_interface(mg, subdomain_pair, primary_secondary_map)
