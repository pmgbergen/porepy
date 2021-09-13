"""
Module for well representation in Well and WellNetworks.

A well is a polyline of at least nsegs=1 segment defined through a list of
npts=nsegs+1 points. Wells are connected in a network.

After defining and meshing a fracture network, the wells may be added to the gb
by
    compute_well_fracture_intersections(well_network, fracture_network)
    well_network.mesh(gb)
"""
import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["gridding"]


class Well:
    """Class representing a single well as a polyline embedded in 3D space.

    The fracture is defined by its vertexes. It contains various utility
    methods, mainly intended for use together with the WellNetwork3d class.

    Attributes:
        p (np.ndarray, 3 x npt): Fracture vertices. Will be stored in a CCW
            order (or CW, depending on which side it is viewed from).
        orig_p (np.ndarray, 3 x npt): Original fracture vertices, kept in case
            the fracture geometry for some reason is modified.
        index (int): Index of well. Intended used in WellNetwork3d.
        tags
    """

    def __init__(
        self,
        points: np.ndarray,
        index: Optional[int] = None,
        tags: Optional[Dict] = None,
    ) -> None:
        """Initialize fractures.

        __init__ defines the points of the well.

        Parameters:
            points (np.ndarray-like, 3 x npt): End points of each of the npt-1
                line segments of the new well.
            index (int, optional): Index of well. Defaults to None.
            tags (dictionary, optional): may e.g. identify different types
                of points. In particular, tags["intersecting_fractures"] has
                length npt and will be used to identify which fracture(s)
                intersects each of the points in points.

        """
        self.p = np.asarray(points, dtype=float)
        self.orig_p = self.p.copy()
        self.dim = 1
        # Set well index
        self.index = index
        # Initialize tag dictionary.
        if tags is None:
            self.tags = {}
        else:
            self.tags = tags

    def set_index(self, i: int) -> None:
        """Set index of this fracture.

        Parameters:
            i (int): Index.

        """
        self.index = i

    def segments(self) -> Iterator[Tuple[List[int], np.ndarray]]:
        """
        Iterate over the segments defined through segment indices and endpoints.
        """
        for i in range(self.num_segments()):
            segment_inds = [i, i + 1]
            endpoints = self.p[:, segment_inds]
            yield segment_inds, endpoints

    def num_points(self) -> int:
        return self.p.shape[1]

    def num_segments(self) -> int:
        return self.num_points() - 1

    def add_point(self, point: np.ndarray, ind: int = None) -> None:
        """Add new pts (3 x 1) to self.p.
        If ind is not specified, the point is appended at the end of
        self.p. Otherwise, it is inserted between the old points number ind
        and ind + 1.
        """
        if ind is None:
            self.p = np.hstack((self.p, point))
        else:
            self.p = np.hstack((self.p[:, :ind], point, self.p[:, ind:]))

    def _mesh_size(self, segment_ind: Tuple[int] = None) -> Optional[float]:
        """Return the mesh size for a well or one of its segments.

        Parameters:
            segment_ind (tuple of ints): indices defining the segment.

        Can be overwritten to yield well specific or segment specific values.
        Meshing of WellNetwork defaults to WellNetwork._mesh_size if None is
        returned by this method.
        """
        return None

    def copy(self):
        """Return a deep copy of the fracture.

        Note that the original points (as given when the fracture was
        initialized) will *not* be preserved.

        Returns:
            Fracture with the same points.

        """
        p = np.copy(self.p)
        t = self.tags.copy()
        return Well(p, tags=t)


class WellNetwork3d:
    """
    Collection of Wells with geometrical information.

    Facilitates meshing of all wells in the network and their addition to
    a grid bucket, see mesh method.

    Attributes:
        _wells (list of Fracture): All fractures forming the network.
        intersections (list of Intersection): All known intersections in the
            network.
        tol (double): Geometric tolerance used in computations.
        domain (dictionary): External bounding box. See
            impose_external_boundary() for details.
        tags (dictionary): Tags used on Wells.
    """

    def __init__(
        self,
        wells: Optional[List[Well]] = None,
        domain: Optional[Dict[str, float]] = None,
        tol: float = 1e-8,
        parameters: Optional[Dict] = None,
    ) -> None:
        """Initialize well network.

        Generate network from specified wells. The wells will have their index
        set (and existing values overridden), according to the ordering in the
        input list.

        Initialization sets most fields (see attributes) to None.

        Parameters:
            wells (list of Well, optional): Wells that make up the network.
                Defaults to None, which will create a domain empty of wells.
            domain (dictionary): Domain specification.
            tol (double, optional): Tolerance used in geometric tolerations. Defaults
                to 1e-8.
            parameters (dictionary, optional): E.g. mesh parameters.

        """
        self.well_dim = 1
        self._wells = []

        if wells is not None:
            for w in wells:
                self._wells.append(w.copy())

        for i, w in enumerate(self._wells):
            w.set_index(i)
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters

        if domain is None:
            domain = {}
        self.domain: Dict = domain

        # Assign an empty tag dictionary
        self.tags: Dict[str, List[bool]] = {}

    def add(self, w: Well):
        """Add a well to the network.

        The well will be assigned a new index, higher than the maximum
        value currently found in the network.

        Parameters:
            w (Well): Well to be added.

        """
        ind = np.array([w.index for w in self._wells])

        if ind.size > 0:
            w.set_index(np.max(ind) + 1)
        else:
            w.set_index(0)
        self._wells.append(w)

    def _mesh_size(self, w: Well, segment_ind: Tuple[int] = None) -> float:
        """Return the mesh size for a well or one of its segments.

        Parameters:
            w (Well): Well for which to access mesh size.
            segment_ind (tuple of ints): indices defining the segment.

        TODO: IS: I have not concluded whether mesh size should be global (Network)
        or local (individual wells/segments). To be decided, with possible
        consequences for this function.
        """
        size = w._mesh_size(segment_ind)
        if size is None:
            size = self.parameters["mesh_size"]
        return size

    def mesh(self, gb: pp.GridBucket) -> None:
        """Produce grids for the network's wells and add to existing grid bucket.

        One grid is constructed for each subline extending between two fracture
        intersections. In the simplest case, the well is a (poly)line with two
        end points, and a single grid is produced.
        Intersection grids are added for all intersection points between wells
        and fractures. Finally, edges are added between intersection points and
        both fractures and well segments.
        Example topology for well intersecting two fractures, terminating at the
        lowermost:

                    |
        g_well_0    |
                    |
                    * e(g_isec_0, g_well_0)
        g_isec_0    . * e(g_isec_0, g_frac_0)  ___________ g_frac_0 (2d)
                    * e(g_isec_0, g_well_1)
                    |
        g_well_1    |
                    |
                    * e(g_isec_1, g_well_1)
        g_isec_1    . * e(g_isec_1, g_frac_1)  ____________ g_frac_1 (2d)

        Note that all edge grids (*) are zero-dimensional, and that those
        connected with the fracture have co-dimension 2!

        Each point defining the well polyline is assumed to have a tag list
        stored in w.tags["intersecting_fractures"]. An empty tag means the point
        does not correspond to a fracture intersection. An integer entry i
        identifies an intersection with the fracture with g.frac_num = i.
        If the list contains multiple tags, the interpretation would be an
        intersection between the well and a fracture intersection line or point.
        This is not implemented. Points not corresponding to a fracture
        intersection, but merely representing a kink in the polyline, will
        not be represented by a 0d grid. Rather, the two neighbouring segments
        are joined and a single _piecewise_ linear grid is produced.

        This function may be split/restructured in the future. One possibility
        is to let gmsh do the actual meshing as done in the FractureNetwork
        classes. For now, this simplified approach is deemed sufficient.
        """
        # Will be added as g.well_num for the well grids
        well_num = 0
        for w in self._wells:
            tags_w = w.tags["intersecting_fractures"]
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
            # corresponds to an intersection. If so, we make an intersection grid and add an
            # edge between that grid and the fracture in question. Note that
            # the edge with the first well segment is added below.
            if tags_w[0].size > 0:
                g_isec = _intersection_grid(w.p[:, 0], gb)
                _add_fracture_2_intersection_edge(g_isec, tags_w[0], gb)
                endp_frac_tags[0] = True

            for inds_seg, seg in w.segments():
                tags_seg = [tags_w[i] for i in inds_seg]
                length = pp.geometry.distances.point_pointset(seg[:, 0], seg[:, 1])
                num_pts = int(length / self._mesh_size(w, inds_seg[0]))
                num_pts = max(num_pts, 2)
                points_loc = np.linspace(seg[:, 0], seg[:, 1], num_pts).T
                points_subline = np.hstack((points_subline, points_loc))

                # Check if the second end point is a fracture intersection. If
                # not, proceed to next segment unless we're at the well's second
                # endpoint

                if tags_seg[1].size == 0:
                    if inds_seg[1] == w.num_points() - 1:
                        # We're at the well end and it corresponds to an internal
                        # tip
                        endp_tip_tags[1] = True
                        endp_frac_tags[1] = False
                    else:
                        # Remove last point, since it is included in next iteration
                        points_subline = np.reshape(points_subline[:, :-1], (3, -1))
                        continue

                # The end point is an intersection. Thus, we make a grid for
                # the subdomain consisting of the segments from
                # endpoint_inds_loc[0] to this end point and for the intersection
                # point.
                subline_endpoint_inds.append(inds_seg[1])
                g_w = pp.TensorGrid(np.arange(points_subline.shape[1]))
                g_w.nodes = points_subline.copy()
                g_w.compute_geometry()
                gb.add_nodes(g_w)
                g_w.well_num = well_num
                well_num += 1

                # Add intersection grid and interfaces if the second segment
                # point is not a tip
                if not endp_tip_tags[1]:
                    endp_frac_tags[1] = True
                    g_isec = _intersection_grid(seg[:, 1], gb)

                    # Add interfaces between intersection grid and both fracture and
                    # well grid.
                    _add_well_2_intersection_edge(g_isec, g_w, gb)
                    _add_fracture_2_intersection_edge(g_isec, tags_w[inds_seg[1]], gb)
                # Further, if the new segment's first end point corresponds to
                # an intersection (as opposed to a global boundary or internal tip),
                # add the interface between this segment and that intersection.
                if endp_frac_tags[0]:
                    # Index for the intersection grid corresponding to the first
                    # endpoint of this subline. Last one if we have not added
                    # for the second endpoint, in which case it's the penultimate 0d grid
                    previous_ind = -1 - endp_frac_tags[1]
                    previous_g_isec = gb.grids_of_dimension(self.well_dim - 1)[
                        previous_ind
                    ]  # EK, is there a preferred method?
                    _add_well_2_intersection_edge(previous_g_isec, g_w, gb)

                # Finally, update tags for the well's faces (boundary, tip, fracture).
                bounding_planes = (
                    pp.geometry.bounding_box.make_bounding_planes_from_box(self.domain)
                )
                boundary = np.zeros(2, dtype=bool)
                endp_inds = [0, -1]
                endpts = g_w.face_centers[:, endp_inds]
                for plane in bounding_planes:
                    dist, _, _ = pp.geometry.distances.points_polygon(endpts, plane)
                    boundary = np.logical_or(boundary, np.isclose(dist, 0))

                endp_tip_tags[0] = np.logical_not(
                    np.logical_or(boundary[0], endp_frac_tags[0])
                )
                g_w.tags["domain_boundary_faces"][endp_inds] = boundary
                g_w.tags["tip_faces"][endp_inds] = endp_tip_tags
                g_w.tags["fracture_faces"][endp_inds] = endp_frac_tags
                # Node tags? Check that all cases are covered ind endp_face_tags

                # Reset the points for next iteration/subline.
                points_subline = np.empty((3, 0))
                subline_endpoint_inds = [inds_seg[1]]
                endp_tip_tags = np.zeros(2, dtype=bool)
                endp_frac_tags = np.array([1, 0], dtype=bool)
        for t in ["domain_boundary", "tip", "fracture"]:
            pp.utils.tags.add_node_tags_from_face_tags(gb, t)


def compute_well_fracture_intersections(
    well_network: WellNetwork3d, fracture_network: pp.FractureNetwork3d
):
    """Compute intersections and store tags identifying which fracture
    and well segments each intersection corresponds.

    Parameters:
        well_network containing the wells, which are assumed to have at least
            two points each (in well.p). Intersection points and tags are added
            and updated in-place.
        fracture_network containing the (3d) fractures.


    A new set of points will be computed for each well, with original points
    and new intersection points. Note that original points may also correspond
    to an intersection with a fracture. Each well's tags are updated with the list
    "intersecting_fractures", with one list for each point in the new set. The
    entries of the inner list are the indices of the fractures intersecting the
    well at the corresponding point. Multiple fractures may intersect in any
    given point, but this might require special treatment elsewhere.
    The tags are crucial to the meshing of the well network.
    """

    for well in well_network._wells:
        well_pts = np.empty((3, 0))
        well_tags = []
        for seg_ind, segment in well.segments():
            # Special treatment of endpoint of the segment, which should not
            # be added to the point array nor have its tag updated unless we are
            # at the endpoint of the well.
            ignore_endpoint_tag = seg_ind[1] < well.num_segments()
            # Keep track of information for this segment
            pts_seg = segment.copy()
            # Initiate tags for this segment, with empty elements for the endpoints
            tags_seg = [np.empty(0), np.empty(0)]
            for fracture, tag in zip(
                fracture_network._fractures, fracture_network.tags["boundary"]
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
            # The last tag might change when it is used for the start point of
            # the next segment. Store remaining tags in correct order
            for i in sort_inds[:stop_ind]:
                well_tags.append(tags_seg[i])
        # Overwrite old points and tags for this well
        well.p = well_pts
        well.tags["intersecting_fractures"] = well_tags


def _argsort_points_along_line_segment(
    seg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort point lying along a segment.

    Parameters:
        seg (array): 3 x npts coordinates of the points to be sorted, assumed
            to lie on a straight line.

    Returns:
        array (npts x 1): Indices of the sorting.
        array (3 x npts): Sorted points.

    The sorting is done so that
    seg[d, inds[0]], seg[d,inds[1]], ..., seg[d,inds[-2]], seg[d,inds[-1]]
    is monotone for at least one dimension d. Ascending or descending order
    is determined by the values of the two end points.
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
    fracture: pp.Fracture,
    tags: List[np.ndarray],
    ignore_endpoint_tag: bool,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute intersection between a single line segment and fracture.

    Computes intersection point. If no intersection exists (distance > 0), no updates are
    done to points or tags. If the intersection is internal (distance between intersection
    and both endpoints > 0), the point is appended to segment_points and a tag appended to
    tags. If the intersection is on one of the existing points, that point's tag is updated,
    unless ignore_endpoint_tag tag is True (see below).

    Parameters:
        segment_points (array, 3 x npts): coordinates of the points on the line segment,
            sorted as [start, end, *any interior points]
        fracture (pp.Fracture): the fracture to be checked for intersections
            with the line segment.
        tags (list of length npts): identifies fractures (by fracture.index)
            intersecting at each of the points in segment_points.
        ingore_endpoint_tag (bool): whether to update the tag of the second endpoint
            or not. To be used when looping over a polyline. The last endpoint
            of this segment will be treated as the first endpoint of the next
            segment.
        tol (float): Used to determine whether there is an intersection between
            the segment and the fracture.

    Returns:
        segment_points (array, 3 x npts): updated coordinates of the points on
            the line segment, sorted as [start, end, *any interior points].
            Any new points have been appended.
        tags (list of length npts): updated tags.
    """
    distance, isec_pt = pp.geometry.distances.segments_polygon(
        segment_points[:, 0], segment_points[:, 1], fracture.p
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
        # internal). Point is not added, but tags are updated with the fracture
        # index.
        ind_loc = ind_point_at_node.nonzero()[0][0]  # type: ignore
        tags[ind_loc] = np.append(tags[ind_loc], fracture.index)
    else:
        # New (internal) point. Store point and tag
        segment_points = np.hstack((segment_points, isec_pt))
        tags.append(np.array(fracture.index))
    return segment_points, tags


def _intersection_grid(point: np.ndarray, gb: pp.GridBucket) -> pp.PointGrid:
    """Make a point grid and add to gb.

    Parameters:
        point (array, 3 x 1): coordinates.
        gb: The grid bucket.

    Returns:
        PointGrid
    """
    g = pp.PointGrid(point)
    g.name.append("Well-fracture intersection grid")
    g.compute_geometry()
    gb.add_nodes(g)
    return g


def _add_fracture_2_intersection_edge(
    g_l: pp.Grid, frac_num: int, gb: pp.GridBucket
) -> None:
    """
    Does not check that the well lies _inside_ a fracture cell and not on the
    face between two cells.
    """
    for g, _ in gb:
        if g.frac_num == frac_num:
            g_h = g
            break  # EK, is there a preferred method?
    cell_l = np.array([0], dtype=int)
    cell_h = g_h.closest_cell(g_l.cell_centers)
    cell_cell_map = sps.coo_matrix(
        (np.ones(1, dtype=bool), (cell_l, cell_h)), shape=(g_l.num_cells, g_h.num_cells)
    )
    _add_edge(g_l, g_h, gb, cell_cell_map)


def _add_well_2_intersection_edge(
    g_l: pp.Grid, g_h: pp.Grid, gb: pp.GridBucket
) -> None:
    cell_l = np.array([0], dtype=int)
    vec = g_h.face_centers - g_l.cell_centers
    face_h = np.array([np.argmin(np.sum(np.power(vec, 2), axis=0))], dtype=int)
    face_cell_map = sps.coo_matrix(
        (np.ones(1, dtype=bool), (cell_l, face_h)), shape=(g_l.num_cells, g_h.num_faces)
    )
    _add_edge(g_l, g_h, gb, face_cell_map)


def _add_edge(
    g_l: pp.Grid, g_h: pp.Grid, gb: pp.GridBucket, primary_secondary_map: np.ndarray
) -> None:
    """Utility method to add an edge to the gb.

    Both grids should already be present in the bucket.
    Parameters:
        g_l: is the intersection point grid.
        g_h. represents fracture or well.
        gb: GridBucket to which the edge will be added.
        primary_secondary_map (array): Map between cells_l and either faces_h
            (codim=1) or cells_h (codim=2).
    """
    codim = g_h.dim - g_l.dim
    edge = (g_h, g_l)
    gb.add_edge(edge, primary_secondary_map)
    d_e = gb.edge_props(edge)
    side_g = {pp.grids.mortar_grid.MortarSides.LEFT_SIDE: g_l.copy()}
    mg = pp.MortarGrid(0, side_g, primary_secondary_map, codim=codim)
    mg.compute_geometry()
    d_e["mortar_grid"] = mg
