""" Module with various functions to constrain a geometry.

Examples are to cut objects to lie within other objects, etc.
"""
import numpy as np

import porepy as pp


def lines_by_polygon(poly_pts, pts, edges):
    """
    Compute the intersections between a polygon (also not convex) and a set of lines.
    The computation is done line by line to avoid the splitting of edges caused by other
    edges. The implementation assume that the polygon and lines are on the plane (x, y).

    Parameters:
    poly_pts (np.ndarray, 3xn or 2xn): the points that define the polygon
    pts (np.ndarray, 3xn or 2xn): the points associated to the lines
    edges (np.ndarray, 2xn): for each column the id of the points for the line

    Returns:
    int_pts (np.ndarray, 2xn): the point associated to the lines after the intersection
    int_edges (np.ndarray, 2xn): for each column the id of the points for the line after
        the intersection. If the input edges have tags, stored in rows [2:], these will
        be preserved.
    edges_kept (np.ndarray, n): Column index of the kept edges. This will have recurring
        values if an edge is cut by a non-convex domain.

    """
    import shapely.geometry as shapely_geometry
    import shapely.speedups as shapely_speedups

    try:
        shapely_speedups.enable()
    except AttributeError:
        pass
    # it stores the points after the intersection
    int_pts = np.empty((2, 0))
    # define the polygon
    poly = shapely_geometry.Polygon(poly_pts[:2, :].T)

    # Kept edges
    edges_kept = []

    # we do the computation for each edge once at time, to avoid the splitting
    # caused by other edges.
    for ei, e in enumerate(edges.T):
        # define the line
        line = shapely_geometry.LineString([pts[:2, e[0]], pts[:2, e[1]]])
        # compute the intersections between the polygon and the current line
        int_lines = poly.intersection(line)
        # only line or multilines are considered, no points
        if (
            isinstance(int_lines, shapely_geometry.LineString)
            and len(int_lines.coords) > 0
        ):
            # consider the case of single intersection by avoiding to consider
            # lines on the boundary of the polygon
            if not int_lines.touches(poly) and int_lines.length > 0:
                int_pts = np.c_[int_pts, np.array(int_lines.xy)]
                edges_kept.append(ei)
        elif type(int_lines) is shapely_geometry.MultiLineString:
            # consider the case of multiple intersections by avoiding to consider
            # lines on the boundary of the polygon
            for int_line in int_lines:
                if not int_line.touches(poly) and int_line.length > 0:
                    int_pts = np.c_[int_pts, np.array(int_line.xy)]
                    edges_kept.append(ei)

    # define the list of edges
    int_edges = np.arange(int_pts.shape[1]).reshape((2, -1), order="F")

    # Also preserve tags, if any
    if len(edges_kept) > 0:
        edges_kept = np.array(edges_kept)
        edges_kept.sort()
        int_edges = np.vstack((int_edges, edges[2:, edges_kept]))
    else:
        # If no edges are kept, return an empty array with the right dimensions
        int_edges = np.empty((edges.shape[0], 0), dtype=np.int)

    return int_pts, int_edges, np.array(edges_kept, dtype=np.int)


def polygons_by_polyhedron(polygons, polyhedron, tol=1e-8):
    """Constrain a seort of polygons in 3d to lie inside a, generally non-convex, polyhedron.

    Polygons not inside the polyhedron will be removed from descriptions.
    For non-convex polyhedra, polygons can be split in several parts.

    Parameters:
        polygons (np.ndarray, or list of arrays): Each element is a 3xnum_vertex
            array, describing the vertexes of a polygon.
        polyhedron (list of np.ndarray): Each element is a 3 x num_vertex array,
            describing the vertexes of the polygons that together form the
            polygon
        tol (double, optional): Tolerance used to compare points. Defaults to 1e-8.

    Returns:
        list of np.ndarray: Of polygons lying inside the polyhedra.
        np.ndarray: For each constrained polygon, corresponding list of its original
            polygon

    """
    import networkx as nx

    if isinstance(polygons, np.ndarray):
        polygons = [polygons]

    constrained_polygons = []
    orig_poly_ind = []

    # Loop over the polygons. For each, find the intersections with all
    # polygons on the side of the polyhedra.
    for pi, poly in enumerate(polygons):
        # Add this polygon to the list of constraining polygons. Put this first
        all_poly = [poly] + polyhedron

        # Find intersections
        coord, point_ind, _, _, seg_vert_all = pp.intersections.polygons_3d(
            all_poly, target_poly=np.arange(1)
        )

        # Find indices of the intersection points for this polygon (the first one)
        isect_poly = point_ind[0]
        # Only consider segment-vertex information for the first polygon
        seg_vert = seg_vert_all[0]

        # If there are no intersection points, we just need to test if the
        # entire polygon is inside the polyhedral
        if isect_poly.size == 0:
            # Testing with a single point should suffice, but until the code
            # for in-polyhedron testing is more mature, we do some safeguarding:
            # Test for all points in the polygon, they should all be on the
            # inside or outside.
            inside = pp.geometry_property_checks.point_in_polyhedron(polyhedron, poly)

            if inside.all():
                # Add the polygon to the constrained ones and continue
                constrained_polygons.append(poly)
                orig_poly_ind.append(pi)
                continue
            elif np.all(np.logical_not(inside)):
                # Do not add it.
                continue
            else:
                # This indicates that the inside_polyhedron test is bad
                assert False

        # At this point we know there are intersections between the polygon and
        # polyhedra. The constrained polygon can have up to three types of segments:
        # 1) Both vertexes are on the boundary. The segment is formed by the pair of
        # intersection points between two polygons.
        # 2) Both vertexes are in the interior. This is one of the original segments
        # of the polygon.
        # 3) A segment of the original polygon crosses on or more of the polyhedron
        # boundaries. This includes the case where the original polygon has a vertex
        # on the polyhedron boundary. This can produce one or several segments.

        # Case 1): Find index of intersection points
        main_ind = point_ind[0]

        # Storage for intersection segments between the main polygon and the
        # polyhedron sides.
        boundary_segments = []

        # First find segments fully on the boundary.
        # Loop over all sides of the polyhedral. Look for intersection points
        # that are both in main and the other
        for other in range(1, len(all_poly)):
            other_ip = point_ind[other]

            common = np.isin(other_ip, main_ind)
            if common.sum() < 2:
                # This is at most a point contact, no need to do anything
                continue
            # There is a real intersection between the segments. Add it.
            boundary_segments.append(other_ip[common])

        boundary_segments = np.array([i for i in boundary_segments]).T

        # For segments with at least one interior point, we need to jointly consider
        # intersection points and the original vertexes
        num_coord = coord.shape[1]
        coord_extended = np.hstack((coord, poly))

        # Convenience arrays for navigating between vertexes in the polygon
        num_vert = poly.shape[1]
        ind = np.arange(num_vert)
        next_ind = 1 + ind
        next_ind[-1] = 0
        prev_ind = np.arange(num_vert) - 1
        prev_ind[0] = num_vert - 1

        # Case 2): Find segments that are defined by two interior points
        points_inside_polyhedron = pp.geometry_property_checks.point_in_polyhedron(
            polyhedron, poly
        )
        # segment_inside[0] tells whehter the point[:, -1] - point[:, 0] is fully inside
        # the remaining elements are point[:, 0] - point[:, 1] etc.
        segments_inside = np.logical_and(
            points_inside_polyhedron, points_inside_polyhedron[next_ind]
        )
        # Temporary list of interior segments, it will be adjusted below
        interior_segments = np.vstack((ind[segments_inside], next_ind[segments_inside]))

        # From here on, we will lean heavily on information on segments that cross the
        # boundary.

        # The test for interior points does not check if the segment crosses the
        # domain boundary due to a convex domain; these must be removed.
        # What we really want is multiple small segments, excluding those that are on
        # the outside of the domain. These are identified below, under case 3.

        # First, count the number of times a segment of the polygon is associated with
        # an intersection point
        count_boundary_segment = np.zeros(num_vert, dtype=np.int)
        for isect in seg_vert:
            # Only consider segment intersections, not interior (len==0), and vertexes
            if len(isect) > 0 and isect[1]:
                count_boundary_segment[isect[0]] += 1

        # Find presumed interior segments that crosses the boundary
        segment_crosses_boundary = np.where(
            np.logical_and(count_boundary_segment > 0, segments_inside)
        )[0]
        # Sanity check: If both points are interior, there must be an even number of
        # segment crossings
        assert np.all(count_boundary_segment[segment_crosses_boundary] % 2 == 0)
        # The index of the segments are associated with the first row of the interior_segments
        # Find the columns to keep by using invert argument to isin
        keep_ind = np.isin(interior_segments[0], segment_crosses_boundary, invert=True)
        # Delete false interior segments.
        interior_segments = interior_segments[:, keep_ind]

        # Adjust index so that it refers to the extended coordinate array
        interior_segments += num_coord

        # Case 3: Where a segment of the original polygon crosses (including start and
        # end point) the polyhedron an unknown number of times. This gives rise to
        # at least one segment, but can be multiple.

        # Storage of identified segments in the constrained polygon
        segments_interior_boundary = []

        # Check if individual vertexs are on the boundary
        vertex_on_boundary = np.zeros(num_vert, np.bool)
        for isect in seg_vert:
            if len(isect) > 0 and not isect[1]:
                vertex_on_boundary[isect[0]] = 1

        # Storage of the intersections associated with each segment of the original polygon
        isects_of_segment = np.zeros(num_vert, np.object)
        for i in range(num_vert):
            isects_of_segment[i] = []

        # Identify intersections of each segment.
        # This is a bit involved, possibly because of a poor choice of data formats:
        # The actual identification of the sub-segments (next for-loop) uses the
        # identified intersection points, with an empty point list signifying that there
        # are no intersections (that is, no sub-segments from this original segment).
        # The only problem is the case where the original segment runs from a vertex
        # on the polyhedron boundary to an interior point: This segment must be processed
        # despite there being no intersections. We achieve that by adding an empty
        # list to the relevant data field, and then remove the list if a true
        # intersection is found later
        for isect_ind, isect in enumerate(seg_vert):
            if len(isect) > 0:
                if isect[1]:
                    # intersection point lies ona segment
                    if len(isects_of_segment[isect[0]]) == 0:
                        isects_of_segment[isect[0]] = [isect_ind]
                    else:
                        # Remove empty list if necessary, then add the information
                        if isinstance(isects_of_segment[isect[0]][0], list):
                            isects_of_segment[isect[0]] = [isect_ind]
                        else:
                            isects_of_segment[isect[0]].append(isect_ind)
                else:
                    # intersection point is on a segment
                    # This segment can be connected to both the previous and next point
                    if len(isects_of_segment[isect[0]]) == 0:
                        isects_of_segment[isect[0]].append([])
                    if len(isects_of_segment[prev_ind[isect[0]]]) == 0:
                        isects_of_segment[prev_ind[isect[0]]].append([])

        # For all original segments that have intersection points (or vertex) on a
        # polyhedron boundary, find all points along the segment (original endpoints
        # and intersection points. Find out which of these sub-segments are inside and
        # outside the polyhedron, remove exterior parts
        for seg_ind in range(num_vert):
            if len(isects_of_segment[seg_ind]) == 0:
                continue
            # Index and coordinate of intersection points on this segment
            loc_isect_ind = np.asarray(isects_of_segment[seg_ind], dtype=np.int).ravel()

            # Consider unique intersection points; there may be repititions in cases
            # where the polyhedron has multiple parallel sides.
            isect_coord, _, _ = pp.utils.setmembership.unique_columns_tol(
                coord[:, loc_isect_ind], tol
            )

            # Start and end of the full segment
            start = poly[:, seg_ind].reshape((-1, 1))
            end = poly[:, next_ind[seg_ind]].reshape((-1, 1))

            # Sanity check
            assert pp.geometry_property_checks.points_are_collinear(
                np.hstack((start, isect_coord, end))
            )
            # Sort the intersection points according to their distance from the start
            sorted_ind = np.argsort(np.sum((isect_coord - start) ** 2, axis=0))

            # Indices (in terms of columns in coords_extended) along the segment
            index_along_segment = np.hstack(
                (
                    num_coord + seg_ind,
                    loc_isect_ind[sorted_ind],
                    num_coord + next_ind[seg_ind],
                )
            )
            # Since the sub-segments are formed by intersection points, every second
            # will be in the interior of the polyhedron. The first one is interior if
            # the start point is in the interior or on the boundary of the polyhedron.
            if points_inside_polyhedron[seg_ind] or vertex_on_boundary[seg_ind]:
                start_pairs = 0
            else:
                start_pairs = 1
            # Define the vertex pairs of the sub-segmetns, and add the relevant ones.
            pairs = np.vstack((index_along_segment[:-1], index_along_segment[1:]))
            for pair_ind in range(start_pairs, pairs.shape[1], 2):
                segments_interior_boundary.append(pairs[:, pair_ind])

        # Clean up boundary-interior segments
        if len(segments_interior_boundary) > 0:
            segments_interior_boundary = np.array(
                [i for i in segments_interior_boundary]
            ).T
        else:
            segments_interior_boundary = np.zeros((2, 0), dtype=np.int)

        # At this stage, we have identified all segments, possibly with duplicates.
        # Next task is to arrive at a unique representation of the segments.
        # To that end, first collect the segments in a single list
        segments = np.sort(
            np.hstack(
                (boundary_segments, interior_segments, segments_interior_boundary)
            ),
            axis=0,
        )
        # Uniquify intersection coordinates, and update the segments
        unique_coords, _, ib = pp.utils.setmembership.unique_columns_tol(
            coord_extended, tol=tol
        )
        unique_segments = ib[segments]
        # Then uniquify the segments, in terms of the unique coordinates
        unique_segments, *rest = pp.utils.setmembership.unique_columns_tol(
            unique_segments
        )
        # Remove point segments.
        point_segment = unique_segments[0] == unique_segments[1]
        unique_segments = unique_segments[:, np.logical_not(point_segment)]

        # The final stage is to collect the constrained polygons.
        # If the segments are connected, which will always be the case if the
        # polyhedron is convex, the graph will have a single connected component.
        # If not, there will be multiple connected components. Find these, and
        # make a separate polygon for each.

        # Represent the segments as a graph.
        graph = nx.Graph()
        for i in range(unique_segments.shape[1]):
            graph.add_edge(unique_segments[0, i], unique_segments[1, i])

        # Loop over connected components
        for component in nx.connected_components(graph):
            # Extract subgraph of this cluster
            sg = graph.subgraph(component)
            # Make a list of edges of this subgraph
            el = []
            for e in sg.edges():
                el.append(e)
            el = np.array([e for e in el]).T

            # The vertexes of the polygon must be ordered. This is done slightly
            # differently depending on whether the polygon forms a closed circle
            # or not
            count = np.bincount(el.ravel())

            if np.any(count > 2):
                # A single component (polygon) has nodes occuring more than twice.
                # This is presumably caused by overlapping segments in the constrained
                # polygon, which can happen if the constraining polyhedron has
                # parallel sides.
                # Remove these by projecting to the 2d plane of the main polygon, and
                # then use standard function for intersection removal there.
                center = unique_coords.mean(axis=1).reshape((-1, 1))
                coords_centered = unique_coords - center
                R = pp.map_geometry.project_plane_matrix(coords_centered)
                pt = R.dot(coords_centered)[:2]
                _, el = pp.intersections.split_intersecting_segments_2d(pt, el, tol)

            if np.any(count == 1):
                # There should be exactly two loose ends, if not, this is really
                # several polygons, and who knows how we ended up there.
                assert np.sum(count == 1) == 2
                sorted_pairs, _ = pp.utils.sort_points.sort_point_pairs(
                    el, is_circular=False
                )
                inds = np.hstack((sorted_pairs[0], sorted_pairs[1, -1]))
                # TODO: check for hanging nodes here?
            else:
                sorted_pairs, _ = pp.utils.sort_points.sort_point_pairs(el)

                # Check for hanging nodes
                hang_ind = pp.geometry_property_checks.polygon_hanging_nodes(
                    unique_coords, sorted_pairs
                )
                if hang_ind.size > 0:
                    # We will need to decrease the index of the edges with hanging nodes
                    # as we delete previous edges (with hanging nodes)
                    decrease = 0
                    for edge_ind in np.sort(hang_ind):  # sort to be sure
                        ei = edge_ind - decrease  # effective index
                        # Adjust the endpoint of the this edge
                        if ei < sorted_pairs.shape[1] - 1:
                            sorted_pairs[1, ei] = sorted_pairs[1, ei + 1]
                            sorted_pairs = np.delete(sorted_pairs, ei + 1, axis=1)
                        else:
                            # special treatment at the end of the node
                            sorted_pairs[1, ei] = sorted_pairs[1, 0]
                            sorted_pairs = np.delete(sorted_pairs, 0, axis=1)

                        # Adjust the decrease index
                        decrease += 1

                inds = sorted_pairs[0]

            # And there we are
            constrained_polygons.append(unique_coords[:, inds])
            orig_poly_ind.append(pi)

    return constrained_polygons, np.array(orig_poly_ind)


def snap_points_to_segments(p_edges, edges, tol, p_to_snap=None):
    """
    Snap points in the proximity of lines to the lines.

    Note that if two vertices of two edges are close, they may effectively
    be co-located by the snapping. Thus, the modified point set may have
    duplicate coordinates.

    Parameters:
        p_edges (np.ndarray, nd x npt): Points defining endpoints of segments
        edges (np.ndarray, 2 x nedges): Connection between lines in p_edges.
            If edges.shape[0] > 2, the extra rows are ignored.
        tol (double): Tolerance for snapping, points that are closer will be
            snapped.
        p_to_snap (np.ndarray, nd x npt_snap, optional): The points to snap. If
            not provided, p_edges will be snapped, that is, the lines will be
            modified.

    Returns:
        np.ndarray (nd x n_pt_snap): A copy of p_to_snap (or p_edges) with
            modified coordinates.

    """

    if p_to_snap is None:
        p_to_snap = p_edges
        mod_edges = True
    else:
        mod_edges = False

    pn = p_to_snap.copy()

    nl = edges.shape[1]
    for ei in range(nl):

        # Find start and endpoint of this segment.
        # If we modify the edges themselves (mod_edges==True), we should use
        # the updated point coordinates. If not, we risk trouble for almost
        # coinciding vertexes.
        if mod_edges:
            p_start = pn[:, edges[0, ei]].reshape((-1, 1))
            p_end = pn[:, edges[1, ei]].reshape((-1, 1))
        else:
            p_start = p_edges[:, edges[0, ei]].reshape((-1, 1))
            p_end = p_edges[:, edges[1, ei]].reshape((-1, 1))
        d_segment, cp = pp.distances.points_segments(pn, p_start, p_end)
        hit = np.argwhere(d_segment[:, 0] < tol)
        for i in hit:
            if mod_edges and (i == edges[0, ei] or i == edges[1, ei]):
                continue
            pn[:, i] = cp[i, 0, :].reshape((-1, 1))
    return pn
