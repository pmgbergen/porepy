"""Module with functions for computing intersections between geometric objects."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

# Module level logger
logger = logging.getLogger(__name__)


def segments_2d(
    start_1: np.ndarray,
    end_1: np.ndarray,
    start_2: np.ndarray,
    end_2: np.ndarray,
    tol: float = 1e-8,
) -> Optional[np.ndarray]:
    r"""Check if two line segments, defined by their start- and endpoints, intersect.

    The lines are assumed to be in 2D.

    If the lines are (almost) parallel, i.e. if

    .. math::
        \begin{vmatrix}
            start_1 & end_1 \\
            start_2 & end_2
        \end{vmatrix}\leq tol\times|start_1-end_1|\times|start_2-end_2|,

    a line segment is returned instead of an intersection point.

    Todo:
        This function can be replaced by a call to :meth:`segments_3d`.

    Example:
        >>> segments_2d([0, 0], [1, 1], [0, 1], [1, 0])
        array([[ 0.5],
               [ 0.5]])

        >>> segments_2d([0, 0], [1, 1], [0, 0], [2, 2])
        array([[0., 1.],
               [0., 1.]])

        >>> segments_2d([0, 0], [1, 0], [0, 1], [1, 1]) is None
        True

    Parameters:
        start_1: ``shape=(2,)``

            Coordinates of start point for first line.
        end_1: ``shape=(2,)``

            Coordinates of end point for first line.
        start_2: ``shape=(2,)``

            Coordinates of start point for first line.
        end_2: ``shape=(2,)``

            Coordinates of end point for first line.
        tol: ``default=1e-8``

            Tolerance for detecting parallel lines.

    Raises:
        ValueError: If the start and endpoints of a line are the same.

    Returns:
        Coordinates of intersection point, or the endpoints of the intersection segments
        if relevant. In the case of a segment, the first point (column) will be closest
        to ``start_1``. Shape is ``(2, np)``, where ``np`` is ``1`` for a point
        intersection, or ``2`` for a segment intersection. If the lines do not
        intersect, ``None`` is returned.

    """
    start_1 = np.asarray(start_1).astype(float)
    end_1 = np.asarray(end_1).astype(float)
    start_2 = np.asarray(start_2).astype(float)
    end_2 = np.asarray(end_2).astype(float)

    # Vectors along first and second line
    d_1 = end_1 - start_1
    d_2 = end_2 - start_2

    length_1 = np.sqrt(np.sum(d_1 * d_1))
    length_2 = np.sqrt(np.sum(d_2 * d_2))

    # Vector between the start points
    d_s = start_2 - start_1

    # An intersection point is characterized by
    #   start_1 + d_1 * t_1 = start_2 + d_2 * t_2
    #
    # which on component form becomes
    #
    #   d_1[0] * t_1 - d_2[0] * t_2 = d_s[0]
    #   d_1[1] * t_1 - d_2[1] * t_2 = d_s[1]
    #
    # First check for solvability of the system (e.g. parallel lines) by the determinant
    # of the matrix.

    discr = d_1[0] * (-d_2[1]) - d_1[1] * (-d_2[0])

    # Check if lines are parallel.
    # The tolerance should be relative to the length of d_1 and d_2
    if np.abs(discr) < tol * length_1 * length_2:
        # The lines are parallel, and will only cross if they are also colinear
        logger.debug("The segments are parallel")
        # Cross product between line 1 and line between start points on line
        start_cross_line = d_s[0] * d_1[1] - d_s[1] * d_1[0]
        if np.abs(start_cross_line) < tol * max(length_1, length_2):
            logger.debug("Lines are colinear")
            # The lines are co-linear

            # Write l1 on the form start_1 + t * d_1, find the parameter value needed
            # for equality with start_2 and end_2
            if np.abs(d_1[0]) > tol * length_1:
                t_start_2 = (start_2[0] - start_1[0]) / d_1[0]
                t_end_2 = (end_2[0] - start_1[0]) / d_1[0]
            elif np.abs(d_1[1]) > tol * length_2:
                t_start_2 = (start_2[1] - start_1[1]) / d_1[1]
                t_end_2 = (end_2[1] - start_1[1]) / d_1[1]
            else:
                # d_1 is zero
                logger.error("Found what must be a point-edge")
                raise ValueError("Start and endpoint of line should be different")
            if t_start_2 < 0 and t_end_2 < 0:
                logger.debug("Lines are not overlapping")
                return None
            elif t_start_2 > 1 and t_end_2 > 1:
                logger.debug("Lines are not overlapping")
                return None
            # We have an overlap, find its parameter values
            t_min = max(min(t_start_2, t_end_2), 0)
            t_max = min(max(t_start_2, t_end_2), 1)

            if t_max - t_min < tol:
                # It seems this can only happen if they are also equal to 0 or 1, that
                # is, the lines share a single point
                p_1 = start_1 + d_1 * t_min
                logger.debug("Colinear lines share a single point")
                return p_1.reshape((-1, 1))

            logger.debug("Colinear lines intersect along segment")
            p_1 = start_1 + d_1 * t_min
            p_2 = start_1 + d_1 * t_max
            return np.array([[p_1[0], p_2[0]], [p_1[1], p_2[1]]])

        else:
            logger.debug("Lines are not colinear")
            # Lines are parallel, but not colinear
            return None
    else:
        # Solve linear system using Cramer's rule
        t_1 = (d_s[0] * (-d_2[1]) - d_s[1] * (-d_2[0])) / discr
        t_2 = (d_1[0] * d_s[1] - d_1[1] * d_s[0]) / discr

        isect_1 = start_1 + t_1 * d_1
        isect_2 = start_2 + t_2 * d_2
        # Safeguarding
        assert np.allclose(isect_1, isect_2, tol)

        # The intersection lies on both segments if both t_1 and t_2 are on the unit
        # interval.
        # Use tol to allow some approximations
        if t_1 >= -tol and t_1 <= (1 + tol) and t_2 >= -tol and t_2 <= (1 + tol):
            logger.debug("Segment intersection found in one point")
            return np.array([[isect_1[0]], [isect_1[1]]])

        return None


def segments_3d(
    start_1: np.ndarray,
    end_1: np.ndarray,
    start_2: np.ndarray,
    end_2: np.ndarray,
    tol: float = 1e-8,
) -> Optional[np.ndarray]:
    """Find intersection points (or segments) of two 3d lines.

    Parameters:
        start_1: ``shape=(3,)``

            Coordinates of start point for first line.
        end_1: ``shape=(3,)``

            Coordinates of end point for first line.
        start_2: ``shape=(3,)``

            Coordinates of start point for first line.
        end_2: ``shape=(3,)``

            Coordinates of end point for first line.
        tol: ``default=1e-8``

            Tolerance for detecting parallel lines.

    Returns:
        Coordinates of intersection points. Shape is ``(3, np)``, where ``np`` is ``1``
        for a point intersection, or ``2`` for a segment intersection. If the lines do
        not intersect, ``None`` is returned.

    """

    # Shorthand for component of start and end points, as well as vectors
    # along lines.
    xs_1 = start_1[0]
    ys_1 = start_1[1]
    zs_1 = start_1[2]

    xe_1 = end_1[0]
    ye_1 = end_1[1]
    ze_1 = end_1[2]

    dx_1 = xe_1 - xs_1
    dy_1 = ye_1 - ys_1
    dz_1 = ze_1 - zs_1

    xs_2 = start_2[0]
    ys_2 = start_2[1]
    zs_2 = start_2[2]

    xe_2 = end_2[0]
    ye_2 = end_2[1]
    ze_2 = end_2[2]

    dx_2 = xe_2 - xs_2
    dy_2 = ye_2 - ys_2
    dz_2 = ze_2 - zs_2

    # The lines are parallel in the x-y plane, but we don't know about the
    # z-direction. Check this.
    deltas_1 = np.array([dx_1, dy_1, dz_1])
    deltas_2 = np.array([dx_2, dy_2, dz_2])

    # Find non-zero elements
    mask_1 = np.abs(deltas_1) > tol
    mask_2 = np.abs(deltas_2) > tol

    # Check for two dimensions that are not parallel with at least one line
    mask_sum = mask_1 + mask_2

    if mask_sum.sum() > 1:
        if mask_sum[0] and mask_sum[1]:
            in_discr = np.array([0, 1])
            not_in_discr = 2
        elif mask_sum[0] and mask_sum[2]:
            in_discr = np.array([0, 2])
            not_in_discr = 1
        else:
            in_discr = np.array([1, 2])
            not_in_discr = 0
    else:
        # We're going to have a zero discriminant anyhow, just pick some dimensions.
        in_discr = np.arange(2)
        not_in_discr = 2

    discr = (
        deltas_1[in_discr[0]] * deltas_2[in_discr[1]]
        - deltas_1[in_discr[1]] * deltas_2[in_discr[0]]
    )

    # An intersection will be a solution of the linear system
    #   xs_1 + dx_1 * t_1 = xs_2 + dx_2 * t_2 (1)
    #   ys_1 + dy_1 * t_1 = ys_2 + dy_2 * t_2 (2)
    #
    # In addition, the solution should satisfy
    #   zs_1 + dz_1 * t_1 = zs_2 + dz_2 * t_2 (3)
    #
    # The intersection is on the line segments if 0 <= (t_1, t_2) <= 1

    # Either the lines are parallel in two directions
    if np.abs(discr) < tol:
        # If the lines are (almost) parallel, there is no single intersection, but it
        # may be a segment.

        # First check if the third dimension is also parallel, if not, no intersection.

        # A first, simple test
        if np.any(mask_1 != mask_2):
            return None

        t = deltas_1[mask_1] / deltas_2[mask_2]

        # Second, test for alignment in all directions
        if t.size == 2 and abs(t[0] - t[1]) > tol:
            return None
        elif t.size == 3 and (abs(t[0] - t[1]) > tol or abs(t[0] - t[2]) > tol):
            return None

        # If we have made it this far, the lines are indeed parallel. Next, check that
        # they lay along the same line.
        diff_start = start_2 - start_1

        dstart_x_delta_x = diff_start[1] * deltas_1[2] - diff_start[2] * deltas_1[1]
        if np.abs(dstart_x_delta_x) > tol:
            return None
        dstart_x_delta_y = diff_start[2] * deltas_1[0] - diff_start[0] * deltas_1[2]
        if np.abs(dstart_x_delta_y) > tol:
            return None
        dstart_x_delta_z = diff_start[0] * deltas_1[1] - diff_start[1] * deltas_1[0]
        if np.abs(dstart_x_delta_z) > tol:
            return None

        # For dimensions with an incline, the vector between segment start points should
        # be parallel to the segments. Since the masks are equal, we can use any of
        # them. For dimensions with no incline, the start coordinates should be the
        # same.
        if not np.allclose(start_1[~mask_1], start_2[~mask_1], tol):
            return None

        # We have overlapping lines! finally check if segments are overlapping.

        # Since everything is parallel, it suffices to work with a single coordinate
        s_1 = start_1[mask_1][0]
        e_1 = end_1[mask_1][0]
        s_2 = start_2[mask_1][0]
        e_2 = end_2[mask_1][0]

        max_1 = max(s_1, e_1)
        min_1 = min(s_1, e_1)
        max_2 = max(s_2, e_2)
        min_2 = min(s_2, e_2)

        # Rule out case with non-overlapping segments
        if max_1 < min_2:
            return None
        elif max_2 < min_1:
            return None

        # The lines are overlapping, we need to find their common line
        lines = np.array([s_1, e_1, s_2, e_2])
        sort_ind = np.argsort(lines)

        # The overlap will be between the middle two points in the sorted list
        target = sort_ind[1:3]

        # Array of the full coordinates - same order as lines
        lines_full = np.vstack((start_1, end_1, start_2, end_2)).transpose()
        # Our segment consists of the second and third column. We're done!
        return lines_full[:, target]

    # or we are looking for a point intersection
    else:
        # Solve 2x2 system by Cramer's rule

        discr = deltas_1[in_discr[0]] * (-deltas_2[in_discr[1]]) - deltas_1[
            in_discr[1]
        ] * (-deltas_2[in_discr[0]])
        t_1 = (
            (start_2[in_discr[0]] - start_1[in_discr[0]]) * (-deltas_2[in_discr[1]])
            - (start_2[in_discr[1]] - start_1[in_discr[1]]) * (-deltas_2[in_discr[0]])
        ) / discr

        t_2 = (
            deltas_1[in_discr[0]] * (start_2[in_discr[1]] - start_1[in_discr[1]])
            - deltas_1[in_discr[1]] * (start_2[in_discr[0]] - start_1[in_discr[0]])
        ) / discr

        # Check that we are on the line segment
        if t_1 < 0 or t_1 > 1 or t_2 < 0 or t_2 > 1:
            return None

        # Compute the z-coordinates of the intersection points
        z_1_isect = start_1[not_in_discr] + t_1 * deltas_1[not_in_discr]
        z_2_isect = start_2[not_in_discr] + t_2 * deltas_2[not_in_discr]

        if np.abs(z_1_isect - z_2_isect) < tol:
            vec = np.zeros(3)
            vec[in_discr] = start_1[in_discr] + t_1 * deltas_1[in_discr]
            vec[not_in_discr] = z_1_isect
            return vec.reshape((-1, 1))
        else:
            return None


def polygons_3d(
    polys: list[np.ndarray],
    target_poly: Optional[Union[int, np.ndarray]] = None,
    tol: float = 1e-8,
    include_point_contact: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """Compute the intersection between polygons embedded in 3d.

    In addition to intersection points, the function also decides:

        1) Whether intersection points lie in the interior, on a segment or a vertex.
           If segment or vertex, the index of the segment or vertex is returned.
        2) Whether a pair of intersection points lie on the same boundary segment of a
           polygon, that is, if the polygon has a T or L-type intersection with another
           polygon.

    Assumptions:

        - All polygons are convex. Non-convex polygons will simply be treated in a wrong
          way. To circumvent this, split the non-convex polygon into convex parts.
        - No polygon contains three points on a line, that is, an angle of pi. This can
          be included, possibly by temporarily stripping the hanging node from the
          polygon definition.
        - If two polygons meet in a vertex, this is not considered an intersection.
        - If two polygons lie in the same plane, intersection types (vertex, segment,
          interior) are not classified. This will be clear from the returned values.
          Inclusion of this should be possible, but it has not been a priority.
        - Contact between polygons in a single point may not be accurately calculated.

    Parameters:
        polys: ``shape=(3, np)``

            Each list item represents a polygon, specified by its
            vertices as a numpy array. There should be at least three vertices in the
            polygon.
        target_poly: ``default=None``

            Index in poly of the polygons that should be targeted for
            intersection findings. These will be compared with the whole set in poly.

            If not provided, all polygons are compared with each other.
        tol: ``default=1e-8``

            Geometric tolerance for the computations.
        include_point_contact: ``default=True``

            If True, point contacts will be considered an
            intersection. This is an experimental feature, use with care.

    Returns:
        Returns a tuple consisting of:

        :obj:`~numpy.ndarray`: ``shape=(3, np)``

            Intersection coordinates.
        :obj:`~numpy.ndarray`:
            For each of the polygons, give the index of the intersection points,
            referring to the columns of the intersection coordinates.
        :obj:`~numpy.ndarray`:
            For each polygon, a list telling whether each of the intersections is on the
            boundary of the polygon or not. For polygon ``i``, the first element in this
            list tells whether the point formed by point-indices ``0`` and ``1`` in the
            previous return argument is on the boundary.
        :obj:`list`:
            Each list element is a 2-tuple with the indices of intersecting polygons.
        :obj:`~numpy.ndarray`:
            For each polygon, for all intersection points (same order as the second
            return value), a 2-tuple, where the first value gives an index, the second
            is a Boolean, ``True`` if the intersection is on a segment, ``False`` if
            vertex. The index identifies the vertex, or the first vertex of the segment.
            If the intersection is in the interior of a polygon, the tuple is replaced
            by an empty list.
        :obj:`~numpy.ndarray`:
            For each polygon, for all intersection points, ``True`` if this intersection
            is formed by a single point.

    """

    if target_poly is None:
        target_poly = np.arange(len(polys))
    elif isinstance(target_poly, int):
        target_poly = np.array(target_poly)

    # Obtain bounding boxes for the polygons
    x_min, x_max, y_min, y_max, z_min, z_max = _axis_aligned_bounding_box_3d(polys)

    # If a polygon is perfectly aligned with a coordinate axis, and another polygon
    # terminates in the first one, rounding errors in the coordinates may lead to the
    # intersection not being picked up. To circumvent the issue, detect such situations
    # and give ourselves a bit wiggle room.
    # It seems that this will not give problems in other cases.
    for cmin, cmax in [(x_min, x_max), (y_min, y_max), (z_min, z_max)]:
        hit = cmax - cmin < tol
        cmin[hit] -= 0.5 * tol
        cmax[hit] += 0.5 * tol

    # Identify overlapping bounding boxes: First, use a fast method to find overlapping
    # rectangles in the xy-plane.
    pairs_xy = _identify_overlapping_rectangles(x_min, x_max, y_min, y_max)
    # Next, find overlapping intervals in the z-directien
    pairs_z = _identify_overlapping_intervals(z_min, z_max)

    # Finally, do the intersection
    pairs = _intersect_pairs(pairs_xy, pairs_z)

    # Various utility functions
    def center(p):
        # Compute the mean coordinate of a set of points
        return p.mean(axis=1).reshape((-1, 1))

    def normalize(v):
        # Normalize a vector
        nrm = np.sqrt(np.sum(v**2, axis=0))
        return v / nrm

    def mod_sign(v, tol=1e-8):
        # Modified signum function: The value is 0 if it is very close to zero.
        if isinstance(v, np.ndarray):
            sgn = np.sign(v)
            sgn[np.abs(v) < tol] = 0
            return sgn
        else:
            if abs(v) < tol:
                return 0
            elif v < 0:
                return -1
            else:
                return 1

    def intersection(start, end, normal, center):
        # Find a point p on the segment between start and end, so that the vector p -
        # center is perpendicular to normal

        # Vector along the segment
        dx = end - start
        dot_prod = np.sum(normal.ravel() * dx)
        assert np.abs(dot_prod) > 1e-6
        t = -np.sum((start - center.ravel()) * normal.ravel()) / dot_prod

        assert t >= 0 and t <= 1
        return start + t * dx

    def vector_pointset_point(a, b, tol=1e-8):
        # Create a set of non-zero vectors from a point in the plane spanned by a, to
        # all points in b.
        # Loop over all points in a, search for a point that is sufficiently far away
        # from b. Mainly this involves finding a point in a which is not in b
        dist = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            dist[i] = np.min(
                np.sqrt(np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0))
            )

        ind = np.argmax(dist)
        assert dist[ind] > tol
        return b - a[:, ind].reshape((-1, 1))

    num_polys = len(polys)

    # Storage array for storing the index of the intersection points for each polygon
    isect_pt = np.empty(num_polys, dtype=object)
    # Storage for whether an intersection is on the boundary of a polygon
    is_bound_isect = np.empty_like(isect_pt)
    # Storage for which segment or vertex of a polygon is intersected
    segment_vertex_intersection = np.empty_like(isect_pt)
    is_point_contact = np.empty(num_polys, dtype=object)

    # Initialization
    for i in range(isect_pt.size):
        isect_pt[i] = []
        is_bound_isect[i] = []
        segment_vertex_intersection[i] = []
        is_point_contact[i] = []

    # Array for storing the newly found points
    new_pt = []
    new_pt_ind = 0

    # Index of the main fractures, to which the other ones will be compared. Filter out
    # all that are not among the targets.
    start_inds = np.intersect1d(target_poly, pairs)

    # Store index of pairs of intersecting polygons
    polygon_pairs = []

    # Pre-compute polygon normals to save computational time
    polygon_normals = [
        pp.map_geometry.compute_normal(poly, tol=tol).reshape((-1, 1)) for poly in polys
    ]

    # Loop over all fracture pairs (taking more than one simultaneously if an index
    # occurs several times in pairs[0]), and look for intersections
    for di, line_ind in enumerate(start_inds):
        # The algorithm first does a coarse filtering, to check if the candidate pairs
        # both crosses each others plane. For those pairs that passes this test, we next
        # compute the intersection points, and check if they are contained within the
        # fractures.

        # The main fracture, from the first row in pairs
        main = line_ind

        # Find the other fracture of all pairs starting with the main one
        hit = np.where(pairs[0] == main)
        other = pairs[1, hit][0]

        # Center point and normal vector of the main fracture
        main_center = center(polys[main])
        main_normal = polygon_normals[main]

        # Create an expanded version of the main points, so that the start and end
        # points are the same. Thus the segments can be formed by merging
        # main_p_expanded[:-1] with main_p_expanded[1:]
        num_main = polys[main].shape[1]
        ind_main_cyclic = np.arange(num_main + 1) % num_main
        main_p_expanded = polys[main][:, ind_main_cyclic]

        # Declare types for the seg_vert information. The data structure is somewhat
        # awkward, but it is what it is.
        seg_vert_main_0: tuple[Any, Union[str, bool]]
        seg_vert_main_1: tuple[Any, Union[str, bool]]
        seg_vert_other_0: tuple[Any, Union[str, bool]]
        seg_vert_other_1: tuple[Any, Union[str, bool]]

        # Loop over the other polygon in the pairs, look for intersections
        for o in other:
            # Expanded version of the other polygon
            num_other = polys[o].shape[1]
            ind_other_cyclic = np.arange(num_other + 1) % num_other
            other_p_expanded = polys[o][:, ind_other_cyclic]

            # Normal vector and cetner of the other polygon
            other_normal = polygon_normals[o]
            other_center = center(polys[o])

            # Point a vector from the main center to the vertices of the other polygon.
            # Then take the dot product with the normal vector of the main fracture. If
            # all dot products have the same sign, the other fracture does not cross the
            # plane of the main polygon. Note that we use mod_sign to safeguard the
            # computation - if the vertices are close, we will take a closer look at the
            # combination.
            vec_from_main = normalize(
                vector_pointset_point(polys[main], other_p_expanded)
            )
            dot_prod_from_main = mod_sign(np.sum(main_normal * vec_from_main, axis=0))

            # Similar procedure: Vector from ohter center to the main polygon,
            # then dot product.
            vec_from_other = normalize(vector_pointset_point(polys[o], main_p_expanded))
            dot_prod_from_other = mod_sign(
                np.sum(other_normal * vec_from_other, axis=0)
            )

            # If one of the polygons lie completely on one side of the other, there can
            # be no intersection.
            if (
                np.all(dot_prod_from_main > 0)
                or np.all(dot_prod_from_main < 0)
                or np.all(dot_prod_from_other > 0)
                or np.all(dot_prod_from_other < 0)
            ):
                continue

            # At this stage, we are fairly sure both polygons cross the plane of the
            # other polygon.
            # Identify the segments where the polygon crosses the plane
            sign_change_main = np.where(np.abs(np.diff(dot_prod_from_main)) > 0)[0]
            sign_change_other = np.where(np.abs(np.diff(dot_prod_from_other)) > 0)[0]

            # The default option is that the intersection is not on the boundary of main
            # or other, that is, the two intersection points are identical to two
            # vertices of the polygon.
            isect_on_boundary_main = False
            isect_on_boundary_other = False
            # We know that the polygons at least are very close to intersecting each-
            # others planes. There are four options, differing in whether the vertices
            # are in the plane of the other polygon or not:
            #   1) The polygon has no vertex in the other plane. Intersection is found
            #      by computing intersection between polygon segments and the other
            #      plane.
            #   2) The polygon has one vertex in the other plane. This is one
            #      intersection point. The other one should be on a segment, that is,
            #      the polygon should have points on both sides of the plane.
            #   3) The polygon has two vertices in the other plane. These will be the
            #      intersection points. The remaining vertices should be on the same
            #      side of the plane.
            #   4) All vertices lie in the plane. The intersection points will be found
            #      by what is essentially a 2d algorithm. Note that the current
            #      implementation if this case is a bit rudimentary.
            #
            # NOTE: This part of the code only considers intersection between polygon
            # and plane. The analysis whether the intersection points are within
            # each polygon is done below.
            #
            # We first compute the intersection of the other polygon with the plane of
            # the main one. The reverse operation is found below.
            if np.all(dot_prod_from_main != 0):
                # In the case where one polygon does not have a vertex in the plane of
                # the other polygon, there should be exactly two segments crossing the
                # plane.
                assert sign_change_main.size == 2
                # Compute the intersection points between the segments of the other
                # polygon and the plane of the main polygon.
                other_intersects_main_0 = intersection(
                    other_p_expanded[:, sign_change_main[0]],
                    other_p_expanded[:, sign_change_main[0] + 1],
                    main_normal,
                    main_center,
                )
                other_intersects_main_1 = intersection(
                    other_p_expanded[:, sign_change_main[1]],
                    other_p_expanded[:, sign_change_main[1] + 1],
                    main_normal,
                    main_center,
                )
                # First indices, next is whether this refers to segment. False means
                # vertex.
                seg_vert_other_0 = (sign_change_main[0], True)
                seg_vert_other_1 = (sign_change_main[1], True)

            elif np.sum(dot_prod_from_main[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include only
                # one of them when counting the number of points in the plane of the
                # other fracture.
                hit = np.where(dot_prod_from_main[:-1] == 0)[0]
                other_intersects_main_0 = other_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_main)) > 1)[0]
                if sign_change_full.size == 0:
                    # This corresponds to a point contact between one polygon and the
                    # other (at least other plane, perhaps also other polygon)
                    if not include_point_contact:
                        continue

                    # Point of the intersection - known to be on o.
                    tmp_p = polys[o][:, hit].reshape((-1, 1))

                    # Check whether the point is inside, or on the boundary, of outside.
                    # in_or_on is 0 for outside, 1 for on boundary, 2 for internal. If
                    # the contact is on an index of other, vert_ind_on_other gives the
                    # index of this vertex, if not, it is False.
                    in_or_on, vert_ind_on_main = _point_in_or_on_polygon(
                        tmp_p, polys[main], tol=tol
                    )

                    if in_or_on > 0:
                        # The intersection is between the polygons proper.
                        # Store point, assign it to both polygons.
                        new_pt.append(tmp_p)
                        isect_pt[main].append(new_pt_ind + np.arange(1))
                        isect_pt[o].append(new_pt_ind + np.arange(1))
                        polygon_pairs.append((main, o))
                        new_pt_ind += 1

                        # This is certainly a point on the boundary on main.
                        is_bound_isect[o].append(True)

                        # Store index of the point contact vertex for main.
                        is_point_contact[o].append(True)
                        # Store vertex information for other.
                        is_point_contact[main].append(True)
                        segment_vertex_intersection[o].append([hit[0], False])

                        # Store boundary information on other.
                        if in_or_on == 1:
                            assert isinstance(vert_ind_on_main, tuple)
                            assert len(vert_ind_on_main) == 2
                            # EK: mypy complains about a tuple index out of range below
                            # despite the check that the length is 2. Ignore it.
                            if vert_ind_on_main[1] is None:  # type: ignore
                                # This is a segment, but not a vertex intersection
                                segment_vertex_intersection[main].append(
                                    [vert_ind_on_main[0], True]
                                )
                            else:
                                # Intersection is on vertex of other as well
                                segment_vertex_intersection[main].append(
                                    [vert_ind_on_main[1], False]  # type: ignore
                                )
                            is_bound_isect[main].append(True)
                        else:
                            segment_vertex_intersection[main].append([])
                            is_bound_isect[main].append(False)

                    # if in_or_or is 0, the intersection is external, and we do nothing.
                    # There is no need to do further processing of the combination of
                    # main and o.
                    continue

                other_intersects_main_1 = intersection(
                    other_p_expanded[:, sign_change_full[0]],
                    other_p_expanded[:, sign_change_full[0] + 1],
                    main_normal,
                    main_center,
                )
                seg_vert_other_0 = (hit[0], False)
                seg_vert_other_1 = (sign_change_full[0], True)

            elif np.all(dot_prod_from_main[:-1] == 0):
                # The two polygons lie in the same plane. The intersection points will
                # be found on the segments of the polygons
                isect = np.zeros((3, 0))
                # Loop over both set of polygon segments, look for intersections
                for sm in range(polys[main].shape[1]):
                    # Store the intersection points found for this segment of the main
                    # polygon. If there are more than one, we know that the intersection
                    # is on the boundary of that polygon.
                    tmp_isect = np.zeros((3, 0))
                    for so in range(polys[o].shape[1]):
                        loc_isect = segments_3d(
                            main_p_expanded[:, sm],
                            main_p_expanded[:, sm + 1],
                            other_p_expanded[:, so],
                            other_p_expanded[:, so + 1],
                        )
                        if loc_isect is None:
                            continue
                        else:
                            isect = np.hstack((isect, loc_isect))
                            tmp_isect = np.hstack((tmp_isect, loc_isect))

                    # Uniquify the intersection points found on this segment of main.
                    # If more than one, the intersection is on the boundary of main.
                    tmp_unique_isect, *rest = pp.utils.setmembership.uniquify_point_set(
                        tmp_isect, tol=tol
                    )
                    if tmp_unique_isect.shape[1] > 1:
                        isect_on_boundary_main = True

                isect, *rest = pp.utils.setmembership.uniquify_point_set(isect, tol=tol)

                if isect.shape[1] == 0:
                    # The polygons share a plane, but no intersections
                    continue
                elif isect.shape[1] == 1:
                    if not include_point_contact:
                        continue

                    # Register point
                    new_pt.append(isect)
                    isect_pt[main].append(new_pt_ind + np.arange(1))
                    isect_pt[o].append(new_pt_ind + np.arange(1))
                    polygon_pairs.append((main, o))
                    new_pt_ind += 1

                    # This is certainly a point on the boundary on both
                    is_bound_isect[main].append(True)
                    is_bound_isect[o].append(True)
                    # Store point index information
                    is_point_contact[o].append(True)
                    is_point_contact[main].append(True)

                    # For each of the polygons, check proximity of intersection first
                    # with vertices, next segments.
                    for tmp_ind in [main, o]:
                        dist_vert = pp.distances.point_pointset(isect, polys[tmp_ind])
                        if dist_vert.min() < tol:
                            # This is a point
                            segment_vertex_intersection[tmp_ind].append(
                                [np.argmin(dist_vert), False]
                            )
                        else:
                            # Point failed, look for closest segment.
                            start = polys[tmp_ind]
                            end = np.roll(start, -1, axis=1)
                            dist_seg, _ = pp.distances.points_segments(
                                isect, start, end
                            )
                            segment_vertex_intersection[tmp_ind].append(
                                [np.argmin(dist_seg[0]), True]
                            )
                    # Intersection information is complete, move on.
                    continue
                elif isect.shape[1] == 2:
                    other_intersects_main_0 = isect[:, 0]
                    other_intersects_main_1 = isect[:, 1]
                else:
                    raise ValueError("There should be at most two intersections")

                seg_vert_other_0 = (0, "not implemented for shared planes")
                seg_vert_other_1 = (0, "not implemented for shared planes")

            else:
                # Both of the intersection points are vertices.
                # Check that there are only two points - if this assertion fails, there
                # is a hanging node of the other polygon, which is in the plane of the
                # other polygon. Extending to cover this case should be possible, but
                # further treatment is unclear at the moment.
                assert np.sum(dot_prod_from_main[:-1] == 0) == 2
                hit = np.where(dot_prod_from_main[:-1] == 0)[0]
                other_intersects_main_0 = other_p_expanded[:, hit[0]]
                # Pick the last of the intersection points. This is valid also for
                # multiple (>2) intersection points, but we keep the assertion for now.
                other_intersects_main_1 = other_p_expanded[:, hit[1]]

                seg_vert_other_0 = (hit[0], False)
                seg_vert_other_1 = (hit[1], False)

                # The other polygon has an edge laying in the plane of the main polygon.
                # This will be registered as a boundary intersection, but only if the
                # polygons (not only plane) intersect.
                if (
                    hit[0] + 1 == hit[-1]
                    or hit[0] == 0
                    and hit[-1] == (dot_prod_from_main.size - 2)
                ):
                    isect_on_boundary_other = True

            # Next, analyze intersection between main polygon and the plane of the other
            if np.all(dot_prod_from_other != 0):
                # In the case where one polygon does not have a vertex in the plane of
                # the other polygon, there should be exactly two segments crossing the
                # plane.
                assert sign_change_other.size == 2
                # Compute the intersection points between the segments of the main
                # polygon and the plane of the other polygon.
                main_intersects_other_0 = intersection(
                    main_p_expanded[:, sign_change_other[0]],
                    main_p_expanded[:, sign_change_other[0] + 1],
                    other_normal,
                    other_center,
                )
                main_intersects_other_1 = intersection(
                    main_p_expanded[:, sign_change_other[1]],
                    main_p_expanded[:, sign_change_other[1] + 1],
                    other_normal,
                    other_center,
                )
                seg_vert_main_0 = (sign_change_other[0], True)
                seg_vert_main_1 = (sign_change_other[1], True)

            elif np.sum(dot_prod_from_other[:-1] == 0) == 1:
                # The first and last element represent the same point, thus include only
                # one of them when counting the number of points in the plane of the
                # other fracture.
                hit = np.where(dot_prod_from_other[:-1] == 0)[0]
                main_intersects_other_0 = main_p_expanded[:, hit[0]]
                sign_change_full = np.where(np.abs(np.diff(dot_prod_from_other)) > 1)[0]
                if sign_change_full.size == 0:
                    # This corresponds to a point contact between one polygon and the
                    # other (at least other plane, perhaps also other polygon)

                    if not include_point_contact:
                        continue

                    # Point of the intersection - known to be on main.
                    tmp_p = polys[main][:, hit].reshape((-1, 1))

                    # Check whether the point is inside, or on the boundary, of outside.
                    # in_or_on is 0 for outside, 1 for on boundary, 2 for internal. If
                    # the contact is on an index of other, vert_ind_on_other gives the
                    # index of this vertex, if not, it is False.
                    in_or_on, vert_ind_on_other = _point_in_or_on_polygon(
                        tmp_p, polys[o], tol=tol
                    )

                    if in_or_on > 0:
                        # The intersection is between the polygons proper.
                        # Store point, assign it to both polygons.
                        new_pt.append(tmp_p)
                        isect_pt[main].append(new_pt_ind + np.arange(1))
                        isect_pt[o].append(new_pt_ind + np.arange(1))
                        polygon_pairs.append((main, o))
                        new_pt_ind += 1

                        # This is certainly a point on the boundary on main.
                        is_bound_isect[main].append(True)

                        # Store index of the point contact vertex for main.
                        is_point_contact[main].append(True)
                        # Store vertex information for other.
                        is_point_contact[o].append(True)

                        segment_vertex_intersection[main].append([hit[0], False])

                        # Store boundary information on other.
                        if in_or_on == 1:
                            assert isinstance(vert_ind_on_other, tuple)  # for mypy
                            assert len(vert_ind_on_other) == 2  # for mypy
                            # EK: mypy complains about a tuple index out of range below
                            # despite the check that the length is 2. Ignore it.
                            if vert_ind_on_other[1] is None:  # type: ignore
                                # This is a segment, but not a vertex intersection
                                segment_vertex_intersection[o].append(
                                    [vert_ind_on_other[0], True]
                                )
                            else:
                                # Intersection is on vertex of other as well
                                segment_vertex_intersection[o].append(
                                    [vert_ind_on_other[1], False]  # type: ignore
                                )
                            is_bound_isect[o].append(True)
                        else:
                            segment_vertex_intersection[o].append([])
                            is_bound_isect[o].append(False)

                    # if in_or_or is 0, the intersection is external, and we do nothing.
                    # There is no need to do further processing of the combination of
                    # main and o.
                    continue

                main_intersects_other_1 = intersection(
                    main_p_expanded[:, sign_change_full[0]],
                    main_p_expanded[:, sign_change_full[0] + 1],
                    other_normal,
                    other_center,
                )
                seg_vert_main_0 = (hit[0], False)
                seg_vert_main_1 = (sign_change_full[0], True)

            elif np.all(dot_prod_from_other[:-1] == 0):
                isect = np.zeros((3, 0))
                for so in range(polys[o].shape[1]):
                    tmp_isect = np.zeros((3, 0))
                    for sm in range(polys[main].shape[1]):
                        loc_isect = segments_3d(
                            main_p_expanded[:, sm],
                            main_p_expanded[:, sm + 1],
                            other_p_expanded[:, so],
                            other_p_expanded[:, so + 1],
                        )
                        if loc_isect is None:
                            continue
                        else:
                            isect = np.hstack((isect, loc_isect))
                            tmp_isect = np.hstack((tmp_isect, loc_isect))

                    tmp_unique_isect, *rest = pp.utils.setmembership.uniquify_point_set(
                        tmp_isect, tol=tol
                    )

                    if tmp_unique_isect.shape[1] > 1:
                        isect_on_boundary_other = True

                isect, *rest = pp.utils.setmembership.uniquify_point_set(isect, tol=tol)

                seg_vert_main_0 = (0, "not implemented for shared planes")
                seg_vert_main_1 = (0, "not implemented for shared planes")
                if isect.shape[1] == 0:
                    # The polygons share a plane, but no intersections
                    continue
                elif isect.shape[1] == 1:
                    # Point contact. Must be on the boundary of both, but not clear
                    # whether it is on vertex of both (must be at least on one).
                    if not include_point_contact:
                        continue

                    # Register point
                    new_pt.append(isect)
                    isect_pt[main].append(new_pt_ind + np.arange(1))
                    isect_pt[o].append(new_pt_ind + np.arange(1))
                    polygon_pairs.append((main, o))
                    new_pt_ind += 1

                    # This is certainly a point on the boundary on both
                    is_bound_isect[main].append(True)
                    is_bound_isect[o].append(True)
                    # Store point index information
                    is_point_contact[o].append(True)
                    is_point_contact[main].append(True)

                    # For each of the polygons, check proximity of intersection first
                    # with vertices, next segments.
                    for tmp_ind in [main, o]:
                        dist_vert = pp.distances.point_pointset(isect, polys[tmp_ind])
                        if dist_vert.min() < tol:
                            # This is a point
                            segment_vertex_intersection[tmp_ind].append(
                                [np.argmin(dist_vert), True]
                            )
                        else:
                            # Point failed, look for closest segment.
                            start = polys[tmp_ind]
                            end = np.roll(start, -1, axis=1)
                            dist_seg, _ = pp.distances.points_segments(
                                isect, start, end
                            )
                            segment_vertex_intersection[tmp_ind].append(
                                [np.argmin(dist_seg), False]
                            )

                    # Intersection information is complete, move on.
                    continue
                elif isect.shape[1] == 2:
                    main_intersects_other_0 = isect[:, 0]
                    main_intersects_other_1 = isect[:, 1]
                else:
                    raise ValueError("There should be at most two intersections")

            else:
                # Both of the intersection points are vertices.
                # Check that there are only two points - if this assertion fails, there
                # is a hanging node of the main polygon, which is in the plane of the
                # other polygon. Extending to cover this case should be possible, but
                # further treatment is unclear at the moment.
                # Do not count the last point here, this is identical to the first one.
                assert np.sum(dot_prod_from_other[:-1] == 0) == 2
                hit = np.where(dot_prod_from_other[:-1] == 0)[0]
                main_intersects_other_0 = main_p_expanded[:, hit[0]]
                # Pick the last of the intersection points. This is valid also for
                # multiple (>2) intersection points, but we keep the assertion for now.
                main_intersects_other_1 = main_p_expanded[:, hit[-1]]

                seg_vert_main_0 = (hit[0], False)
                seg_vert_main_1 = (hit[1], False)

                # The main polygon has an edge laying in the plane of the other polygon.
                # If the two intersection points form a segment, this will be registered
                # as a boundary intersection, but only if the polygons (not only plane)
                # intersect. The two points can either be one apart in the main polygon,
                # or it can be the first and the penultimate point (in the latter case,
                # the final point, which is identical to the first one, will also be in
                # the plane, but this is disregarded by the [:-1] above)
                if (
                    hit[0] + 1 == hit[-1]
                    or hit[0] == 0
                    and hit[-1] == (dot_prod_from_other.size - 2)
                ):
                    isect_on_boundary_main = True

            ###
            # We now have the intersections between polygons and planes.
            # To finalize the computation, we need to sort out how the intersection
            # points are located relative to each other. Only if there is an overlap
            # between the intersection points of the main and the other polygon is there
            # a real intersection (contained within the polygons, not only) in their
            # planes, but outside the features themselves.

            # Vectors from the intersection points in the main fracture to the
            # intersection point in the other fracture
            main_0_other_0 = other_intersects_main_0 - main_intersects_other_0
            main_0_other_1 = other_intersects_main_1 - main_intersects_other_0
            main_1_other_0 = other_intersects_main_0 - main_intersects_other_1
            main_1_other_1 = other_intersects_main_1 - main_intersects_other_1

            # e_1 is positive if both points of the other fracture lie on the same side
            # of the first intersection point of the main one.
            # e_1 negative means the first intersection point of main with the plane of
            # the others is surrounded by the intersection points of the other polygon
            # with the main plane.
            # Use a mod_sign here to avoid issues related to rounding errors
            e_1 = mod_sign(np.sum(main_0_other_0 * main_0_other_1))
            # e_2 is positive if both points of the other fracture lie on the same side
            # of the second intersection point of the main one
            e_2 = mod_sign(np.sum(main_1_other_0 * main_1_other_1))
            # e_3 is positive if both points of the main fracture lie on the same side
            # of the first intersection point of the other one
            e_3 = mod_sign(np.sum((-main_0_other_0) * (-main_1_other_0)))
            # e_4 is positive if both points of the main fracture lie on the same side
            # of the second intersection point of the other one
            e_4 = mod_sign(np.sum((-main_0_other_1) * (-main_1_other_1)))

            # This is in essence an implementation of the flow chart in Figure 9 in Dong
            # et al, however the inequality signs are changed a bit to make the logic
            # clearer.
            if e_1 > 0 and e_2 > 0 and e_3 > 0 and e_4 > 0:
                # The intersection points for the two fractures are separated.
                # There is no intersection
                continue
            if (
                sum([e_1 == 0, e_2 == 0]) == 1
                and sum([e_1 > 0, e_2 > 0]) == 1
                and sum([e_3 == 0, e_4 == 0]) == 1
                and sum([e_3 > 0, e_4 > 0]) == 1
            ):
                # Contact in a single point
                continue

            if e_1 >= 0:
                # The first point on the main fracture is at most marginally involved in
                # the intersection (if e_1 == 0, two segments intersect)
                if e_2 >= 0:
                    # The second point on the main fracture is at most marginally
                    # involved.
                    # We know that e_3 and e_4 are non-positive (positive is covered
                    # above and a combination is not possible)

                    # The intersection points are defined by the intersection of other
                    # with the plane of main
                    isect_pt_loc = [other_intersects_main_0, other_intersects_main_1]

                    # Next, we need to classify the intersection types (segments or not)
                    # For the other polygon, we know both intersections are on the
                    # segments
                    segment_vertex_intersection[o].append(seg_vert_other_0)
                    segment_vertex_intersection[o].append(seg_vert_other_1)

                    # For the main segment, the intersection most likely hits in the
                    # interior, however, there is still the chance that the intersection
                    # is on the segment (if e_1 == 0 and / or e__2 == 0)

                    # Check if the first intersection point is on the boundary of main
                    if e_3 == 0:
                        # e_3 = main_0_other_0.dot(main_1_other_0) == 0
                        # We know all of e_i are parallel, thus orthogonality is not
                        # an option. Thus, at least of the components of e_3 is 0.

                        # main_0_other_0 is involved in e_1, check if this is zero
                        if mod_sign(np.abs(main_0_other_0).sum()) == 0:
                            # other_intersects_main_0 == main_intersects_other_0
                            # The first intersection point, seen from main, should have
                            # seg_vert info 0.
                            segment_vertex_intersection[main].append(seg_vert_main_0)
                        else:
                            # other_intersects_main_0 == main_intersects_other_1
                            # The first intersection point, seen from main, should have
                            # seg_vert info 1.
                            segment_vertex_intersection[main].append(seg_vert_main_1)
                    else:
                        if isect_on_boundary_main:
                            # The first intersection coincides with a segment of main
                            ind = seg_vert_main_0[0]
                            segment_vertex_intersection[main].append((ind, False))
                        else:
                            # The first intersection is in the interior of main
                            segment_vertex_intersection[main].append([])

                    # Next, treat the second intersection point
                    # Check if other_intersects_main_1 equalls either
                    # main_intersects_other_0 or main_intersects_other_1
                    if e_4 == 0:
                        # e_4 = main_0_other_1.dot(main_1_other_1) == 0
                        if mod_sign(np.abs(main_1_other_1).sum()) == 0:
                            # other_intersects_main_1 == main_intersects_other_0
                            segment_vertex_intersection[main].append(seg_vert_main_1)
                        else:
                            # other_intersects_main_1 == main_intersects_other_1
                            segment_vertex_intersection[main].append(seg_vert_main_0)

                    else:
                        if isect_on_boundary_main:
                            ind = seg_vert_main_1[0]
                            if not (ind == num_main - 1 and seg_vert_main_0[0] == 0):
                                ind -= 1

                            segment_vertex_intersection[main].append((ind, True))

                        else:
                            segment_vertex_intersection[main].append([])

                else:  # e_2 < 0
                    # The second point on the main fracture is surrounded by points on
                    # the other fracture. One of them will in turn be surrounded by the
                    # points on the main fracture, this is the intersecting one.
                    if e_3 <= 0:
                        # Intersection consists of second point from main, then first
                        # from other
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_0,
                        ]

                        # seg-vert information for the first point is simple for main
                        segment_vertex_intersection[main].append(seg_vert_main_1)

                        # Second point for main is more difficult
                        if mod_sign(np.abs(main_0_other_0).sum()) == 0:
                            # e_1 == 0 gives main_intersects_other_0 equals either
                            #  other_intersects_main_0 or other_intersects_main_1
                            # e_3 == 0 confirms
                            #  main_intersects_other_0 == other_intersects_main_0
                            # (otherwise e_2 would also have been zero)
                            segment_vertex_intersection[main].append(seg_vert_main_0)

                        else:
                            if isect_on_boundary_main:
                                # No intersection for the first point of main
                                ind = seg_vert_main_0[0]
                                segment_vertex_intersection[main].append((ind, False))
                            else:
                                segment_vertex_intersection[main].append([])

                        # seg-vert information for first point, seen from other
                        # We know that e_2 < 0, thus main_intersects_other_1 cannot
                        # equal other_intersects_main_0 or other_intersects_main_1
                        if e_4 == 0:
                            segment_vertex_intersection[o].append(seg_vert_other_1)
                            assert False, "this should not happen"
                        else:
                            if isect_on_boundary_other:
                                ind = seg_vert_other_1[0]
                                if not (
                                    ind == num_other - 1 and seg_vert_other_0[0] == 0
                                ):
                                    ind -= 1
                                segment_vertex_intersection[o].append((ind, True))
                            else:
                                segment_vertex_intersection[o].append([])

                        # seg-vert information for the second point is simple for other
                        segment_vertex_intersection[o].append(seg_vert_other_0)

                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_1,
                            other_intersects_main_1,
                        ]
                        # seg-vert information for the first point is simple for main
                        segment_vertex_intersection[main].append(seg_vert_main_1)

                        # For the second point, we need to check if
                        #  other_intersects_main_1 == main_intersects_other_0
                        # this will imply
                        if mod_sign(np.abs(main_0_other_1).sum()) == 0:
                            # The first point on the main fracture barely hits the other
                            # fracture
                            segment_vertex_intersection[main].append(seg_vert_main_0)
                        else:
                            # No intersection for the first point of main
                            segment_vertex_intersection[main].append([])

                        # Check if main_intersects_other_1 == other_intersects_main_0
                        if e_3 == 0:
                            assert False, "this should not happen for e_2 < 0"
                            segment_vertex_intersection[o].append(seg_vert_other_0)
                        else:
                            segment_vertex_intersection[o].append([])

                        # seg-vert information for the second point is simple for other
                        segment_vertex_intersection[o].append(seg_vert_other_1)

                    else:
                        # We may eventually end up here for overlapping fractures
                        assert False
            elif e_2 >= 0:
                # Since e_1 is known to be negative, we know that main_intersects_other
                # is one intersection point.
                if e_1 < 0:  # Equality is covered above
                    # The first point on the main fracture is surrounded by points on
                    # the other fracture. One of them will in turn be surrounded by the
                    # points on the main fracture, this is the intersecting one.
                    if e_3 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_0,
                            other_intersects_main_0,
                        ]
                        # seg-vert information for the first point is simple for main
                        segment_vertex_intersection[main].append(seg_vert_main_0)
                        if e_3 < 0:  # The second intersection point is interior to main
                            segment_vertex_intersection[main].append([])
                        else:  # On the boundary of main
                            segment_vertex_intersection[main].append(seg_vert_main_1)

                        # For other, the first intersection point is known to be
                        # interior, or else e_1 would have been 0
                        segment_vertex_intersection[o].append([])
                        segment_vertex_intersection[o].append(seg_vert_other_0)

                    elif e_4 <= 0:
                        isect_pt_loc = [
                            main_intersects_other_0,
                            other_intersects_main_1,
                        ]
                        segment_vertex_intersection[main].append(seg_vert_main_0)
                        if mod_sign(np.abs(main_1_other_1).sum()) == 0:
                            segment_vertex_intersection[main].append(seg_vert_main_1)
                        else:
                            segment_vertex_intersection[main].append([])

                        segment_vertex_intersection[o].append([])
                        segment_vertex_intersection[o].append(seg_vert_other_1)

                    else:
                        # We may eventually end up here for overlapping fractures
                        assert False
            elif e_1 < 0 and e_2 < 0:
                # The points in on the main fracture are the intersection points
                isect_pt_loc = [main_intersects_other_0, main_intersects_other_1]
                segment_vertex_intersection[main].append(seg_vert_main_0)
                segment_vertex_intersection[main].append(seg_vert_main_1)

                if isect_on_boundary_other:
                    ind_0 = seg_vert_other_0[0]
                    ind_1 = seg_vert_other_1[0]
                    if abs(ind_0 - ind_1) == 1:
                        segment_vertex_intersection[o].append((min(ind_0, ind_1), True))
                        segment_vertex_intersection[o].append((min(ind_0, ind_1), True))
                    else:
                        segment_vertex_intersection[o].append((num_other - 1, True))
                        segment_vertex_intersection[o].append((num_other - 1, True))
                else:
                    segment_vertex_intersection[o].append([])
                    segment_vertex_intersection[o].append([])
            else:
                # This should never happen
                assert False

            # Append data for this combination of polygons.
            new_pt.append(np.array(isect_pt_loc).T)
            num_new = len(isect_pt_loc)
            isect_pt[main].append(new_pt_ind + np.arange(num_new))
            isect_pt[o].append(new_pt_ind + np.arange(num_new))
            new_pt_ind += num_new
            is_bound_isect[main].append(isect_on_boundary_main)
            is_bound_isect[o].append(isect_on_boundary_other)
            polygon_pairs.append((main, o))

            is_point_contact[main] += num_new * [False]
            is_point_contact[o] += num_new * [False]

    # Cleanup and return. Puh!
    if len(new_pt) > 0:
        new_points = np.hstack([v for v in new_pt])
        for i in range(isect_pt.size):
            if len(isect_pt[i]) > 0:
                isect_pt[i] = np.hstack([v for v in isect_pt[i]])
            else:
                isect_pt[i] = np.empty(0)

    else:
        new_points = np.empty((3, 0))
        for i in range(isect_pt.size):
            isect_pt[i] = np.empty(0)
    return (
        new_points,
        isect_pt,
        is_bound_isect,
        polygon_pairs,
        segment_vertex_intersection,
        is_point_contact,
    )


def segments_polygon(
    start: np.ndarray, end: np.ndarray, poly: np.ndarray, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the internal intersection from line segments to a polygon in 3d.
    Intersections with the boundary of the polygon are not computed.

    Note:
        It is required that all points lie in a plane. A sanity check will be
        performed.

    Example:
        >>> import numpy as np
        >>> import porepy as pp
        >>> start = np.array([0.5, 0.5, -0.5])
        >>> end = np.array([0.5, 0.5, 0.5])
        >>> poly = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        >>> is_cross, pt = pp.intersections.segments_polygon(start, end, poly)
        >>> print(is_cross)
        >>> print(pt)

    Parameters:
        start: ``shape=(nd, num_segments)``

            One endpoint of segments.
        end: ``shape=(nd, num_segments)``

            Other endpoint of segments.
        poly: ``shape=(nd, num_vertices)``

            Vertices of polygon.
        tol: ``default=1e-5``

            Tolerance for the geometric computations.

    Returns:
        A tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(num_segments)``

            boolean array, identifying whether a segment has an intersection with the
            polygon (useful to filter the second return parameter).
        :obj:`~numpy.ndarray`: ``shape=(nd, num_segments)``

            float array containing the intersection points.

    """
    # Reshape if only one point is given
    if start.size < 4:
        start = start.reshape((-1, 1))
    if end.size < 4:
        end = end.reshape((-1, 1))

    num_p = start.shape[1]
    nd = start.shape[0]

    cp = np.zeros((nd, num_p))

    # First translate the points so that the first plane is located at the origin
    center = np.mean(poly, axis=1).reshape((-1, 1))

    poly = poly - center
    start = start - center
    end = end - center

    # Obtain the rotation matrix that projects p1 to the xy-plane
    rot_p = pp.map_geometry.project_plane_matrix(poly)
    irot = rot_p.transpose()
    poly_rot = rot_p.dot(poly)

    # Sanity check: The points should lie on a plane
    assert np.all(np.abs(poly_rot[2]) < tol)

    poly_xy = poly_rot[:2]

    # Rotate the point set, using the same coordinate system.
    start = rot_p.dot(start)
    end = rot_p.dot(end)

    dz = end[2] - start[2]
    non_zero_incline = np.abs(dz) > tol

    # Parametrization along line of intersection point
    t = 0 * dz

    # Intersection point for segments with non-zero incline
    t[non_zero_incline] = -start[2, non_zero_incline] / dz[non_zero_incline]

    # Segments with z=0 along the segment
    zero_along_segment = np.logical_and(
        non_zero_incline, np.logical_and(t >= 0 - tol, t <= 1 + tol).astype(bool)
    )
    x0 = start + (end - start) * t
    # Check if zero point is inside the polygon
    inside = pp.geometry_property_checks.point_in_polygon(poly_xy, x0[:2])
    crosses = np.logical_and(inside, zero_along_segment)

    # Verify that the computed points are in the interior of the associated segments
    dot_product = np.einsum("ij,ij->j", x0 - start, end - start)
    sq_length = np.einsum("ij,ij->j", end - start, end - start)

    crosses[dot_product < 0] = False
    crosses[dot_product > sq_length + tol] = False

    # Rotate back the points
    x0[2, crosses] = 0
    cp[:, crosses] = center + irot.dot(x0[:, crosses])

    return crosses, cp


def segments_polyhedron(
    start: np.ndarray, end: np.ndarray, poly: np.ndarray, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the intersection from line segments to the interior of a convex
    polyhedron. Intersections with the boundary of the polyhedron are not computed.

    Note:
        There are four possibilities for each segment:

        1. the segment is completely inside the polyhedron, meaning that its vertices
        are both inside the polyhedron;
        2. the segment has only one vertex in the polyhedron;
        3. the segment is completely outside the polyhedron;
        4. the segment has in intersection but both vertices are outside the
        polyhedron.

    Example:

        >>> import numpy as np
        >>> import porepy as pp
        >>> s = np.array([0.5, 0.5, 0.25])
        >>> e = np.array([0.5, 0.5, 0.75])
        >>> p = np.array(
        >>>     [
        >>>         [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        >>>         [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        >>>         [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        >>>         [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        >>>         [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        >>>         [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        >>>     ]
        >>> )
        >>> pts, s_in, e_in, perc = pp.intersections.segments_polyhedron(s, e, p)

    Parameters:
        start: ``shape=(nd, num_segments)``

            One endpoint of segments.
        end: ``shape=(nd, num_segments)``

            Other endpoint of segments.
        poly: ``shape(nd, num_vertices)``

            Vertices of polyhedron organised face by face.
        tol: ``default=1e-5``

            Tolerance for the geometric computations.

    Returns:
        A tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(num_segments,)``

            Intersection points with the polyhedron, start and end points are not
            included in this list.
        :obj:`~numpy.ndarray`: ``shape=(num_segments,)``

            Boolean array indicating whether the start of a segment is inside the
            polyhedron.
        :obj:`~numpy.ndarray`: ``shape=(num_segments,)``

            Boolean array indicating whether the end of a segment is inside the
            polyhedron.
        :obj:`~numpy.ndarray`: ``shape=(num_segments,)``

            Length percentage of a segment inside the polyhedron.

    """
    # For a single point make its shape consistent
    if len(start.shape) == 1:
        start = start.reshape((start.shape[0], -1))
        end = end.reshape((end.shape[0], -1))

    # Check if the vertices are inside or outside the polyhedron
    is_in_start = pp.geometry_property_checks.point_in_polyhedron(poly, start, tol)
    is_in_end = pp.geometry_property_checks.point_in_polyhedron(poly, end, tol)

    # Check how many intersections a segment has with the faces of the polyhedron
    extra_pts = np.empty(start.shape[1], dtype=object)
    extra_pts.fill(np.empty((3, 0)))
    for face in poly:
        # the face vertices need to be sorted
        sort_ind = pp.utils.sort_points.sort_point_plane(face, np.average(face, axis=1))
        # compute if the current face intersect the segments
        is_inside, pts = segments_polygon(start, end, face[:, sort_ind], tol=tol)
        for i in np.flatnonzero(is_inside):
            extra_pts[i] = np.c_[extra_pts[i], pts[:, i]]

    # Loop on the segments and compute the length in the polyhedron
    length = np.zeros(start.shape[1])
    for seg in np.arange(start.shape[1]):
        # The segment is all inside
        if is_in_start[seg] and is_in_end[seg]:
            length[seg] = np.linalg.norm(end[:, seg] - start[:, seg])
        elif is_in_start[seg] and extra_pts[seg].shape[1] > 0:
            length[seg] = np.linalg.norm(start[:, seg] - extra_pts[seg].flatten())
        elif is_in_end[seg] and extra_pts[seg].shape[1] > 0:
            length[seg] = np.linalg.norm(end[:, seg] - extra_pts[seg].flatten())
        elif extra_pts[seg].shape[1] > 1:
            length[seg] = np.linalg.norm(extra_pts[seg][:, 0] - extra_pts[seg][:, 1])

    # Compute the percentage of segment in the polyhedron
    ratio: np.ndarray = length / np.sqrt(
        np.einsum("ij,ij->j", end - start, end - start)
    )

    return extra_pts, is_in_start, is_in_end, ratio


def _point_in_or_on_polygon(
    p: np.ndarray, poly: np.ndarray, tol: float = 1e-8
) -> tuple[int, Optional[tuple[int, None] | tuple[None, int]]]:
    """Helper function to get intersection information between a point and a polygon.

    The polygon is classified as being outside, on the boundary or in the interior of
    the polygon. If on the boundary, the intersection is further classified according
    to whether it is on a segment or vertex.

    Parameters:
        p: ``shape=(nd,)``

            Point.
        poly: ``shape=(nd, num_vertices)``

            Vertices of polygon.
        tol: ``default=1e-8``

            Tolerance for the geometric computations.

    Returns:
        A tuple consisting of

        :obj:`int`:
            value ``0`` for 'outside', ``1`` for 'on boundatry', and ``2`` for
            'interior'.

        Optional[tuple[int, None] | tuple[None, int]]:
            ``None`` if the first value is ``0`` or ``2``. A ``tuple`` if the first
            value is ``1``. The first and second items are ``None`` in the case of
            an intersection on a segment and vertex, respectively. Otherwise, the
            segment or vertex index, correspondingly.

    """

    # Rotate polygon to its natural plane, also map point.
    cp = poly.mean(axis=1).reshape((-1, 1))
    rot = pp.map_geometry.project_plane_matrix(poly - cp)
    rot_poly = rot.dot(poly - cp)[:2]
    rot_p = rot.dot(p - cp)

    if np.abs(rot_p[2]) > tol:
        # Point not even in the plane of the polygon.
        return 0, None

    if not pp.geometry_property_checks.point_in_polygon(rot_poly, rot_p, default=True):
        # Point outside the polygon
        return 0, None

    dist, _ = pp.distances.points_segments(p, poly, np.roll(poly, -1, axis=1))

    if dist.min() < tol:
        # Intersection on boundary. Either vertex of segment.
        vert_dist = pp.distances.point_pointset(p, poly)
        if vert_dist.min() < tol:
            return 1, (None, int(np.argmin(vert_dist)))
        else:
            return 1, (int(np.argmin(dist)), None)
    else:
        # Point inside
        return 2, None


def triangulations(
    p_1: np.ndarray, p_2: np.ndarray, t_1: np.ndarray, t_2: np.ndarray
) -> list[tuple[int, int, float]]:
    """Compute intersection of two triangle tessellations of a surface.

    The function will identify partly overlapping triangles between ``t_1`` and ``t_2``,
    and compute their common area. If parts of domain 1 or 2 are covered by one
    tessellation only, this will simply be ignored by the function.

    Note:
        The function relies on the intersection algorithm in shapely.geometry.Polygon.

    See also:
        :meth:`surface_tessellations`

    Parameters:
        p_1: ``shape=(2, n_p1)``

            Points in first tessellation.
        p_2: ``shape=(2, n_p2)``

            Points in second tessellation.
        t_1: ``shape=(3, n_tri_1)``

            Triangles in first tessellation, referring
            to indices in p_1.
        t_2: ``shape = (3, n_tri_1)``

            Triangles in second tessellation, referring to indices in p_2.

    Returns:
        List of tuples with each representing an overlap. The tuple consists of

        :obj:`int`:
            the index of the triangle in the first tesselation.
        :obj:`int`:
            the index of the triangle in the second tesselation.
        :obj:`float`:
            the common area of the two triangles in the two tessellations.

    """

    # NOTE: Regarding the use of shapely. It may be possible to extend the functionality
    # to other cell shapes. This would require more general data structures, but should
    # not be too much of an effort.
    import shapely.geometry as shapely_geometry

    n_1 = t_1.shape[1]
    n_2 = t_2.shape[1]
    t_1 = t_1.T
    t_2 = t_2.T

    # Find x and y coordinates of the triangles of first tessellation
    x_1 = p_1[0, t_1]
    y_1 = p_1[1, t_1]
    # Same with second tessellation
    x_2 = p_2[0, t_2]
    y_2 = p_2[1, t_2]

    intersections: list[tuple[int, int, float]] = []

    # Bounding box of each triangle for first and second tessellation
    min_x_1 = np.min(x_1, axis=1)
    max_x_1 = np.max(x_1, axis=1)
    min_y_1 = np.min(y_1, axis=1)
    max_y_1 = np.max(y_1, axis=1)

    min_x_2 = np.min(x_2, axis=1)
    max_x_2 = np.max(x_2, axis=1)
    min_y_2 = np.min(y_2, axis=1)
    max_y_2 = np.max(y_2, axis=1)

    # Represent the second tessellation using a Polygon from the shapely package
    poly_2 = [
        shapely_geometry.Polygon(
            [(x_2[j, 0], y_2[j, 0]), (x_2[j, 1], y_2[j, 1]), (x_2[j, 2], y_2[j, 2])]
        )
        for j in range(n_2)
    ]

    # Loop over all triangles in first tessellation, look for overlapping members in
    # second tessellation
    for i in range(n_1):
        # Polygon representation of the first triangle.
        poly_1 = shapely_geometry.Polygon(
            [(x_1[i, 0], y_1[i, 0]), (x_1[i, 1], y_1[i, 1]), (x_1[i, 2], y_1[i, 2])]
        )
        # Find triangles in the second tessellation that are outside the
        # bounding box of this triangle.
        right = np.squeeze(np.where(min_x_2 > max_x_1[i]))
        left = np.squeeze(np.where(max_x_2 < min_x_1[i]))
        above = np.squeeze(np.where(min_y_2 > max_y_1[i]))
        below = np.squeeze(np.where(max_y_2 < min_y_1[i]))

        # Candidates for intersection are only elements not outside
        outside = np.unique(np.hstack((right, left, above, below)))
        candidates = np.setdiff1d(np.arange(n_2), outside, assume_unique=True)

        # Loop over remaining candidates, call upon shapely to find
        # intersection
        for j in candidates:
            isect = poly_1.intersection(poly_2[j])
            if isinstance(isect, shapely_geometry.Polygon):
                intersections.append((i, j, isect.area))
    return intersections


def line_tessellation(
    p1: np.ndarray, p2: np.ndarray, l1: np.ndarray, l2: np.ndarray
) -> list[tuple[int, int, float]]:
    """Compute intersection of two line segment tessellations of a line.

    The function will identify partly overlapping line segments between l1 and
    l2, and compute their common length.

    Parameters:
        p1: ``shape=(3, n_p1)``

            Points in first tessellation.
        p2: ``shape=(3, n_p2)``

            Points in second tessellation.
        l1: ``shape=(2, n_tri_1)``

            Line segments in first tessellation, referring to indices in ``p2``.
        l2: ``shape= (2, n_tri_1)``

            Line segments in second tessellation, referring to indices in ``p2``.

    Raise:
        AssertionError: If ``porepy.intersections.segments_3d`` returns an unknown
            shape.

    Returns:
        List of tuples with each representing an overlap. The tuple consists of

        :obj:`int`:
            the index of the segment in the first tesselation.
        :obj:`int`:
            the index of the segment in the second tesselation.
        :obj:`float`:
            the common length of the two segments in the two tessellations.

    """
    # Loop over both set of lines, use segment intersection method to compute common
    # segments, thus areas.
    intersections: list[tuple[int, int, float]] = []
    for i in range(l1.shape[1]):
        start_1 = p1[:, l1[0, i]]
        end_1 = p1[:, l1[1, i]]
        for j in range(l2.shape[1]):
            start_2 = p2[:, l2[0, j]]
            end_2 = p2[:, l2[1, j]]
            X = segments_3d(start_1, end_1, start_2, end_2)
            if X is None:
                continue
            elif X.shape[1] == 1:  # Point intersection (zero measure)
                intersections.append((i, j, 0.0))
            elif X.shape[1] == 2:
                intersections.append((i, j, np.sqrt(np.sum((X[:, 0] - X[:, 1]) ** 2))))
            else:
                raise AssertionError()

    return intersections


def surface_tessellations(
    poly_sets: list[list[np.ndarray]], return_simplexes: bool = False
) -> tuple[list[np.ndarray], list[sps.csr_matrix]]:
    """Intersect a set of surface tessellations to find a finer subdivision that does
    not intersect with any of the input tessellations.

    It is assumed that the polygon sets are 2d.

    The implementation relies heavily on shapely's intersection finders.

    Parameters:
        poly_sets: Lists of polygons to be intersected.
        return_simplexes: ``default=False``

            If True, the subdivision is further split into a
            triangulation. The mappings from the original polygons is updated
            accordingly.

    Raises:
        NotImplementedError: If a triangulation of a non-convex polygon is attempted.
            Can only happen if ``return_simplexes`` is ``True``.

    Returns:
        Tuple consisting of

        list of :obj:`~numpy.ndarray`:
            Each element being a polygon so that the list together form a subdivision
            of the intersection of all polygons in the input sets.
        list of :obj:`~scipy.sparse.spmatrix`:
            Mappings from each of the input polygons to the intersected polygons. If
            the mapping's ``item[i][j, k]`` is non-zero, polygon ``k`` in set ``i``
            has a (generally partial) overlap with polygon ``j`` in the intersected
            polygon set. Specifically the value will be ``1``.

    """

    # local imports
    import shapely.geometry as shapely_geometry

    def _min_max_coord(coord):
        # Convenience function to get max and minimum coordinates for a set of polygons
        min_coord = np.array([c.min() for c in coord])
        max_coord = np.array([c.max() for c in coord])

        return min_coord, max_coord

    # Convert polygons into a more convenient data structure
    list_of_sets: list[tuple[np.ndarray, np.ndarray]] = []
    for poly in poly_sets:
        x = [poly[i][0] for i in range(len(poly))]
        y = [poly[i][1] for i in range(len(poly))]

        list_of_sets.append((x, y))  # type: ignore

    # The below algorithm relies heavily on shapely's functionality for intersection of
    # polygons. The idea is to intersect represent each set of polygons in the shapely
    # format, do the intersections with a new set of polygons to find a finer
    # intersection, and move on. Also keep track of the mapping from each of the sets of
    # polygons to the intersected mesh.

    # Initialize the intersection set as the first set of polygons
    poly_x, poly_y = list_of_sets[0]

    min_x_poly, max_x_poly = _min_max_coord(poly_x)
    min_y_poly, max_y_poly = _min_max_coord(poly_y)

    # poly_shapely will at any time represent the intersected polygon in shapely format,
    # for the currently covered set of polygon sets.
    poly_shapely: shapely_geometry.Polygon = []
    for px, py in zip(poly_x, poly_y):
        poly_shapely.append(
            shapely_geometry.Polygon([(px[i], py[i]) for i in range(px.size)])
        )

    # Data structure for mappings from original polygon sets to the intersected one
    # As the partition is extended to cover more polygons, a new mapping will be added
    # and the previous mappings are updated to account for the new intersection level.
    nc = len(poly_shapely)
    mappings: list[sps.csr_matrix] = [
        sps.dia_matrix((np.ones(nc, dtype=int), 0), shape=(nc, nc)).tocsr()
    ]

    # Loop over all set of polygons, do intersection with existing
    for i in range(1, len(list_of_sets)):
        # Represent this polygon as in shapely format. Also find max and min coordinates
        new_x, new_y = list_of_sets[i]
        min_x_new, max_x_new = _min_max_coord(new_x)
        min_y_new, max_y_new = _min_max_coord(new_y)
        new_shapely = []
        for px, py in zip(new_x, new_y):
            new_shapely.append(
                shapely_geometry.Polygon([(px[i], py[i]) for i in range(px.size)])
            )
        num_new = len(new_shapely)

        # Data structure to store the new intersected polygon
        isect_x, isect_y = [], []

        # Data structure to construct mappings to the new intersection from both this
        # and the previously covered polygons
        row_new, row_poly = [], []
        col_new, col_poly = [], []

        isect_counter = 0

        # Loop over all elements in the intersected polygon
        for j in range(len(poly_shapely)):
            # Find cells in the new polygon that are clearly outside this polygon This
            # corresponds to creating a box around this intersected polygon, and
            # disregard all new polygons clearly outside this box.
            right = np.squeeze(np.where(min_x_new > max_x_poly[j]))
            left = np.squeeze(np.where(max_x_new < min_x_poly[j]))
            above = np.squeeze(np.where(min_y_new > max_y_poly[j]))
            below = np.squeeze(np.where(max_y_new < min_y_poly[j]))

            outside = np.unique(np.hstack((right, left, above, below)))
            # Candidates are near this box
            candidates = np.setdiff1d(np.arange(num_new), outside, assume_unique=True)

            # Loop over remaining candidates, call upon shapely to find intersection
            for k in candidates:
                isect = poly_shapely[j].intersection(new_shapely[k])
                if isinstance(isect, shapely_geometry.Polygon):
                    # This is what must be done to get the coordinates from shapely
                    c = list(isect.exterior.coords)
                    # The shapely Polygon has the start/endpoint represented twice.
                    # Disregard the end.
                    isect_x.append(np.array([c[ci][0] for ci in range(len(c) - 1)]))
                    isect_y.append(np.array([c[ci][1] for ci in range(len(c) - 1)]))

                    # Build up the mapping to the new intersected polygon
                    col_new += [k]
                    col_poly += [j]
                    row_new += [isect_counter]
                    row_poly += [isect_counter]
                    isect_counter += 1

        # Mapping from the previously considered polygon to the newly found dissection.
        # This will be applied to update all previous mappings.
        matrix = sps.coo_matrix(
            (np.ones(isect_counter, dtype=int), (row_poly, col_poly)),
            shape=(isect_counter, len(poly_shapely)),
        ).tocsr()
        for mi in range(len(mappings)):
            mappings[mi] = matrix * mappings[mi]

        # Add a mapping between the current polygon and the newly found intersection.
        mappings.append(
            sps.coo_matrix(
                (np.ones(isect_counter, dtype=int), (row_new, col_new)),
                shape=(isect_counter, len(new_shapely)),
            ).tocsr()
        )

        # Define the new set of intersected polygons
        min_x_poly, max_x_poly = _min_max_coord(isect_x)
        min_y_poly, max_y_poly = _min_max_coord(isect_y)
        poly_shapely = []
        for px, py in zip(isect_x, isect_y):
            poly_shapely.append(
                shapely_geometry.Polygon([(px[i], py[i]) for i in range(px.size)])
            )

    # Finally, translate the intersected polygons back to a list of np.ndarrays
    isect_polys: list[np.ndarray] = [
        np.vstack((px, py)) for px, py in zip(isect_x, isect_y)
    ]

    if return_simplexes:
        # Finally, if requested, convert the subdivision into a triangulation. This
        # option is primarily intended for easy quadrature on the subdivision. Note that
        # no guarantees are given on the quality of the triangulation.

        # IMPLEMENTATION NOTE: This could have been turned into a separate function.
        # However, the code is only tested for a limited set of cases (specifically,
        # we have considered intersection of non-matching grids on surfaces), so it
        # seems premature to promote it to a general-purpose function.

        # We will need a triangulation below
        from scipy.spatial import Delaunay

        # Data structure for the mapping from isect_polys to the triangulation
        rows: list[int] = []
        cols: list[int] = []
        tri_counter: int = 0
        # Data structure for the triangulation
        tri: list[np.ndarray] = []

        # Loop over all isect_polys, split those with more than three vertices
        # EK: Somehow, mypy does not understand poly will be an np.ndarray, thus all
        # ignores
        for pi, poly in enumerate(isect_polys):  # type: ignore
            if poly.shape[1] == 3:  # type: ignore
                # Triangles can be used as they are
                tri.append(poly)  # type: ignore
                cols.append(pi)
                rows.append(tri_counter)
                tri_counter += 1

            else:
                # Check if the polygon is convex. Loop over the polygon vertices, and
                # check if they form a CW or CCW part of the polygon. If they all have
                # the same configuration, the polygon is convex

                # Three representation of the polygon vertices, by shifting their order
                start = poly
                # This is the vertex we test
                middle = np.roll(poly, -1, axis=1)  # type: ignore
                end = np.roll(poly, -2, axis=1)  # type: ignore
                # Use ccw test on all vertices in the polygon
                is_ccw = np.array(
                    [
                        pp.geometry_property_checks.is_ccw_polyline(
                            start[:, i], middle[:, i], end[:, i]  # type:ignore
                        )
                        for i in range(poly.shape[1])  # type:ignore
                    ]
                )

                if np.all(is_ccw) or np.all(np.logical_not(is_ccw)):
                    # This is a convex polygon. The triangulation can be formed by a
                    # Delaunay tessellation of the polygon. In an attempt to improve the
                    # quality of the simplexes, we add the center of the polygon
                    # (defined as the mean coordinate, should be fine since the polygon
                    # is convex) to the points to be triangulated. This may not be
                    # necessary, and should be up for revision. If the polygon has a bad
                    # shape, the triangulation will also have bad triangles - to improve
                    # we would need to do a more careful triangulation, adding more
                    # points
                    center = np.mean(poly, axis=1).reshape((-1, 1))  # type: ignore
                    ext_poly = np.hstack((poly, center)).T  # type: ignore
                    for t in Delaunay(ext_poly).simplices:
                        tri.append(ext_poly[t].T)
                        #
                        cols.append(pi)
                        rows.append(tri_counter)
                        tri_counter += 1
                else:
                    # For non-convex polygons, the Delaunay triangulation will generate
                    # simplexes not inside the polygon; specifically the triangulation
                    # will cover the convex hull of the polygon. These can likely be
                    # pruned by excluding triangles with a center not inside the polygon
                    # (would need a point-in-polygon test for non-convex polygons), but
                    # that would be for another day.
                    raise NotImplementedError("Non-convex polygons not covered")

        # Also update the mapping.
        matrix = sps.coo_matrix(
            (np.ones(len(rows), dtype=int), (rows, cols)),
            shape=(len(rows), len(isect_polys)),
        ).tocsr()

        for mi in range(len(mappings)):
            mappings[mi] = matrix * mappings[mi]

        isect_polys = tri

    return isect_polys, mappings


def split_intersecting_segments_2d(
    p: np.ndarray, e: np.ndarray, tol: float = 1e-8, return_argsort: bool = False
) -> Union[
    tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray],
    tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
]:
    """Process a set of points and connections between them so that the result
    is an extended point set and new connections that do not intersect.

    The function is written for gridding of fractured domains, but may be of use in
    other cases as well. The geometry is assumed to be 2D.

    The connections are defined by their start and endpoints, and can also have tags
    assigned. If so, the tags are preserved as connections are split. The connections
    are uniquified, so that no combination of point indices occurs more than once.

    Note:
        For (partly) overlapping segments, only one of the tags will survive the
        uniquification. The other can be reconstructed by using the third output.

    Parameters:
        p: ``shape=(2, n_pt)``

            Coordinates of points to be processed.
        e: ``shape=(n, n_con)``

            Connections between lines. n >= 2, row
            0 and 1 are index of start and endpoints, additional rows are tags.
        tol: ``default=1e-8``

            Tolerance used for comparing equal points.
        return_argsort: ``default=False``

            Return the mapping between the input segments and the output
            segments.

    Returns:
        A tuple with three or four entries (last only returned if ``return_argsort``
        is ``True``). The entries are as follows

        :obj:`~numpy.ndarray`: ``shape=(2, n_pt)``

            Points.
        :obj:`~numpy.ndarray`: ``shape=)2, n_edges)``

            new, non-intersecting edges.
        2-tuple of :obj:`~numpy.ndarray`:
            two arrays with length ``n_con`` with the first item being a set of tags,
            before uniquification of the edges, and the second being a column mapping
            from the unique edges to all edges. To recover lost tags associated with
            the points in column ``i``, first find all original columns which maps
            to ``i`` (``tuple[1] == i``), then recover the tags by the hits.
        :obj:`~numpy.ndarray`: ``shape=(n_edges,)``

            mapping of the new edges to the input edges.

    """
    if p.dtype == int:
        p = p.astype(float)

    # Find the bounding box
    x_min, x_max, y_min, y_max = _axis_aligned_bounding_box_2d(p, e)

    # If a polygon is perfectly aligned with a coordinate axis, and another polygon
    # terminates in the first one, rounding errors in the coordinates may lead to the
    # intersection not being picked up. To circumvent the issue, detect such situations
    # and give ourselves a bit wiggle room. It seems that this will not give problems in
    # other cases.
    for cmin, cmax in [(x_min, x_max), (y_min, y_max)]:
        hit = cmax - cmin < tol
        cmin[hit] -= 0.5 * tol
        cmax[hit] += 0.5 * tol

    # Identify fractures with overlapping bounding boxes
    pairs = _identify_overlapping_rectangles(x_min, x_max, y_min, y_max)

    # Identify all fractures that are the first (by index) of a potentially crossing
    # pair. A better way to group the fractures may be feasible, but this has not been
    # investigated.
    start_inds = np.unique(pairs[0])

    num_lines = e.shape[1]

    # Data structure for storage of intersection points. For each fracture,
    # we have an array that will contain the index of the intersections.
    isect_pt = np.empty(num_lines, dtype=object)
    for i in range(isect_pt.size):
        isect_pt[i] = np.empty(0, dtype=int)

    # Array of new points, found in the intersection of old ones.
    new_pts = []
    # The new points will be appended to the old ones, thus their index
    # must be adjusted.
    new_ind = p.shape[1]

    # Loop through all candidate pairs of intersecting fractures, check if they do
    # intersect. If so, store the point, and for each crossing fracture take note of the
    # index of the cross point.
    for _, line_ind in enumerate(start_inds):
        # First fracture in the candidate pair
        main = line_ind
        # Find all other fractures that is in a pair with the main as the first one.
        hit = np.where(pairs[0] == main)
        # Sort the other points; this makes debugging simpler if nothing else.
        other = np.sort(pairs[1, hit][0])

        # We will first do a coarse sorting, to rule out fractures that are clearly
        # not intersecting, and then do a finer search for an intersection below.

        # Utility function to pull out one or several points from an array based on
        # index
        def pt(p: np.ndarray, ind: np.ndarray) -> np.ndarray:
            a = p[:, ind]
            if ind.size == 1:
                return a.reshape((-1, 1))
            else:
                return a

        # Obtain start and endpoint of the main and other fractures
        start_main = pt(p, e[0, main])
        end_main = pt(p, e[1, main])
        start_other = pt(p, e[0, other])
        end_other = pt(p, e[1, other])

        # Utility function to normalize the fracture length
        def normalize(v):
            nrm = np.sqrt(np.sum(v**2, axis=0))

            # If the norm of the vector is essentially zero, do not normalize the vector
            hit = nrm < tol
            nrm[hit] = 1
            return v / nrm

        def dist(a, b):
            return np.sqrt(np.sum((a - b) ** 2))

        # Vectors along the main fracture, and from the start of the main to the start
        # and end of the other fractures. All normalized. If the other edges share start
        # or endpoint with the main one, normalization of the distance vector will make
        # the vector nans. In this case, we use another point along the other line, this
        # works equally well for the coarse identification (based on cross products). If
        # the segments are overlapping, there will still be issues with nans, but these
        # are dealt with below.
        main_vec = normalize(end_main - start_main)
        if dist(start_other, start_main) > tol:
            main_other_start = normalize(start_other - start_main)
        else:
            main_other_start = normalize(0.5 * (start_other + end_other) - start_main)
        if dist(end_other, start_main) > tol:
            main_other_end = normalize(end_other - start_main)
        else:
            # Values 0.3 and 0.7 are quite random here.
            main_other_end = normalize(0.3 * start_other + 0.7 * end_other - start_main)

        # Modified signum function: The value is 0 if it is very close to zero.
        def mod_sign(v, tol):
            sgn = np.sign(v)
            sgn[np.abs(v) < tol] = 0
            return sgn

        # Take the cross product between the vector along the main line, and the
        # vectors to the start and end of the other lines, respectively.
        start_cross = mod_sign(
            main_vec[0] * main_other_start[1] - main_vec[1] * main_other_start[0], tol
        )
        end_cross = mod_sign(
            main_vec[0] * main_other_end[1] - main_vec[1] * main_other_end[0], tol
        )

        # If the start and endpoint of the other fracture are clearly on the same side
        # of the main one, these are not crossing. For completely ovrelapping edges, the
        # normalization will leave the vectors nan. There may be better ways of dealing
        # with this, but we simply run the intersection finder in this case.
        relevant = np.where(
            np.logical_or(
                (start_cross * end_cross < 1),
                np.any(np.isnan(main_other_start + main_other_end), axis=0),
            )
        )[0]

        # Loop over all relevant (possibly crossing) fractures, look closer for an
        # intersection.
        for ri in relevant:
            ipt = segments_2d(
                start_main, end_main, pt(start_other, ri), pt(end_other, ri), tol
            )
            # Add the intersection point, if any. If two intersection points are found,
            # that is the edges are overlapping both points are added.
            if ipt is not None:
                num_isect = ipt.shape[1]
                # Add indices of the new points to the main and other edge
                isect_pt[main] = np.append(
                    isect_pt[main], new_ind + np.arange(num_isect)
                )
                isect_pt[other[ri]] = np.append(
                    isect_pt[other[ri]], new_ind + np.arange(num_isect)
                )
                new_ind += num_isect

                # Add the one or two intertion points
                if num_isect == 1:
                    new_pts.append(ipt.squeeze())
                else:
                    # It turned out the transport was needed to get the code to work
                    new_pts.append(ipt.squeeze().T)

    # If we have found no intersection points, we can safely return the incoming
    # points and edges.
    if len(new_pts) == 0:
        # Tag information is trivial in this case
        tags = e[2:].copy()
        mapping = np.arange(e.shape[1])
        tag_info = (tags, mapping)
        if return_argsort:
            return p, e, tag_info, np.arange(e.shape[1])
        else:
            return p, e, tag_info

    # If intersection points are found, the intersecting lines must be split into
    # shorter segments.
    else:
        # The full set of points, both original and newly found intersection points
        all_pt = np.hstack((p, np.vstack([i for i in new_pts]).T))
        # Remove duplicates in the point set.
        # NOTE: The tolerance used here is a bit sensitive, if set too loose, this
        # may merge non-intersecting fractures.

        unique_all_pt, ia, ib = pp.utils.setmembership.uniquify_point_set(all_pt, tol)

        # Data structure for storing the split edges.
        new_edge = np.empty((e.shape[0], 0), dtype=int)
        argsort: np.ndarray = np.empty(0, dtype=int)

        # Loop over all lines, split it into non-overlapping segments.
        for ei in range(num_lines):
            # Find indices of all points involved in this fracture. Map them to the
            # unique point set, and uniquify
            inds = np.unique(ib[np.hstack((e[:2, ei], isect_pt[ei]))])
            num_branches = inds.size - 1
            # Get the coordinates themselves.
            loc_pts = pt(unique_all_pt, inds)
            # Specifically get the start point: Pick one of the points of the original
            # edge, e[0, ei], which is known to be at an end of the edge. map to the
            # unique indices
            loc_start = pt(unique_all_pt, ib[e[0, ei]])
            # Measure the distance of the points from the start. This can be used to
            # sort the points along the line
            dist = np.sum((loc_pts - loc_start) ** 2, axis=0)
            assert isinstance(dist, np.ndarray)  # Needed to appease mypy
            order = np.argsort(dist)
            new_inds = inds[order]
            # All new segments share the tags of the old one.
            loc_tags = e[2:, ei].reshape((-1, 1)) * np.ones(num_branches, dtype=int)
            # Define the new segments, in terms of the unique points
            loc_edge = np.vstack((new_inds[:-1], new_inds[1:], loc_tags))

            # Add to the global list of segments
            new_edge = np.hstack((new_edge, loc_edge))
            argsort = np.hstack((argsort, [ei] * loc_edge.shape[1]))

        # Finally, uniquify edges. This operation is necessary for overlapping edges.
        # Operate on sorted point indices per edge
        new_edge[:2] = np.sort(new_edge[:2], axis=0)

        # Keep the old tags before uniquifying
        tags = new_edge[2:].copy().ravel()
        # Uniquify.
        _, edge_map, all_2_unique = pp.utils.setmembership.unique_columns_tol(
            new_edge[:2].astype(int), tol
        )
        tag_info = (tags, all_2_unique)

        new_edge = new_edge[:, edge_map]
        argsort = argsort[edge_map]

        if return_argsort:
            return unique_all_pt, new_edge.astype(int), tag_info, argsort
        else:
            return unique_all_pt, new_edge.astype(int), tag_info


def _axis_aligned_bounding_box_2d(
    p: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For a set of lines in 2d, obtain the bounding box for each line.

    The lines are specified as a list of points, together with connections between
    the points.

    Parameters:
        p: ``shape=(2, n_pt)``

            Coordinates of points to be processed.
        e: ``shape = (n, n_con)``

            Connections between lines. ``n >= 2``, row 0 and 1 are indices of start and
            endpoints, additional rows are tags

    Returns:
        Tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``
            Minimum x-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``
            Maximum x-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``
            Minimum y-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``
            Maximum y-coordinate for all lines.

    """
    x = p[0]
    y = p[1]

    x_0 = x[e[0]]
    x_1 = x[e[1]]
    y_0 = y[e[0]]
    y_1 = y[e[1]]

    x_min = np.minimum(x_0, x_1)
    x_max = np.maximum(x_0, x_1)
    y_min = np.minimum(y_0, y_1)
    y_max = np.maximum(y_0, y_1)

    return x_min, x_max, y_min, y_max


def _axis_aligned_bounding_box_3d(
    polys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For a set of polygons embedded in 3d, obtain the bounding box for each object.

    The polygons are specified as a list of numpy arrays.

    Parameters:
        p: ``shape=(3, n_pt)``

            Each list element specifies a polygon,
            described by its vertices in a ``(3, num_points)`` array.

    Returns:
        Tuple consisting of

        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Minimum x-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Maximum x-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Minimum y-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Maximum y-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Minimum z-coordinate for all lines.
        :obj:`~numpy.ndarray`: ``shape=(n_pt,)``

            Maximum z-coordinate for all lines.

    """

    polys = list(polys)

    num_poly = len(polys)

    x_min = np.empty(num_poly)
    x_max = np.empty_like(x_min)
    y_min = np.empty_like(x_min)
    y_max = np.empty_like(x_min)
    z_min = np.empty_like(x_min)
    z_max = np.empty_like(x_min)

    for ind, p in enumerate(polys):
        x_min[ind] = np.min(p[0])
        x_max[ind] = np.max(p[0])
        y_min[ind] = np.min(p[1])
        y_max[ind] = np.max(p[1])
        z_min[ind] = np.min(p[2])
        z_max[ind] = np.max(p[2])

    return x_min, x_max, y_min, y_max, z_min, z_max


def _identify_overlapping_intervals(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Based on a set of start and end coordinates for intervals, identify pairs of
    overlapping intervals.

    Note:
        For all corresponding entries in ``left`` and ``right``, ``left <= right``
        is required, but equality is allowed.

    Parameters:
        left: ``shape=(num_intervals,)``

            Minimum coordinates of the intervals.
        right: ``shape=(num_intervals,)``

            Maximum coordinates of the intervals.

    Returns:
        Array with shape ``(2, num_overlaps)`` with each column containing a pair of
        overlapping intervals, refering to their placement in left and right.
        The pairs are sorted so that the lowest index is in the first column.

    """
    # There can be no overlaps if there is less than two intervals
    if left.size < 2:
        return np.empty((2, 0))

    # Sort the coordinates
    sort_ind_left = np.argsort(left)
    sort_ind_right = np.argsort(right)

    # pointers to the next start and end point of an interval
    next_right = 0
    next_left = 0

    # List of pairs we have found
    pairs: list = []
    # List of intervals we are currently in. All intervals will join and leave this set.
    active: list = []

    num_lines = left.size

    # Loop through the line, add and remove intervals as we come across them.
    while True:
        # Check if the next start (left) point is before the next endpoint, but only if
        # there are more left points available. Less or equal is critical here, or else
        # cases where a point interval is combined with the start of another interval
        # may not be discovered.
        if (
            next_left < num_lines
            and left[sort_ind_left[next_left]] <= right[sort_ind_right[next_right]]
        ):
            # We have started a new interval. This will be paired with all active
            # intervals
            for a in active:
                pairs.append([a, sort_ind_left[next_left]])
            # Also join the new intervals to the active set.
            active.append(sort_ind_left[next_left])
            # Increase the index
            next_left += 1
        else:
            # We have reached the end of the interval - remove it from the active ones.
            active.remove(sort_ind_right[next_right])
            next_right += 1
            # Check if we have come to the end
            if next_right == num_lines:
                break

    if len(pairs) == 0:
        return np.empty((2, 0))
    else:
        final_pairs = np.asarray(pairs).T
        # First sort the pairs themselves
        final_pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(final_pairs[0])
        final_pairs = final_pairs[:, sort_ind]
        return final_pairs


def _identify_overlapping_rectangles(
    xmin: np.ndarray,
    xmax: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Based on a set of start and end coordinates for bounding boxes, identify pairs
    of overlapping rectangles.

    Note:
        For all corresponding entries it has to hold ``xmin <= xmax``, but equality
        is allowed. Analogously for the y-components.

    Note:
        The algorithm was found in 'A fast method for fracture intersection detection in
        discrete fracture networks' by Dong et al, Computers and Geotechniques 2018.

    Parameters:
        xmin: ``shape=(num_rectangles,)``

            Minimum coordinates of the rectangle on the
            first axis.
        xmax: ``shape=(num_rectangles,)``

            Maximum coordinates of the rectangle on the
            first axis.
        ymin: ``shape=(num_rectangles,)``

            Minimum coordinates of the rectangle on the
            second axis.
        ymax: ``shape=(num_rectangles,)``

            Maximum coordinates of the rectangle on the
            second axis.

    Returns:
        Array with shape ``(2, num_overlaps)`` with each column containing a pair of
        overlapping intervals, refering to their placement in left and right. The pairs
        are sorted so that the lowest index is in the first column.

    """
    # There can be no overlaps if there is less than two rectangles
    if xmin.size < 2:
        return np.empty((2, 0))

    # Sort the coordinates
    sort_ind_min = np.argsort(xmin)
    sort_ind_max = np.argsort(xmax)

    # pointers to the next start and end point of an interval
    next_min = 0
    next_max = 0

    # List of pairs we have found
    pairs = []
    # List of intervals we are currently in. All intervals will join and leave this
    # set.
    active: list = []

    num_lines = xmax.size

    # Pass along the x-axis, identify the start and end of rectangles as we go. The idea
    # is then for each new interval to check which of the active intervals also have
    # overlap along the y-axis. These will be identified as pairs.
    while True:
        # Check if the next start (xmin) point is before the next endpoint, but only if
        # there are more left points available. Less or equal is critical here, or else
        # cases where a point interval is combined with the start of another interval
        # may not be discovered.
        if (
            next_min < num_lines
            and xmin[sort_ind_min[next_min]] <= xmax[sort_ind_max[next_max]]
        ):
            # Find active rectangles where the y-interval is also overlapping
            between = np.where(
                np.logical_and(
                    ymax[sort_ind_min[next_min]] >= ymin[active],
                    ymin[sort_ind_min[next_min]] <= ymax[active],
                )
            )[0]
            # For all identified overlaps, add the new pairs
            for a in between:
                pairs.append([active[a], sort_ind_min[next_min]])
            # Add this to the active rectangles, and increase the index
            active.append(sort_ind_min[next_min])
            next_min += 1
        else:
            # We are leaving a rectangle.
            active.remove(sort_ind_max[next_max])
            next_max += 1
            # Check if we have come to the end
            if next_max == num_lines:
                break

    if len(pairs) == 0:
        return np.empty((2, 0))
    else:
        final_pairs = np.asarray(pairs).T
        # First sort the pairs themselves
        final_pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(final_pairs[0])
        final_pairs = final_pairs[:, sort_ind]
        return final_pairs


def _intersect_pairs(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """For two lists containing pair of indices, find the intersection.

    Parameters:
        p1: ``shape=(2, n)``

            Each column contains a pair of indices.
        p2: ``shape=(2, m)``

            Each column contains a pair of indices.

    Returns:
        Array with shape ``(2, k)`` where ``k <= min(n, m)`` with each column containing
        a pair of indices that are found in both ``p1`` and ``p2``. The array is sorted
        so that items in the first row is less or equal to the second row. The columns
        are sorted according to the numbers in the first row.

    """
    # Special treatment of empty lists
    if p1.size == 0 or p2.size == 0:
        return np.empty((2, 0))
    else:
        # Do the intersection
        _, ind = pp.utils.setmembership.ismember_rows(p1, p2)
        pairs = p2[:, ind]

        # First sort the pairs themselves
        pairs.sort(axis=0)
        # Next, sort the columns so that the first row is non-decreasing
        sort_ind = np.argsort(pairs[0])
        pairs = pairs[:, sort_ind]
        return pairs
