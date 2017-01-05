"""
Various utility functions related to computational geometry.

Some functions (add_point, split_edges, ...?) are mainly aimed at finding
intersection between lines, with grid generation in mind, and should perhaps
be moved to a separate module.

"""

import numpy as np
from math import sqrt
import sympy

#------------------------------------------------------------------------------#

def snap_to_grid(pts, precision=1e-3, box=None):
    """
    Snap points to an underlying Cartesian grid.
    Used e.g. for avoiding rounding issues when testing for equality between
    points.

    Anisotropy in the rounding rules can be enforced by the parameter box.

    >>> snap_to_grid([[0.2443], [0.501]])
    array([[ 0.244],
    	   [ 0.501]])

    >>> snap_to_grid([[0.2443], [0.501]], box=[[10], [1]])
    array([[ 0.24 ],
           [ 0.501]])

    >>> snap_to_grid([[0.2443], [0.501]], precision=0.01)
    array([[ 0.24],
           [ 0.5 ]])

    Parameters:
        pts (np.ndarray, nd x n_pts): Points to be rounded.
        box (np.ndarray, nd x 1, optional): Size of the domain, precision will
            be taken relative to the size. Defaults to unit box.
        precision (double, optional): Resolution of the underlying grid.

    Returns:
        np.ndarray, nd x n_pts: Rounded coordinates.

    """

    pts = np.asarray(pts)

    nd = pts.shape[0]

    if box is None:
        box = np.reshape(np.ones(nd), (nd, 1))
    else:
        box = np.asarray(box)

    # Precission in each direction
    delta = box * precision
    pts = np.rint(pts.astype(np.float64) / delta) * delta
    return pts

#------------------------------------------------------------------------------#

def __nrm(v):
    return np.sqrt(np.sum(v * v, axis=0))

#------------------------------------------------------------------------------#

def __dist(p1, p2):
    return __nrm(p1 - p2)

#------------------------------------------------------------------------------#
#
# def is_unique_vert_array(pts, box=None, precision=1e-3):
#     nd = pts.shape[0]
#     num_pts = pts.shape[1]
#     pts = snap_to_grid(pts, box, precision)
#     for iter in range(num_pts):
#

#------------------------------------------------------------------------------#

def __points_equal(p1, p2, box, precesion=1e-3):
    d = __dist(p1, p2)
    nd = p1.shape[0]
    return d < precesion * sqrt(nd)

#------------------------------------------------------------------------------#

def split_edge(vertices, edges, edge_ind, new_pt, **kwargs):
    """
    Split a line into two by introcuding a new point.
    Function is used e.g. for gridding purposes.

    The input can be a set of points, and lines between them, of which one is
    to be split.

    A new line will be inserted, unless the new point coincides with the
    start or endpoint of the edge (under the given precision).

    The code is intended for 2D, in 3D use with caution.

    Examples:
        >>> p = np.array([[0, 0], [0, 1]])
        >>> edges = np.array([[0], [1]])
        >>> new_pt = np.array([[0], [0.5]])
        >>> v, e, nl = split_edge(p, edges, 0, new_pt)
        >>> e
        array([[0, 2],
               [2, 1]])

    Parameters:
        vertices (np.ndarray, nd x num_pt): Points of the vertex sets.
        edges (np.ndarray, n x num_edges): Connections between lines. If n>2,
            the additional rows are treated as tags, that are preserved under
            splitting.
        edge_ind (int): index of edge to be split, refering to edges.
        new_pt (np.ndarray, nd x 1): new point to be inserted. Assumed to be
            on the edge to be split.
        **kwargs: Arguments passed to snap_to_grid

    Returns:
        np.ndarray, nd x n_pt: new point set, possibly with new point inserted.
        np.ndarray, n x n_con: new edge set, possibly with new lines defined.
        boolean: True if a new line is created, otherwise false.

    """
    start = edges[0, edge_ind]
    end = edges[1, edge_ind]
    # Save tags associated with the edge.
    tags = edges[2:, edge_ind]

    # Add a new point
    vertices, pt_ind, _ = add_point(vertices, new_pt, **kwargs)
    # If the new point coincide with the start point, nothing happens
    if start == pt_ind or end == pt_ind:
        new_line = False
        return vertices, edges, new_line

    # If we get here, we know that a new point has been created.

    # Add any tags to the new edge.
    if tags.size > 0:
        new_edges = np.vstack((np.array([[start, pt_ind],
                                         [pt_ind, end]]),
                               np.tile(tags[:, np.newaxis], 2)))
    else:
        new_edges = np.array([[start, pt_ind],
                              [pt_ind, end]])
    # Insert the new edge in the midle of the set of edges.
    edges = np.hstack((edges[:, :edge_ind], new_edges, edges[:, edge_ind+1:]))
    new_line = True
    return vertices, edges, new_line

#------------------------------------------------------------------------------#

def add_point(vertices, pt, precision=1e-3, **kwargs):
    """
    Add a point to a point set, unless the point already exist in the set.

    Point coordinates are compared relative to an underlying Cartesian grid,
    see snap_to_grid for details.

    The function is created with gridding in mind, but may be useful for other
    purposes as well.

    Parameters:
        vertices (np.ndarray, nd x num_pts): existing point set
        pt (np.ndarray, nd x 1): Point to be added
        precesion (double): Precision of underlying Cartesian grid
        **kwargs: Arguments passed to snap_to_grid

    Returns:
        np.ndarray, nd x n_pt: New point set, possibly containing a new point
        int: Index of the new point added (if any). If not, index of the
            closest existing point, i.e. the one that made a new point
            unnecessary.
        np.ndarray, nd x 1: The new point, or None if no new point was needed.

    """
    if not 'precision' in kwargs:
        kwargs['precision'] = precision

    nd = vertices.shape[0]
    # Before comparing coordinates, snap both existing and new point to the
    # underlying grid
    vertices = snap_to_grid(vertices, **kwargs)
    pt = snap_to_grid(pt, **kwargs)

    # Distance 
    dist = __dist(pt, vertices)
    min_dist = np.min(dist)

    if min_dist < precision * np.sqrt(nd):
    	# No new point is needed
        ind = np.argmin(dist)
        new_point = None
        return vertices, ind, new_point
    else:
        ind = vertices.shape[1]-1
        # Append the new point at the end of the point list
        vertices = np.append(vertices, pt, axis=1)
        ind = vertices.shape[1] - 1
        return vertices, ind, pt

#------------------------------------------------------------------------------#

def lines_intersect(start_1, end_1, start_2, end_2):
    """
    Check if two line segments defined by their start end endpoints, intersect.

    The lines are assumed to be in 2D.

    The function uses sympy to find intersections. At the moment (Jan 2017),
    sympy is not very effective, so this may become a bottleneck if the method
    is called repeatedly. An purely algebraic implementation is simple, but
    somewhat cumbersome.

    Note that, oposed to other functions related to grid generation such as
    remove_edge_crossings, this function does not use the concept of
    snap_to_grid. This may cause problems at some point, although no issues
    have been discovered so far.

    Example:
        >>> lines_intersect([0, 0], [1, 1], [0, 1], [1, 0])
        array([[ 0.5],
	       [ 0.5]])

        >>> lines_intersect([0, 0], [1, 0], [0, 1], [1, 1])

    Parameters:
        start_1 (np.ndarray or list): coordinates of start point for first
            line.
        end_1 (np.ndarray or list): coordinates of end point for first line.
        start_2 (np.ndarray or list): coordinates of start point for first
            line.
        end_2 (np.ndarray or list): coordinates of end point for first line.

    Returns:
        np.ndarray: coordinates of intersection point, or None if the lines do
            not intersect.
    """


    # It seems that if sympy is provided point coordinates as integers, it may
    # do calculations in integers also, with an unknown approach to rounding.
    # Cast the values to floats to avoid this. It is not the most pythonic
    # style, but tracking down such a bug would have been a nightmare.
    start_1 = np.asarray(start_1).astype(np.float)
    end_1 = np.asarray(end_1).astype(np.float)
    start_2 = np.asarray(start_2).astype(np.float)
    end_2 = np.asarray(end_2).astype(np.float)

    p1 = sympy.Point(start_1[0], start_1[1])
    p2 = sympy.Point(end_1[0], end_1[1])
    p3 = sympy.Point(start_2[0], start_2[1])
    p4 = sympy.Point(end_2[0], end_2[1])

    l1 = sympy.Segment(p1, p2)
    l2 = sympy.Segment(p3, p4)

    isect = l1.intersection(l2)
    if isect is None or len(isect) == 0:
        return None
    else:
        p = isect[0]
        return np.array([[p.x], [p.y]], dtype='float')

#------------------------------------------------------------------------------#

def remove_edge_crossings(vertices, edges, **kwargs):
    """
    Process a set of points and connections between them so that the result
    is an extended point set and new connections that do not intersect.

    The function is written for gridding of fractured domains, but may be
    of use in other cases as well. The geometry is assumed to be 2D, (the
    corresponding operation in 3D requires intersection between planes, and
    is a rather complex, although doable, task).

    The connections are defined by their start and endpoints, and can also
    have tags assigned. If so, the tags are preserved as connections are split.

    Parameters:
	vertices (np.ndarray, 2 x n_pt): Coordinates of points to be processed
	edges (np.ndarray, n x n_con): Connections between lines. n >= 2, row
            0 and 1 are index of start and endpoints, additional rows are tags
        **kwargs: Arguments passed to snap_to_grid

    Returns:
	np.ndarray, (2 x n_pt), array of points, possibly expanded.
	np.ndarray, (n x n_edges), array of new edges. Non-intersecting.

    Raises:
	NotImplementedError if a 3D point array is provided.

    """
    num_edges = edges.shape[1]
    nd = vertices.shape[0]

    # Only 2D is considered. 3D should not be too dificult, but it is not
    # clear how relevant it is
    if nd != 2:
        raise NotImplementedError('Only 2D so far')

    edge_counter = 0

    vertices = snap_to_grid(vertices, **kwargs)

    # Loop over all edges, search for intersections. The number of edges can
    #  change due to splitting.
    while edge_counter < edges.shape[1]:
        # The direct test of whether two edges intersect is somewhat
        # expensive, and it is hard to vectorize this part. Therefore,
        # we first identify edges which crosses the extention of this edge (
        # intersection of line and line segment). We then go on to test for
        # real intersections.


        # Find start and stop coordinates for all edges
        start_x = vertices[0, edges[0]]
        start_y = vertices[1, edges[0]]
        end_x = vertices[0, edges[1]]
        end_y = vertices[1, edges[1]]

        a = end_y - start_y
        b = -(end_x - start_x)
        xm = (start_x[edge_counter] + end_x[edge_counter]) / 2
        ym = (start_y[edge_counter] + end_y[edge_counter]) / 2

        # For all lines, find which side of line i it's two endpoints are.
        # If c1 and c2 have different signs, they will be on different sides
        # of line i. See
        #
        # http://stackoverflow.com/questions/385305/efficient-maths-algorithm-to-calculate-intersections
        #
        # for more information.
        c1 = a[edge_counter] * (start_x - xm) \
             + b[edge_counter] * (start_y - ym)
        c2 = a[edge_counter] * (end_x - xm) + b[edge_counter] * (end_y - ym)
        line_intersections = np.sign(c1) != np.sign(c2)

        # Find elements which may intersect.
        intersections = np.argwhere(line_intersections)
        # np.argwhere returns an array of dimensions (1, dim), so we reduce
        # this to truly 1D, or simply continue with the next edge if there
        # are no candidate edges
        if intersections.size > 0:
            intersections = intersections.ravel(0)
        else:
            # There are no candidates for intersection
            edge_counter += 1
            continue

        int_counter = 0
        while intersections.size > 0 and int_counter < intersections.size:
            # Line intersect (inner loop) is an intersection if it crosses
            # the extension of line edge_counter (outer loop) (ie intsect it
            #  crosses the infinite line that goes through the endpoints of
            # edge_counter), but we don't know if it actually crosses the
            # line segment edge_counter. Now we do a more refined search to
            # find if the line segments intersects. Note that there is no
            # help in vectorizing lines_intersect and computing intersection
            #  points for all lines in intersections, since line i may be
            # split, and the intersection points must recalculated. It may
            # be possible to reorganize this while-loop by computing all
            # intersection points(vectorized), and only recompuing if line
            # edge_counter is split, but we keep things simple for now.
            intsect = intersections[int_counter]

            # Check if this point intersects
            new_pt = lines_intersect(vertices[:, edges[0, edge_counter]],
                                       vertices[:, edges[1, edge_counter]],
                                       vertices[:, edges[0, intsect]],
                                       vertices[:, edges[1, intsect]])

            if new_pt is not None:
                # Split edge edge_counter (outer loop), unless the
                # intersection hits an existing point (in practices this
                # means the intersection runs through an endpoint of the
                # edge in an L-type configuration, in which case no new point
                # is needed)
                vertices, edges, split_outer_edge = split_edge(vertices, edges,
                                                               edge_counter,
                                                               new_pt,
                                                               **kwargs)
                # If the outer edge (represented by edge_counter) was split,
                # e.g. inserted into the list of edges we need to increase the
                # index of the inner edge
                intsect += split_outer_edge
                # Possibly split the inner edge
                vertices, edges, split_inner_edge = split_edge(vertices, edges,
                                                               intsect,
                                                               new_pt,
                                                               **kwargs)
                # Update index of possible intersections
                intersections += split_outer_edge + split_inner_edge

            # We're done with this candidate edge. Increase index of inner loop
            int_counter += 1
        # We're done with all intersections of this loop. increase index of
        # outer loop
        edge_counter += 1
    return vertices, edges

#------------------------------------------------------------------------------#

def is_planar( pts, normal = None ):
    """ Check if the points lie on a plane.

    Parameters:
    pts (np.ndarray, 3xn): the points.
    normal: (optional) the normal of the plane, otherwise three points are
        required.

    Returns:
    check, bool, if the points lie on a plane or not.

    """

    if normal is None:
        normal = compute_normal( pts )
    else:
        normal = normal / np.linalg.norm( normal )

    check = np.array( [ np.isclose( np.dot( normal, pts[:,0] - p ), 0. ) \
                      for p in pts[:,1:].T ], dtype=np.bool )
    return np.all( check )

#------------------------------------------------------------------------------#

def project_plane_matrix( pts, normal = None ):
    """ Project the points on a plane using local coordinates.

    The projected points are computed by a dot product.
    example: np.array( [ np.dot( R, p ) for p in pts.T ] ).T
    
    Parameters:
        pts (np.ndarray, 3xn): the points.
        normal: (optional) the normal of the plane, otherwise three points are
            required.

    Returns:
    	np.ndarray, 3x3, projection matrix.

    """
    if normal is None: 
        normal = compute_normal( pts )
    else:
        normal = normal / np.linalg.norm( normal )

    reference = np.array( [0., 0., 1.] )
    angle = np.arccos( np.dot( normal, reference ) )
    vect = np.cross( normal, reference )
    return rot( angle, vect )

#------------------------------------------------------------------------------#

def rot( a, vect ):
    """ Compute the rotation matrix about a vector by an angle using the matrix
    form of Rodrigues formula.

    Parameters:
        a: double, the angle.
        vect: np.array, 3, the vector.

    Returns:
        matrix: np.ndarray, 3x3, the rotation matrix.

    """
    if np.allclose( vect, [0.,0.,0.] ): 
        return np.identity(3)
    vect = vect / np.linalg.norm( vect )
    W = np.array( [ [       0., -vect[2],  vect[1] ],
                    [  vect[2],       0., -vect[0] ],
                    [ -vect[1],  vect[0],       0. ] ] )
    return np.identity(3) + np.sin(a)*W + \
           (1.-np.cos(a))*np.linalg.matrix_power(W,2)

#------------------------------------------------------------------------------#

def compute_normal( pts ):
    """ Compute the normal of a set of points.

    The algorithm assume that the points lie on a plane.
    Three points are required.

    Parameters:
        pts: np.ndarray, 3xn, the points.

    Returns:
        normal: np.array, 1x3, the normal.

    """
    assert( pts.shape[1] > 2 )
    normal = np.cross( pts[:,0] - pts[:,1], pts[:,0] - np.mean( pts, axis = 1 ) )
    return normal / np.linalg.norm( normal )

#------------------------------------------------------------------------------#
