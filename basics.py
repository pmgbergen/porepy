import numpy as np
from math import sqrt
import sympy


def snap_to_grid(pts, box=None, precision=1e-3):

    nd = pts.shape[0]

    if box is None:
        box = np.reshape(np.ones(nd), (nd, 1))

    # Precission in each direction
    delta = box * precision
    pts = np.rint(pts.astype(np.float64) / delta) * delta
    return pts


def __nrm(v):
    return np.sqrt(np.sum(v * v, axis=0))


def __dist(p1, p2):
    return __nrm(p1 - p2)

#
# def is_unique_vert_array(pts, box=None, precision=1e-3):
#     nd = pts.shape[0]
#     num_pts = pts.shape[1]
#     pts = snap_to_grid(pts, box, precision)
#     for iter in range(num_pts):
#


def __points_equal(p1, p2, box, precesion=1e-3):
    d = __dist(p1, p2)
    nd = p1.shape[0]
    return d < precesion * sqrt(nd)


def split_edge(vertices, edges, edge_ind, new_pt, box, precision):

    start = edges[0, edge_ind]
    end = edges[1, edge_ind]
    tags = edges[2:, edge_ind]

    vertices, pt_ind, _ = add_point(vertices, new_pt, box, precision)
    if start == pt_ind or end == pt_ind:
        new_line = False
        return vertices, edges, new_line
    if tags.size > 0:
        new_edges = np.vstack((np.array([[start, pt_ind],
                                         [pt_ind, end]]),
                               np.tile(tags[:, np.newaxis], 2)))
    else:
        new_edges = np.array([[start, pt_ind],
                             [pt_ind, end]])
    edges = np.hstack((edges[:, :edge_ind], new_edges, edges[:, edge_ind+1:]))
    new_line = True
    return vertices, edges, new_line


def add_point(vertices, pt, box=None, precision=1e-3):
    nd = vertices.shape[0]
    vertices = snap_to_grid(vertices, box, precision)
    pt = snap_to_grid(pt, box, precision)
    dist = __dist(pt, vertices)
    min_dist = np.min(dist)
    if min_dist < precision * nd:
        ind = np.argmin(dist)
        new_point = None
        return vertices, ind, new_point
    vertices = np.append(vertices, pt, axis=1)
    ind = vertices.shape[1]-1
    return vertices, ind, pt


def lines_intersect(start_1, end_1, start_2, end_2):
    # Check if lines intersect. For the moment, we do this by methods
    # incorpoated in sympy. The implementation can be done by pure algebra
    # if necessary (although this is a bit dirty).
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
        # Should this be a column vector?
        return np.array([[p.x], [p.y]])


def remove_edge_crossings(vertices, edges, box=None, precision=1e-3):
    """

    Parameters
    ----------
    vertices
    edges
    box
    precision

    Returns
    -------

    """
    num_edges = edges.shape[1]
    nd = vertices.shape[0]

    # Only 2D is considered. 3D should not be too dificult, but it is not
    # clear how relevant it is
    if nd != 2:
        raise NotImplementedError('Only 2D so far')

    edge_counter = 0

    # Loop over all edges, search for intersections. The number of edges can
    #  change due to splitting.
    while edge_counter < num_edges:
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
            intersections = intersections[0]
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
                # means the intersection runs through and endpoint of the
                # edge, in which case )
                vertices, edges, split_outer_edge = split_edge(vertices, edges,
                                                               edge_counter,
                                                               new_pt, box,
                                                               precision)
                # If the outer edge (represented by edge_counter) was split,
                # e.g. inserted into the list of edges we need to increase the
                # index of the inner edge
                intsect += split_outer_edge
                # Possibly split the inner edge
                vertices, edges, split_inner_edge = split_edge(vertices,  edges,
                                                               intsect,
                                                               new_pt, box,
                                                               precision)
                # Update index of possible intersections
                intersections += split_outer_edge + split_inner_edge

            # We're done with this candidate edge. Increase index of inner loop
            int_counter += 1
        # We're done with all intersectiosn of this loop. increase index of
        # outer loop
        edge_counter += 1
    return vertices, edges


if __name__ == '__main__':
    p = np.array([[-1, 1, 0, 0],
                  [0, 0, -1, 1]])
    lines = np.array([[0, 2],
                      [1, 3],
                      [1, 2],
                      [3, 4]])
    box = np.array([[2], [2]])
    new_pts, new_lines = remove_edge_crossings(p, lines, box)
    assert np.allclose(new_pts, p)
    assert np.allclose(new_lines, lines)

