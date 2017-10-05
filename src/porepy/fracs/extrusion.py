import numpy as np

from porepy.utils import comp_geom as cg
from porepy.fracs.fractures import EllipticFracture, Fracture


def _intersection_by_num_node(edges, num):
    """ Find all edges involved in intersections with a certain number of
    intersecting lines.

    Parameters:
        edges: fractures
        num: Target number of nodes in intersections

    Returns:
        crosses: Nodes with the prescribed number of edges meeting.
        edges_of_crosses (n x num): Each row gives edges meeting in a node.

    """
    num_occ = np.bincount(edges[:2].ravel())
    crosses = np.where(num_occ == num)[0]

    num_crosses = crosses.size

    edges_of_crosses = np.zeros((num_crosses, num), dtype=np.int)
    for i, pi in enumerate(crosses):
        edges_of_crosses[i] = np.where(np.any(edges[:2] == pi, axis=0))[0]
    return crosses, edges_of_crosses


def t_intersections(edges):
    """ Find points involved in T-intersections.

    A t-intersection is defined as a point involved in three fracture segments,
    of which two belong to the same fracture.

    The fractures should have been split (cg.remove_edge_crossings) before
    calling this function.

    Parameters:
        edges (np.array, 3 x n): Fractures. First two rows give indices of
            start and endpoints. Last row gives index of fracture that the
            segment belongs to.

    Returns:
        abutments (np.ndarray, int): indices of points that are
            T-intersections
        primal_frac (np.ndarray, int): Index of edges that are split by a
            t-intersection
        sec_frac (np.ndarray, int): Index of edges that ends in a
            T-intersection
        other_point (np.ndarray, int): For the secondary fractures, the end
            that is not in the t-intersection.

    """
    frac_num = edges[-1]
    abutments, edges_of_abutments = _intersection_by_num_node(edges, 3)

    num_abut = abutments.size
    primal_frac = np.zeros(num_abut, dtype=np.int)
    sec_frac = np.zeros(num_abut, dtype=np.int)
    other_point = np.zeros(num_abut, dtype=np.int)
    for i, (pi, ei) in enumerate(zip(abutments, edges_of_abutments)):
        # Count number of occurences for each fracture associated with this
        # intersection.
        fi_all = frac_num[edges_of_abutments]
        fi, count = np.unique(fi_all, return_counts=True)
        assert fi.size == 2
        # Find the fracture number associated with main and abutting edge.
        if count[0] == 1:
            primal_frac[i] = fi[1]
            sec_frac[i] = fi[0]
        else:
            primal_frac[i] = fi[0]
            sec_frac[i] = fi[1]
        # Also find the other point of the abutting edge
        ind = np.where(fi_all == sec_frac[i])[1]
        ei_abut = ei[ind]
        assert ei_abut.size == 1
        if edges[0, ei_abut] == pi:
            other_point[i] = edges[1, ei_abut]
        else:
            other_point[i] = edges[0, ei_abut]

    return abutments, primal_frac, sec_frac, other_point


def x_intersections(edges):
    """ Obtain nodes and edges involved in an x-intersection

    A x-intersection is defined as a point involved in four fracture segments,
    with two pairs belonging to two fractures each.

    The fractures should have been split (cg.remove_edge_crossings) before
    calling this function.

    Parameters:
        edges (np.array, 3 x n): Fractures. First two rows give indices of
            start and endpoints. Last row gives index of fracture that the
            segment belongs to.
    Returns:
        nodes: Index of nodes that form x intersections
        x_fracs (2xn): Index of fractures crossing in the nodes
        x_edges (4xn): Index of edges crossing in the nodes

    """
    frac_num = edges[-1]
    nodes, x_edges = _intersection_by_num_node(edges, 4)

    # Convert from edges (split fractures) to fractures themselves.
    num_x = nodes.size
    x_fracs = np.zeros((2, num_x))
    for i, ei in enumerate(frac_num[x_fracs]):
        x_fracs[:, i] = np.unique(ei)
    return nodes, x_fracs, x_edges

def fracture_length(pt, e):
    """ Compute length of fracture lines.

    Parameters:
        pt (np.array, 2xnpt): Coordinates of fracture endpoints
        e (np.array, 2xn_frac): Index of fracture endpoints.

    Returns:
        np.array, n_frac: Length of fractures.

    """
    x0 = pt[0, e[0]]
    x1 = pt[0, e[1]]
    y0 = pt[1, e[0]]
    y1 = pt[1, e[1]]

    return np.sqrt(np.power(x1-x0, 2) + np.power(y1-y0, 2))

def _disc_radius_center(lengths, p0, p1):
    """ Compute radius and center of a disc, based on the length of a chord
    through the disc, and assumptions on the location of the chord.

    For the moment, it is assumed that the chord is struck at an arbitrary
    hight of the circle.  Also, we assume that the chord is horizontal. In the
    context of an exposed fracture, this implies that the exposure is
    horizontal (I believe), and that an arbitrary portion of the original
    (disc-shaped) fracture has been eroded.

    Parameters:
        length (np.array, double, size: num_frac): Of the chords
        p0 (np.array, 2 x num_frac): One endpoint of fractures.
        p1 (np.array, 2 x num_frac): Second endpoint of fractures

    Returns:
        np.array, num_frac: Radius of discs
        np.array, 3 x num_frac: Center of the discs (assuming vertical disc)

    """

    num_frac = lengths.size

    # Angle between a vertical line through the disc center and the line
    # through the disc center and one of the fracture endpoints. Assumed to be
    # uniform, for the lack of more information.
    # Restrict the angle in the interval (0.1, 0.9) * pi to avoid excessively
    # large fractures.
    theta = np.pi * (0.1 + 0.8 * np.random.rand(num_frac))

    radius = 0.5 * lengths / np.sin(theta)

    # Midpoint in (x, y)-coordinate given as midpoint of exposed line
    mid_point = 0.5 * (p0 + p1)
    # z-coordinate from angle
    depth = radius * np.cos(theta)

    return radius, np.vstack((mid_point, depth))

def discs_from_exposure(pt, edges):
    """ Create fracture discs based on exposed lines in an outrcrop.

    The outcrop is assumed to be in the xy-plane.

    The returned disc will be vertical, and the points on the outcrop will be
    included in the polygon representation. The disc center is calculated using
    disc_radius_center(), see that function for assumptions.

    Parameters:
        pt (np.array, 2 x num_pts): Coordinates of exposed points.
        edges (np.array, 2 x num_fracs): Connections between fractures.

    Returns:
        list of Fracture: One per fracture trace.

    """

    num_fracs = edges.shape[1]

    lengths = fracture_length(pt, edges)
    p0 = pt[:, edges[0]]
    p1 = pt[:, edges[1]]

    v = p1 - p0
    strike_angle = np.arctan2(v[1], v[0])

    radius, center = _disc_radius_center(lengths, p0, p1)

    fracs = []

    for i in range(num_fracs):
        # The simplest way of distributing points along the disc seems to be to
        # create an elliptic fracture, and pick out the points. 
        f = EllipticFracture(center=center[:, i], major_axis=radius[i],
                             minor_axis=radius[i], dip_angle=np.pi/2,
                             strike_angle=strike_angle[i], major_axis_angle=0)

        # Add the points on the exposed surface. This creates an unequal
        # distribution of the points, but it is the only hard information we
        # have on the fracture
        f.add_points(np.vstack((np.hstack((p0[:, i], 0)),
                                np.hstack((p1[:, i], 0)))).T)
        fracs.append(Fracture(f.p))

    return fracs


def impose_inlcine(fracs, exposure, family, family_mean, family_std):
    """ Impose incline on the fractures from family-based parameters.

    The incline for each family is specified in terms of its mean and standard
    deviation. A normal distribution in assumed. The rotation is taken around
    the line of exposure, thus the resulting fractures are consistent with the
    outcrop.

    Parameters:
        fracs (list of Frature): Fractures to be inclined.
        exposure (np.array, 3xnum_frac): Exposed line for each fracture
            (visible in outcrop). Rotation will be around this line.
        family (np.array, num_fracs): For each fracture, which family does it
            belong to.
        family_mean (np.array, num_family): Mean value of incline for each
            family. In radians.
        family_std (np.array, num_family): Standard deviation of incine for
            each family. In radians.

    """
    def rotate_fracture(frac, vec, angle):
        # Helper function to carry out rotation.
        rot = cg.rot(angle, vec)
        p = frac.p
        center = np.mean(p, axis=1).reshape((-1, 1))
        frac.p = center + rot.dot(p - center)
        frac.points_2_ccw()

    exposure = np.vstack((exposure, np.zeros(len(fracs))))
    for fi, f in enumerate(fracs):
        fam = family[fi]
        ang = np.random.normal(loc=family_mean[fam], scale=family_std[fam])
        rotate_fracture(f, exposure[:, fi], ang)


def cut_fracture_by_plane(main_frac, other_frac, reference_point, tol=1e-4):
    """ Cut a fracture by a plane, and confine it to one side of the plane.

    Intended use is to confine abutting fractures (T-intersections) to one side
    of the fracture it hits.

    Parameters:
        main_frac (Fracture): The fracture to be cut.
        other_frac (Fracture): The fracture that defines the confining plane.
        reference_point (np.array, nd): Point on the main frature that defines
            which side should be kept. Will typically be the other point of the
            exposed line.

    Returns:
        Fracture: A copy of the main fracture, cut by the other fracture.

    Raises:
        ValueError if the points in the other fracture is too close. This could
        probably be handled by a scaling of coordinates, it is tacitly assumed
        that we're working in something resembling the unit box.

    """
    reference_point = reference_point.reshape((-1, 1))
    if reference_point.size == 2:
        reference_point = np.vstack((reference_point, 0))

    # First determine extent of the main fracture
    main_min = main_frac.p.min(axis=1)
    main_max = main_frac.p.max(axis=1)

    # Equation for the plane through the other fracture, on the form
    #  n_x(x-c_x) + n_y(y-c_y) + n_z(z-c_z) = 0
    n = cg.compute_normal(other_frac.p).reshape((-1, 1))
    c = other_frac.center

    # max and min coordinates that extends outside the main fracture
    main_min -= 1
    main_max += 1

    # Define points in the plane of the second fracture with min and max
    # coordinates picked from the main fracture.
    # The below tricks with indices are needed to find a dimension with a
    # non-zero gradient of the plane, so that we can divide safely.
    # Implementation note: It might have been possible to do this with a
    # rotation to the natural plane of the other fracture, but it is not clear
    # this will really be simpler.

    # Not sure about the tolerance here
    # We should perhaps do a scaling of coordinates.
    non_zero = np.where(np.abs(n) > 1e-8)[0]
    if non_zero.size == 0:
        raise ValueError('Could not compute normal vector of other fracture')
    ind = np.setdiff1d(np.arange(3), non_zero[0])
    i0 = ind[0]
    i1 = ind[1]
    i2 = non_zero[0]

    p = np.zeros((3, 4))
    # A for-loop might have been possible here.
    p[i0, 0] = main_min[i0]
    p[i1, 0] = main_min[i1]
    p[i2, 0] = c[i2] - (n[i0] * (main_min[i0] - c[i0])
                      + n[i1] * (main_min[i1] - c[i1])) / n[i2]

    p[i0, 1] = main_max[i0]
    p[i1, 1] = main_min[i1]
    p[i2, 1] = c[i2] - (n[i0] * (main_max[i0] - c[i0])
                      + n[i1] * (main_min[i1] - c[i1])) / n[i2]

    p[i0, 2] = main_max[i0]
    p[i1, 2] = main_max[i1]
    p[i2, 2] = c[i2] - (n[i0] * (main_max[i0] - c[i0])
                      + n[i1] * (main_max[i1] - c[i1])) / n[i2]

    p[i0, 3] = main_min[i0]
    p[i1, 3] = main_max[i1]
    p[i2, 3] = c[i2] - (n[i0] * (main_min[i0] - c[i0])
                      + n[i1] * (main_max[i1] - c[i1])) / n[i2]

    # Create an auxiliary fracture that spans the same plane as the other
    # fracture, and with a larger extension than the main fracture.
    aux_frac = Fracture(p)

    isect_pt, _, _ = main_frac.intersects(aux_frac, tol)

    # Next step is to eliminate points in the main fracture that are on the
    # wrong side of the other fracture.
    v = main_frac.p - other_frac.center.reshape((-1, 1))
    sgn = np.sign(np.sum(v * n, axis=0))
    ref_v = reference_point - other_frac.center.reshape((-1, 1))
    right_sign = np.sign(np.sum(ref_v * n, axis=0))

    # Eliminate points that are on the other side.
    eliminate = np.where(sgn * right_sign < 0)[0]
    main_frac.remove_points(eliminate)

    # Add intersection points on the main fracture. One of these may already be
    # present, as the point of extrusion, but add_point will uniquify the point
    # cloud.
    # We add the points after elimination, to ensure that the points on the
    # plane are present in the final fracture.
    main_frac.add_points(isect_pt)

    return main_frac
