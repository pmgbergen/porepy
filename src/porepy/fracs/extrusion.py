""" Simple functionality for extrusion of fractures (2D->3D). The main entry
point is the function fractures_from_outcrop(), this wraps the other functions.

The extrusion is carried out to be consistent with a set of 2D lines, e.g. an
outcrop. Fracture discs are constructed from the exposed lines. In case of
intersections in the exposed plane, the fractures will either cross
(if X-intersection), or the the abutting relation will be preserved (if
Y/T-intersection). If the extruded discs intersect outside the plane of
exposure, this will (almost surely, no checks are actually made) become a
X-intersection.

The fractures can be assigned a dip from the vertical, again taken as
consistent with the exposed line.

No attempts are made to create fractures that do not cross the confined plane.

For more information, see the tutorial on fracture extrusion.

KNOWN ISSUES:
    * When the extrusion is applied with rotations relative to outcrop plane,
      two fractures may meet in a point only. A warning is issued in this case.
      The consequences in terms of meshing are hard to predict. To fix this, it
      is likely necessary to constrain rotation angles to the extrusion angles
      in T-intersections.

"""
import numpy as np
import warnings
import logging

from porepy.utils import comp_geom as cg
from porepy.fracs.fractures import EllipticFracture, Fracture


logger = logging.getLogger()


def fractures_from_outcrop(
    pt, edges, ensure_realistic_cuts=True, family=None, extrusion_type="disc", **kwargs
):
    """ Create a set of fractures compatible with exposed lines in an outcrop.

    See module-level documentation for futher comments.

    Parameters:
        pt (np.array, 2 x num_pts): Coordinates of start and endpoints of
            extruded lines.
        edges (np.array, 2 x num_pts): Connections between points. Should not
            have their crossings removed before.
        ensure_realistic_cut (boolean, defaults to True): If True, we ensure
            that T-intersections do not have cut fractures that extend beyond
            the confining fracture. May overrule user-supplied controls on
            fracture sizes.
        **kwargs: Potentially user defined options. Forwarded to
            discs_from_exposure() and impose_inclines()

    Returns:
        list of Fracture: Fracture planes.

    """
    logging.info("Extrusion recieved " + str(edges.shape[1]) + " lines")
    assert edges.shape[0] == 2, "Edges have two endpoints"
    edges = np.vstack((edges, np.arange(edges.shape[1], dtype=np.int)))

    # identify crossings
    logging.info("Identify crossings")
    split_pt, split_edges = cg.remove_edge_crossings(pt, edges, **kwargs)
    logging.info("Fractures composed of " + str(split_edges.shape[0]) + "branches")

    # Find t-intersections
    abutment_pts, prim_frac, sec_frac, other_pt = t_intersections(split_edges)
    logging.info("Found " + str(prim_frac.size) + " T-intersections")

    # Calculate fracture lengths
    lengths = fracture_length(pt, edges)

    # Extrude to fracture discs
    logging.info("Create fractures from exposure")
    if extrusion_type.lower().strip() == "disc":
        fractures, extrude_ang = discs_from_exposure(pt, edges, **kwargs)
        disc_type = True
    else:
        fractures = rectangles_from_exposure(pt, edges)
        disc_type = False

    p0 = pt[:, edges[0]]
    p1 = pt[:, edges[1]]
    exposure = p1 - p0

    # Impose incline.
    logging.info("Impose incline")
    rot_ang = impose_inlcine(fractures, exposure, p0, frac_family=family, **kwargs)

    # Cut fractures
    for prim, sec, p in zip(prim_frac, sec_frac, other_pt):
        _, radius = cut_fracture_by_plane(
            fractures[sec], fractures[prim], split_pt[:, p], **kwargs
        )
        # If specified, ensure that cuts in T-intersections appear realistic.
        if ensure_realistic_cuts and disc_type and radius is not None:
            ang = np.arctan2(0.5 * lengths[prim], radius)

            # Ensure the center of both fractures are on the same side of the
            # exposed plane - if not, the cut will still be bad.
            if extrude_ang[sec] > np.pi / 2 and ang < np.pi / 2:
                ang = np.pi - ang
            elif extrude_ang[sec] < np.pi / 2 and ang > np.pi / 2:
                ang = np.pi - ang

            e0 = p0[:, prim]
            e1 = p1[:, prim]
            new_radius, center, _ = disc_radius_center(lengths[prim], e0, e1, theta=ang)
            strike = np.arctan2(e1[1] - e0[1], e1[0] - e0[0])
            f = create_disc_fracture(
                center, new_radius, np.pi / 2, strike, np.vstack((e0, e1)).T
            )
            rotate_fracture(f, e1 - e0, rot_ang[prim], p0[:, prim])
            fractures[prim] = f

    return fractures


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


def t_intersections(edges, remove_three_families=True):
    """ Find points involved in T-intersections.

    A t-intersection is defined as a point involved in three fracture segments,
    of which two belong to the same fracture.

    The fractures should have been split (cg.remove_edge_crossings) before
    calling this function.

    Parameters:
        edges (np.array, 3 x n): Fractures. First two rows give indices of
            start and endpoints. Last row gives index of fracture that the
            segment belongs to.
        remove_three_families (boolean, defaults to True): If True,
            T-intersections with where fractures from three different families
            meet will be removed.

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

    # Initialize fields for abutments.
    num_abut = abutments.size
    primal_frac = np.zeros(num_abut, dtype=np.int)
    sec_frac = np.zeros(num_abut, dtype=np.int)
    other_point = np.zeros(num_abut, dtype=np.int)

    # If fractures meeting in a T-intersection all have different family names,
    # these will be removed from the list.
    remove = np.zeros(num_abut, dtype=np.bool)

    for i, (pi, ei) in enumerate(zip(abutments, edges_of_abutments)):
        # Count number of occurences for each fracture associated with this
        # intersection.
        fi_all = frac_num[ei]
        fi, count = np.unique(fi_all, return_counts=True)
        if fi.size > 2:
            remove[i] = 1
            continue
        # Find the fracture number associated with main and abutting edge.
        if count[0] == 1:
            primal_frac[i] = fi[1]
            sec_frac[i] = fi[0]
        else:
            primal_frac[i] = fi[0]
            sec_frac[i] = fi[1]
        # Also find the other point of the abutting edge
        ind = np.where(fi_all == sec_frac[i])
        ei_abut = ei[ind]
        assert ei_abut.size == 1
        if edges[0, ei_abut] == pi:
            other_point[i] = edges[1, ei_abut]
        else:
            other_point[i] = edges[0, ei_abut]

    # Remove any T-intersections that did not belong to
    if remove_three_families and remove.any():
        remove = np.where(remove)[0]
        abutments = np.delete(abutments, remove)
        primal_frac = np.delete(primal_frac, remove)
        sec_frac = np.delete(sec_frac, remove)
        other_point = np.delete(other_point, remove)

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

    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))


def disc_radius_center(lengths, p0, p1, theta=None):
    """ Compute radius and center of a disc, based on the length of a chord
    through the disc, and assumptions on the location of the chord.

    The relation between the exposure and the center of the fracture is
    given by the theta, which gives the angle between a vertical line through
    the disc center and the line through the disc center and any of the
    exposure endpoints. If no values are given a random value is assigned,
    corresponding to an arbitrary portion of the original (disc-shaped)
    fracture has been eroded.

    Parameters:
        length (np.array, double, size: num_frac): Of the chords
        p0 (np.array, 2 x num_frac): One endpoint of fractures.
        p1 (np.array, 2 x num_frac): Second endpoint of fractures
        angle (np.array, num_frac, optional): Angle determining disc center,
            see description above. Defaults to random values.

    Returns:
        np.array, num_frac: Radius of discs
        np.array, 3 x num_frac: Center of the discs (assuming vertical disc)

    """

    num_frac = lengths.size

    # Restrict the angle in the interval (0.1, 0.9) * pi to avoid excessively
    # large fractures.
    if theta is None:
        rnd = np.random.rand(num_frac)
        # Angles of pi/2 will read to point contacts that cannot be handled
        # of the FractureNetwork. Point contacts also make little physical
        # sense, so we vaoid them.
        limit = 0.3
        hit = rnd > 1 - limit
        rnd[hit] -= limit
        hit = rnd < limit
        rnd[hit] += limit
        theta = np.pi * (limit + (1 - 2 * limit) * rnd)

    radius = 0.5 * lengths / np.sin(theta)

    # Midpoint in (x, y)-coordinate given as midpoint of exposed line
    mid_point = 0.5 * (p0 + p1).reshape((2, -1))

    # z-coordinate from angle
    depth = radius * np.cos(theta)

    return radius, np.vstack((mid_point, depth)), theta


def discs_from_exposure(
    pt, edges, exposure_angle=None, outcrop_consistent=True, **kwargs
):
    """ Create fracture discs based on exposed lines in an outrcrop.

    The outcrop is assumed to be in the xy-plane. The returned disc will be
    vertical, and the points on the outcrop will be included in the polygon
    representation.

    The location of the center is calculated from the angle, see
    disc_radius_center() for details.

    Parameters:
        pt (np.array, 2 x num_pts): Coordinates of exposed points.
        edges (np.array, 2 x num_fracs): Connections between fractures.
        exposure_angle (np.array of size num_fracs or double, optional):
            See above, and disc_radius_center() for description. Defaults to
            zero, which gives vertical fractures. Scalar input gives same angle
            to all fractures. Values very close to pi/2, 0 and pi will be
            modified to avoid unphysical extruded fractures.  If not provided,
            random values will be drawn. Measured in radians.
            Should be between 0 and pi.
        outcrop_consistent (boolean, optional): If True (default), points will
            be added at the outcrop surface z=0. This is necessary for the
            3D network to be consistent with the outcrop, but depending on
            the location of the points of the fracture polygon, it may result
            in very small edges.

    Returns:
        list of Fracture: One per fracture trace.

    """

    num_fracs = edges.shape[1]

    lengths = fracture_length(pt, edges)
    p0 = pt[:, edges[0]]
    p1 = pt[:, edges[1]]

    v = p1 - p0
    strike_angle = np.arctan2(v[1], v[0])

    if exposure_angle is not None:
        if isinstance(exposure_angle, np.ndarray) or isinstance(exposure_angle, list):
            exp_ang = np.asarray(exposure_angle)
        else:
            exp_ang = np.array(exposure_angle)

        # Angles of pi/2 will give point contacts
        hit = np.abs(exp_ang - np.pi / 2) < 0.01
        exp_ang[hit] = exp_ang[hit] + 0.01

        # Angles of 0 and pi give infinite fractures.
        hit = exp_ang < 0.2
        exp_ang[hit] = 0.2
        hit = np.pi - exp_ang < 0.2
        exp_ang[hit] = 0.2
    else:
        exp_ang = exposure_angle

    radius, center, ang = disc_radius_center(lengths, p0, p1, exp_ang)

    fracs = []

    for i in range(num_fracs):
        z = 2 * center[2, i]
        if outcrop_consistent:
            extra_point_depth = np.array([0, 0, z, z])
            extra_points = np.vstack(
                (
                    np.vstack((p0[:, i], p1[:, i], p0[:, i], p1[:, i])).T,
                    extra_point_depth,
                )
            )
        else:
            extra_points = np.zeros((3, 0))

        fracs.append(
            create_disc_fracture(
                center[:, i], radius[i], np.pi / 2, strike_angle[i], extra_points
            )
        )
    return fracs, ang


def rectangles_from_exposure(pt, edges, height=None, **kwargs):

    num_fracs = edges.shape[1]

    lengths = fracture_length(pt, edges)
    p0 = pt[:, edges[0]]
    p1 = pt[:, edges[1]]

    if height is None:
        height = lengths

    x0 = p0[0]
    x1 = p1[0]
    y0 = p0[1]
    y1 = p1[1]

    fracs = []

    for i in range(num_fracs):
        p = np.array(
            [
                [x0[i], y0[i], -height[i]],
                [x1[i], y1[i], -height[i]],
                [x1[i], y1[i], height[i]],
                [x0[i], y0[i], height[i]],
            ]
        ).T
        fracs.append(Fracture(p))
    return fracs


def create_disc_fracture(center, radius, dip, strike, extra_points):
    """ Create a single circular fracture consistent with a given exposure.

    The exposed points will be added to the fracture description.

    Parameters:
        center (np.array-like, dim 3): Center of the fracture.
        radius (double): Fracture radius.
        dip (double): dip angle of the fracture. See EllipticFracture for
            details.
        strike (np.array-like, dim 3): Strike angle for rotation. See
            EllipticFracture for details.
        extra_points (np.array, 3xnpt): Extra points to be added to the
            fracture. The points are assumed to lie on the ellipsis.

    Returns:
        Fracture: New fracture, according to the specifications.

    """
    if extra_points.shape[0] == 2:
        extra_points = np.vstack((extra_points, np.zeros(extra_points.shape[1])))

    # The simplest way of distributing points along the disc seems to be to
    # create an elliptic fracture, and pick out the points.
    f = EllipticFracture(
        center=center,
        major_axis=radius,
        minor_axis=radius,
        dip_angle=dip,
        strike_angle=strike,
        major_axis_angle=0,
    )
    # Add the points on the exposed surface. This creates an unequal
    # distribution of the points, but it is the only hard information we have
    # on the fracture
    f.add_points(extra_points, check_convexity=False, enforce_pt_tol=0.01)
    # Not sure if f still shoudl be EllipticFracture here, or if we should
    # create a new fracture with the same point distribution.
    return f


def rotate_fracture(frac, vec, angle, exposure):
    """ Rotate a fracture along a specified strike vector, and centered on a
    given point on the fracture surface.

    Modification of the fracture coordinates is done in place.

    TODO: Move this to the fracture itself?

    Parameters:
        frac (Fracture): To be rotated. Points are modified in-place.
        vec (np.array-like): Rotation will be around this vector.
        ang (double). Rotation angle. Measured in radians.
        exposure (np.array-like): Point on the strike vector, rotation will be
            centered around the line running through this point.

    """
    vec = np.asarray(vec)
    exposure = np.asarray(exposure)

    if vec.size == 2:
        vec = np.append(vec, 0)
    if exposure.size == 2:
        exposure = np.append(exposure, 0)
    exposure = exposure.reshape((3, 1))

    rot = cg.rot(angle, vec)
    p = frac.p
    frac.p = exposure + rot.dot(p - exposure)

    frac.points_2_ccw()
    frac.compute_centroid()
    frac.compute_normal()


def impose_inlcine(
    fracs,
    exposure_line,
    exposure_point,
    frac_family=None,
    family_mean_incline=None,
    family_std_incline=None,
    **kwargs
):
    """ Impose incline on the fractures from family-based parameters.

    The incline for each family is specified in terms of its mean and standard
    deviation. A normal distribution in assumed. The rotation is taken around
    the line of exposure, thus the resulting fractures are consistent with the
    outcrop.

    Parameters:
        fracs (list of Frature): Fractures to be inclined.
        exposure_line (np.array, 3xnum_frac): Exposed line for each fracture
            (visible in outcrop). Rotation will be around this line.
        exposure_point (np.array, 3xnum_frac): Point on exposure line. This
            point will not be rotated, it's a fixture.
        family (np.array, num_fracs): For each fracture, which family does it
            belong to. If not provided, all fractures are considered to belong
            to the same family.
        family_mean_incline (np.array, num_family): Mean value of incline for each
            family. In radians. Defaults to zero.
        family_std_incline (np.array, num_family): Standard deviation of incine for
            each family. In radians. Defaultst to zero.

        To set value for each fracture, set family = np.arange(len(fracs)),
        family_mean_incline=prescribed_value, and family_std_incline=None.

    Returns:
        np.array, size num_frac: Rotation angles.

    """
    if frac_family is None:
        frac_family = np.zeros(len(fracs), dtype=np.int)
    if family_mean_incline is None:
        family_mean_incline = np.zeros(np.unique(frac_family).size)
    if family_std_incline is None:
        family_std_incline = np.zeros(np.unique(frac_family).size)

    exposure_line = np.vstack((exposure_line, np.zeros(len(fracs))))
    all_ang = np.zeros(len(fracs))
    for fi, f in enumerate(fracs):
        fam = frac_family[fi]
        ang = np.random.normal(
            loc=family_mean_incline[fam], scale=family_std_incline[fam]
        )
        rotate_fracture(f, exposure_line[:, fi], ang, exposure_point[:, fi])
        all_ang[fi] = ang

    return all_ang


def cut_fracture_by_plane(
    main_frac, other_frac, reference_point, tol=1e-4, recompute_center=True, **kwargs
):
    """ Cut a fracture by a plane, and confine it to one side of the plane.

    Intended use is to confine abutting fractures (T-intersections) to one side
    of the fracture it hits. This is done by deleting points on the abutting
    fracture.

    Parameters:
        main_frac (Fracture): The fracture to be cut.
        other_frac (Fracture): The fracture that defines the confining plane.
        reference_point (np.array, nd): Point on the main frature that defines
            which side should be kept. Will typically be the other point of the
            exposed line.

    Returns:
        Fracture: A copy of the main fracture, cut by the other fracture.
        double: In cases where one interseciton point extends beyond the other
            fracture, this is the distance between the center of the other
            fracture and the intersection point. If both intersections are
            within the polygon, None will be returned.

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
        raise ValueError("Could not compute normal vector of other fracture")
    ind = np.setdiff1d(np.arange(3), non_zero[0])
    i0 = ind[0]
    i1 = ind[1]
    i2 = non_zero[0]

    p = np.zeros((3, 4))
    # A for-loop might have been possible here.
    p[i0, 0] = main_min[i0]
    p[i1, 0] = main_min[i1]
    p[i2, 0] = (
        c[i2]
        - (n[i0] * (main_min[i0] - c[i0]) + n[i1] * (main_min[i1] - c[i1])) / n[i2]
    )

    p[i0, 1] = main_max[i0]
    p[i1, 1] = main_min[i1]
    p[i2, 1] = (
        c[i2]
        - (n[i0] * (main_max[i0] - c[i0]) + n[i1] * (main_min[i1] - c[i1])) / n[i2]
    )

    p[i0, 2] = main_max[i0]
    p[i1, 2] = main_max[i1]
    p[i2, 2] = (
        c[i2]
        - (n[i0] * (main_max[i0] - c[i0]) + n[i1] * (main_max[i1] - c[i1])) / n[i2]
    )

    p[i0, 3] = main_min[i0]
    p[i1, 3] = main_max[i1]
    p[i2, 3] = (
        c[i2]
        - (n[i0] * (main_min[i0] - c[i0]) + n[i1] * (main_max[i1] - c[i1])) / n[i2]
    )

    # Create an auxiliary fracture that spans the same plane as the other
    # fracture, and with a larger extension than the main fracture.
    aux_frac = Fracture(p, check_convexity=False)

    isect_pt, _, _ = main_frac.intersects(aux_frac, tol)

    # The extension of the abutting fracture will cross the other fracture
    # with a certain angle to the vertical. If the other fracture is rotated
    # with a similar angle, point contact results.
    if isect_pt.size == 0:
        warnings.warn(
            """No intersection found in cutting of fractures. This is
                         likely caused by an unfortunate combination of
                         extrusion and rotation angles, which created fractures
                         that only intersect in a single point (the outcrop
                         plane. Will try to continue, but this may cause
                         trouble for meshing etc."""
        )
        return main_frac, None

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
    main_frac.add_points(isect_pt, check_convexity=False)

    if recompute_center:
        main_frac.compute_centroid()

    # If the main fracture is too large compared to the other, the cut line
    # will extend beyond the confining plane. In these cases, compute the
    # distance from the fracture center to the outside intersection point. This
    # can be used to extend the other fracture so as to avoid such strange
    # configurations.
    other_center = other_frac.center.reshape((-1, 1))
    other_p = other_frac.p
    rot = cg.project_plane_matrix(other_p - other_center)

    other_rot = rot.dot(other_p - other_center)[:2]
    isect_rot = rot.dot(isect_pt - other_center)[:2]

    is_inside = cg.is_inside_polygon(other_rot, isect_rot, tol, default=True)
    # At one point (the exposed point) must be in the polygon of the other
    # fracture.
    assert is_inside.any()

    if not is_inside.all():
        hit = np.logical_not(is_inside)
        r = np.sqrt(np.sum(isect_pt[:, hit] ** 2))
        return main_frac, r
    else:
        return main_frac, None
