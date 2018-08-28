"""
A module for representation and manipulations of fractures and fracture sets.

The model relies heavily on functions in the computational geometry library.

Known issues:
The subdomain partitioning may lead to addition of unwanted intersection points,
see FractureNetwork.on_domain_boundary.

"""
# Import of 'standard' external packages
import warnings
import time
import logging
import numpy as np
import sympy
import csv

# Imports of external packages that may not be present at the system. The
# module will work without any of these, but with limited functionalbility.
try:
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    warnings.warn(
        "VTK module is not available. Export of fracture network to\
    vtk will not work."
    )

# Import of internally developed packages.
from porepy.utils import comp_geom as cg
from porepy.utils import setmembership, sort_points
from porepy.grids.gmsh.gmsh_interface import GmshWriter
from porepy.grids.constants import GmshConstants


# Module-wide logger
logger = logging.getLogger(__name__)


class Fracture(object):
    """ Class representing a single fracture, as a convex, planar 2D object
    embedded in 3D space.

    The fracture is defined by its vertexes. It contains various utility
    methods, mainly intended for use together with the FractureNetwork class.

    Attributes:
        p (np.ndarray, 3 x npt): Fracture vertices. Will be stored in a CCW
            order (or CW, depending on which side it is viewed from).
        orig_p (np.ndarray, 3 x npt): Original fracture vertices, kept in case
            the fracture geometry for some reason is modified.
        center (np.ndarray, 3 x 1): Fracture centroid.
        index (int): Index of fracture. Intended used in FractureNetork.
            Exact use is not clear (several fractures can be given same index),
            use with care.

    """

    def __init__(self, points, index=None, check_convexity=True):
        """ Initialize fractures.

        __init__ defines the vertexes of the fracture and sort them to be CCW.
        Also compute centroid and normal, and check for planarity and
        convexity.

        Parameters:
            points (np.ndarray-like, 3 x npt): Vertexes of new fracture. There
                should be at least 3 points to define a plane.
            index (int, optional): Index of fracture. Defaults to None.
            check_convexity (boolean, optional): If True, check if the given
                points are convex. Defaults to True. Can be skipped if the
                plane is known to be convex to save time.

        """
        self.p = np.asarray(points, dtype=np.float)
        # Ensure the points are ccw
        self.points_2_ccw()
        self.compute_centroid()
        self.compute_normal()

        self.orig_p = self.p.copy()

        self.index = index

        assert self.is_planar(), "Points define non-planar fracture"
        if check_convexity:
            assert self.check_convexity(), "Points form non-convex polygon"

    def set_index(self, i):
        """ Set index of this fracture.

        Parameters:
            i (int): Index.

        """
        self.index = i

    def __eq__(self, other):
        """ Equality is defined as two fractures having the same index.

        There are issues with this, behavior may change in the future.

        """
        return self.index == other.index

    def copy(self):
        """ Return a deep copy of the fracture.

        Note that the original points (as given when the fracture was
        initialized) will *not* be preserved.

        Returns:
            Fracture with the same points.

        """
        p = np.copy(self.p)
        return Fracture(p)

    def points(self):
        """
        Iterator over the vexrtexes of the bounding polygon

        Yields:
            np.array (3 x 1): polygon vertexes

        """
        for i in range(self.p.shape[1]):
            yield self.p[:, i].reshape((-1, 1))

    def segments(self):
        """
        Iterator over the segments of the bounding polygon.

        Yields:
            np.array (3 x 2): polygon segment
        """

        sz = self.p.shape[1]
        for i in range(sz):
            yield self.p[:, np.array([i, i + 1]) % sz]

    def is_vertex(self, p, tol=1e-4):
        """ Check whether a given point is a vertex of the fracture.

        Parameters:
            p (np.array): Point to check
            tol (double): Tolerance of point accuracy.

        Returns:
            True: if the point is in the vertex set, false if not.
            int: Index of the identical vertex. None if not a vertex.

        """
        p = p.reshape((-1, 1))
        ap = np.hstack((p, self.p))
        up, _, ind = setmembership.unique_columns_tol(ap, tol=tol * np.sqrt(3))

        # If uniquifying did not remove any points, it is not a vertex.
        if up.shape[1] == ap.shape[1]:
            return False, None
        else:
            occurences = np.where(ind == ind[0])[0]
            return True, (occurences[1] - 1)

    def points_2_ccw(self):
        """
        Ensure that the points are sorted in a counter-clockwise order.

        implementation note:
            For now, the ordering of nodes in based on a simple angle argument.
            This will not be robust for general point clouds, but we expect the
            fractures to be regularly shaped in this sense. In particular, we
            will be safe if the cell is convex.

        Returns:
            np.array (int): The indices corresponding to the sorting.

        """
        # First rotate coordinates to the plane
        points_2d = self.plane_coordinates()
        # Center around the 2d origin
        points_2d -= np.mean(points_2d, axis=1).reshape((-1, 1))

        theta = np.arctan2(points_2d[1], points_2d[0])
        sort_ind = np.argsort(theta)

        self.p = self.p[:, sort_ind]

        return sort_ind

    def add_points(self, p, check_convexity=True, tol=1e-4, enforce_pt_tol=None):
        """
        Add a point to the polygon with ccw sorting enforced.

        Always run a test to check that the points are still planar. By
        default, a check of convexity is also performed, however, this can be
        turned off to speed up simulations (the test uses sympy, which turns
        out to be slow in many cases).

        Parameters:
            p (np.ndarray, 3xn): Points to add
            check_convexity (boolean, optional): Verify that the polygon is
                convex. Defaults to true.
            tol (double, optional): Tolerance used to check if the point
                already exists. Defaults to 1e-4.

        Return:
            boolean, true if the resulting polygon is convex.

        """

        to_enforce = np.hstack(
            (
                np.zeros(self.p.shape[1], dtype=np.bool),
                np.ones(p.shape[1], dtype=np.bool),
            )
        )
        self.p = np.hstack((self.p, p))
        self.p, _, _ = setmembership.unique_columns_tol(self.p, tol=tol)

        # Sort points to ccw
        mask = self.points_2_ccw()

        if enforce_pt_tol is not None:
            to_enforce = np.where(to_enforce[mask])[0]
            dist = cg.dist_pointset(self.p)
            dist /= np.amax(dist)
            np.fill_diagonal(dist, np.inf)
            mask = np.where(dist < enforce_pt_tol)[0]
            mask = np.setdiff1d(mask, to_enforce, assume_unique=True)
            self.remove_points(mask)

        if check_convexity:
            return self.check_convexity() and self.is_planar(tol)
        else:
            return self.is_planar()

    def remove_points(self, ind, keep_orig=False):
        """ Remove points from the fracture definition

        Parameters:
            ind (np array-like): Indices of points to remove.
            keep_orig (boolean, optional): Whether to keep the original points
                in the attribute orig_p. Defaults to False.

        """
        self.p = np.delete(self.p, ind, axis=1)
        if not keep_orig:
            self.orig_p = self.p

    def plane_coordinates(self):
        """
        Represent the vertex coordinates in its natural 2d plane.

        The plane does not necessarily have the third coordinate as zero (no
        translation to the origin is made)

        Returns:
            np.array (2xn): The 2d coordinates of the vertexes.

        """
        rotation = cg.project_plane_matrix(self.p)
        points_2d = rotation.dot(self.p)

        return points_2d[:2]

    def check_convexity(self):
        """
        Check if the polygon is convex.

        Todo: If a hanging node is inserted at a segment, this may slightly
            violate convexity due to rounding errors. It should be possible to
            write an algorithm that accounts for this. First idea: Projcet
            point onto line between points before and after, if the projection
            is less than a tolerance, it is okay.

        Returns:
            boolean, true if the polygon is convex.

        """
        p_2d = self.plane_coordinates()
        return self.as_sp_polygon(p_2d).is_convex()

    def is_planar(self, tol=1e-4):
        """ Check if the points forming this fracture lies in a plane.

        Parameters:
            tol (double): Tolerance for non-planarity. Treated as an absolute
                quantity (no scaling with fracture extent)

        Returns:
            boolean, True if the polygon is planar. False if not.
        """
        p = self.p - np.mean(self.p, axis=1).reshape((-1, 1))
        rot = cg.project_plane_matrix(p)
        p_2d = rot.dot(p)
        return np.max(np.abs(p_2d[2])) < tol

    def compute_centroid(self):
        """
        Compute, and redefine, center of the fracture in the form of the
        centroid.

        The method assumes the polygon is convex.

        """
        # Rotate to 2d coordinates
        rot = cg.project_plane_matrix(self.p)
        p = rot.dot(self.p)
        z = p[2, 0]
        p = p[:2]

        # Vectors from the first point to all other points. Subsequent pairs of
        # these will span triangles which, assuming convexity, will cover the
        # polygon.
        v = p[:, 1:] - p[:, 0].reshape((-1, 1))
        # The cell center of the triangles spanned by the subsequent vectors
        cc = (p[:, 0].reshape((-1, 1)) + p[:, 1:-1] + p[:, 2:]) / 3
        # Area of triangles
        area = 0.5 * np.abs(v[0, :-1] * v[1, 1:] - v[1, :-1] * v[0, 1:])

        # The center is found as the area weighted center
        center = np.sum(cc * area, axis=1) / np.sum(area)

        # Project back again.
        self.center = rot.transpose().dot(np.append(center, z)).reshape((3, 1))

    def compute_normal(self):
        """ Compute normal to the polygon.
        """
        self.normal = cg.compute_normal(self.p)[:, None]

    def as_sp_polygon(self, p=None):
        """ Represent polygon as a sympy object.

        Parameters:
            p (np.array, nd x npt, optional): Points for the polygon. Defaults
                to None, in which case self.p is used.

        Returns:
            sympy.geometry.Polygon: Representation of the polygon formed by p.

        """
        if p is None:
            p = self.p

        sp = [sympy.geometry.Point(p[:, i]) for i in range(p.shape[1])]
        return sympy.geometry.Polygon(*sp)

    def intersects(self, other, tol, check_point_contact=True):
        """
        Find intersections between self and another polygon.

        The function checks for both intersection between the interior of one
        polygon with the boundary of the other, and pure boundary
        intersections. Intersection types supported so far are
            X (full intersection, possibly involving boundaries)
            L (Two fractures intersect at a boundary)
            T (Boundary segment of one fracture lies partly or completely in
                the plane of another)
        Parameters:
            other (Fracture): To test intersection with.
            tol (double): Geometric tolerance
            check_point_contact (boolean, optional): If True, we check if a
                single point (e.g. a vertex) of one fracture lies in the plane
                of the other. This is usually an error, thus the test should be
                on, but may be allowed when imposing external boundaries.

        """

        # Find intersections between self and other. Note that the algorithms
        # for intersections are not fully reflexive in terms of argument
        # order, so we need to do two tests.

        # Array for intersections with the interior of one polygon (as opposed
        # to full boundary intersection, below)
        int_points = np.empty((3, 0))

        # Keep track of whether the intersection points are on the boundary of
        # the polygons.
        on_boundary_self = False
        on_boundary_other = False

        ####
        # First compare max/min coordinates. If the bounding boxes of the
        # fractures do not intersect, there is nothing to do.
        min_self = self.p.min(axis=1)
        max_self = self.p.max(axis=1)
        min_other = other.p.min(axis=1)
        max_other = other.p.max(axis=1)

        # Account for a tolerance in the following test
        max_self *= 1 + np.sign(max_self) * tol
        min_self *= 1 - np.sign(min_self) * tol

        max_other *= 1 + np.sign(max_other) * tol
        min_other *= 1 - np.sign(min_other) * tol

        if np.any(max_self < min_other) or np.any(min_self > max_other):
            return int_points, on_boundary_self, on_boundary_other

        #####
        # Next screening: To intersect, both fractures must have vertexes
        # either on both sides of each others plane, or close to the plane (T,
        # L, Y)-type intersections.

        # Vectors from centers of the fractures to the vertexes of the other
        # fratures.
        s_2_o = other.p - self.center.reshape((-1, 1))
        o_2_s = self.p - other.center.reshape((-1, 1))

        # Take the dot product of distance and normal vectors. Different signs
        # for different vertexes signifies fractures that potentially cross.
        other_from_self = np.sum(s_2_o * self.normal, axis=0)
        self_from_other = np.sum(o_2_s * other.normal, axis=0)

        # To avoid ruling out vertexes that lie on the plane of another
        # fracture, we introduce a threshold for almost-zero values.
        # The scaling factor is somewhat arbitrary here, and probably
        # safeguards too much, but better safe than sorry. False positives will
        # be corrected by the more accurate, but costly, computations below.
        scaled_tol = tol * max(1, max(np.max(np.abs(s_2_o)), np.max(np.abs(o_2_s))))

        # We can terminate only if no vertexes are close to the plane of the
        # other fractures.
        if (
            np.min(np.abs(other_from_self)) > scaled_tol
            and np.min(np.abs(self_from_other)) > scaled_tol
        ):
            # If one of the fractures has all of its points on a single side of
            # the other, there can be no intersections.
            if (
                np.all(np.sign(other_from_self) == 1)
                or np.all(np.sign(other_from_self) == -1)
                or np.all(np.sign(self_from_other) == 1)
                or np.all(np.sign(self_from_other) == -1)
            ):
                return int_points, on_boundary_self, on_boundary_other

        ####
        # Check for intersection between interior of one polygon with
        # segment of the other.

        # Compute intersections, with both polygons as first argument
        isect_self_other = cg.polygon_segment_intersect(
            self.p, other.p, tol=tol, include_bound_pt=True
        )
        isect_other_self = cg.polygon_segment_intersect(
            other.p, self.p, tol=tol, include_bound_pt=True
        )

        # Process data
        if isect_self_other is not None:
            int_points = np.hstack((int_points, isect_self_other))

            # An intersection between self and other (in that order) is defined
            # as being between interior of self and boundary of other. See
            # polygon_segment_intersect for details.
            on_boundary_self = False
            on_boundary_other = False

        if isect_other_self is not None:
            int_points = np.hstack((int_points, isect_other_self))

            # Interior of other intersected by boundary of self
            on_boundary_self = False
            on_boundary_other = False

        if int_points.shape[1] > 1:
            int_points, _, _ = setmembership.unique_columns_tol(int_points, tol=tol)

        # There should be at most two of these points.
        # In some cases, likely involving extrusion, several segments may lay
        # essentially in the fracture plane, producing more than two segments.
        # Thus, if more than two colinear points are found, pick out the first
        # and last one.
        if int_points.shape[1] > 2:
            if cg.is_collinear(int_points, tol):
                sort_ind = cg.argsort_point_on_line(int_points, tol)
                int_points = int_points[:, [sort_ind[0], sort_ind[-1]]]
            else:
                # This is a bug
                raise ValueError(
                    """ Found more than two intersection between
                                 fracture polygons.
                                 """
                )

        ####
        # Next, check for intersections between the polygon boundaries
        bound_sect_self_other = cg.polygon_boundaries_intersect(
            self.p, other.p, tol=tol
        )
        bound_sect_other_self = cg.polygon_boundaries_intersect(
            other.p, self.p, tol=tol
        )

        def point_on_segment(ip, poly, need_two=True):
            # Check if a set of points are located on a single segment
            if need_two and ip.shape[1] < 2 or ((not need_two) and ip.shape[1] < 1):
                return False
            start = poly
            end = np.roll(poly, 1, axis=1)
            for si in range(start.shape[1]):
                dist, cp = cg.dist_points_segments(ip, start[:, si], end[:, si])
                if np.all(dist < tol):
                    return True
            return False

        # Short cut: If no boundary intersections, we return the interior
        # points
        if len(bound_sect_self_other) == 0 and len(bound_sect_other_self) == 0:
            if check_point_contact and int_points.shape[1] == 1:
                #  contacts are not implemented. Give a warning, return no
                # interseciton, and hope the meshing software is merciful
                if hasattr(self, "index") and hasattr(other, "index"):
                    logger.warning(
                        """Found a point contact between fracture
                                   %i
                                   and %i at (%.5f, %.5f, %.5f)""",
                        self.index,
                        other.index,
                        *int_points
                    )
                else:
                    logger.warning(
                        """Found point contact between fractures
                                   with no index. Coordinate: (%.5f, %.5f,
                                   %.5f)""",
                        *int_points
                    )
                return np.empty((3, 0)), on_boundary_self, on_boundary_other
            # None of the intersection points lay on the boundary

        # The 'interior' points can still be on the boundary (naming
        # of variables should be updated). The points form a boundary
        # segment if they all lie on the a single segment of the
        # fracture.
        on_boundary_self = point_on_segment(int_points, self.p)
        on_boundary_other = point_on_segment(int_points, other.p)
        return int_points, on_boundary_self, on_boundary_other

    def impose_boundary(self, box, tol):
        """
        Impose a boundary on a fracture, defined by a bounding box.

        If the fracture extends outside the box, it will be truncated, and new
        points are inserted on the intersection with the boundary. It is
        assumed that the points defining the fracture defines a convex set (in
        its natural plane) to begin with.

        The attribute self.p will be changed if intersections are found. The
        original vertexes can still be recovered from self.orig_p.

        The box is specified by its extension in Cartesian coordinates.

        Parameters:
            box (dicitionary): The bounding box, specified by keywords xmin,
                xmax, ymin, ymax, zmin, zmax.
            tol (double): Tolerance, defines when two points are considered
                equal.

        """

        # Maximal extent of the fracture
        min_coord = self.p.min(axis=1)
        max_coord = self.p.max(axis=1)

        # Coordinates of the bounding box
        x0_box = box["xmin"]
        x1_box = box["xmax"]
        y0_box = box["ymin"]
        y1_box = box["ymax"]
        z0_box = box["zmin"]
        z1_box = box["zmax"]

        # Gather the box coordinates in an array
        box_array = np.array([[x0_box, x1_box], [y0_box, y1_box], [z0_box, z1_box]])

        # We need to be a bit careful if the fracture extends outside the
        # bounding box on two (non-oposite) sides. In addition to the insertion
        # of points along the segments defined by self.p, this will also
        # require the insertion of points at the meeting of the two sides. To
        # capture this, we first look for the intersection between the fracture
        # and planes that extends further than the plane, and then later
        # move the intersection points to lay at the real bounding box
        x0 = np.minimum(x0_box, min_coord[0] - 10 * tol)
        x1 = np.maximum(x1_box, max_coord[0] + 10 * tol)
        y0 = np.minimum(y0_box, min_coord[1] - 10 * tol)
        y1 = np.maximum(y1_box, max_coord[1] + 10 * tol)
        z0 = np.minimum(z0_box, min_coord[2] - 10 * tol)
        z1 = np.maximum(z1_box, max_coord[2] + 10 * tol)

        def outside_box(p, bound_i):
            # Helper function to test if points are outside the bounding box
            relative = np.amax(np.linalg.norm(p, axis=0))
            p = cg.snap_to_grid(p, tol=tol / relative)
            # snap_to_grid will impose a grid of size self.tol, thus points
            # that are more than half that distance away from the boundary
            # are deemed outside.
            # To reduce if-else on the boundary index, we compute all bounds,
            # and then return the relevant one, as specified by bound_i
            west_of = p[0] < x0_box - tol / 2
            east_of = p[0] > x1_box + tol / 2
            south_of = p[1] < y0_box - tol / 2
            north_of = p[1] > y1_box + tol / 2
            beneath = p[2] < z0_box - tol / 2
            above = p[2] > z1_box + tol / 2
            outside = np.vstack((west_of, east_of, south_of, north_of, beneath, above))
            return outside[bound_i]

        # Represent the planes of the bounding box as fractures, to allow
        # us to use the fracture intersection methods.
        # For each plane, we keep the fixed coordinate to the value specified
        # by the box, and extend the two others to cover segments that run
        # through the extension of the plane only.

        # Move these to FractureNetwork.impose_external_boundary.
        # Doesn't immediately work. Depend on the fracture.
        west = Fracture(
            np.array(
                [[x0_box, x0_box, x0_box, x0_box], [y0, y1, y1, y0], [z0, z0, z1, z1]]
            ),
            check_convexity=False,
        )
        east = Fracture(
            np.array(
                [[x1_box, x1_box, x1_box, x1_box], [y0, y1, y1, y0], [z0, z0, z1, z1]]
            ),
            check_convexity=False,
        )
        south = Fracture(
            np.array(
                [[x0, x1, x1, x0], [y0_box, y0_box, y0_box, y0_box], [z0, z0, z1, z1]]
            ),
            check_convexity=False,
        )
        north = Fracture(
            np.array(
                [[x0, x1, x1, x0], [y1_box, y1_box, y1_box, y1_box], [z0, z0, z1, z1]]
            ),
            check_convexity=False,
        )
        bottom = Fracture(
            np.array(
                [[x0, x1, x1, x0], [y0, y0, y1, y1], [z0_box, z0_box, z0_box, z0_box]]
            ),
            check_convexity=False,
        )
        top = Fracture(
            np.array(
                [[x0, x1, x1, x0], [y0, y0, y1, y1], [z1_box, z1_box, z1_box, z1_box]]
            ),
            check_convexity=False,
        )
        # Collect in a list to allow iteration
        bound_planes = [west, east, south, north, bottom, top]

        # Loop over all boundary sides and look for intersections
        for bi, bf in enumerate(bound_planes):
            # Dimensions that are not fixed by bf
            active_dims = np.ones(3, dtype=np.bool)
            active_dims[np.floor(bi / 2).astype("int")] = 0
            # Convert to indices
            active_dims = np.squeeze(np.argwhere(active_dims))

            # Find intersection points
            isect, _, _ = self.intersects(bf, tol, check_point_contact=False)
            num_isect = isect.shape[1]
            if len(isect) > 0 and num_isect > 0:
                num_pts_orig = self.p.shape[1]
                # Add extra points at the end of the point list.
                self.p, _, _ = setmembership.unique_columns_tol(
                    np.hstack((self.p, isect))
                )
                num_isect = self.p.shape[1] - num_pts_orig
                if num_isect == 0:
                    continue

                # Sort fracture points in a ccw order.
                # This must be done before sliding the intersection points
                # (below), since the latter may break convexity of the
                # fracture, thus violate the (current) implementation of
                # points_2_ccw()
                sort_ind = self.points_2_ccw()

                # The next step is to decide whether and how to move the
                # intersection points.
                # The intersection points will lay on the bounding box on one
                # of the dimensions (that of bf, specified by xyz_01_box), but
                # may be outside the box for the other sides (xyz_01). We thus
                # need to move the intersection points in the plane of bf and
                # the fracture plane.
                # The relevant tangent vector is perpendicular to both the
                # fracture and bf
                normal_self = cg.compute_normal(self.p)
                normal_bf = cg.compute_normal(bf.p)
                tangent = np.cross(normal_self, normal_bf)
                # Unit vector
                tangent *= 1. / np.linalg.norm(tangent)

                isect_ind = np.argwhere(sort_ind >= num_pts_orig).ravel("F")
                p_isect = self.p[:, isect_ind]

                # Loop over all intersection points and active dimensions. If
                # the point is outside the bounding box, move it.
                for pi in range(num_isect):
                    for dim, other_dim in zip(active_dims, active_dims[::-1]):
                        # Test against lower boundary
                        lower_diff = p_isect[dim, pi] - box_array[dim, 0]
                        # Modify coordinates if necessary. This dimension is
                        # simply set to the boundary, while the other should
                        # slide along the tangent vector
                        if lower_diff < 0:
                            p_isect[dim, pi] = box_array[dim, 0]
                            # We know lower_diff != 0, no division by zero.
                            # We may need some tolerances, though.
                            t = tangent[dim] / lower_diff
                            p_isect[other_dim, pi] += t * tangent[other_dim]

                        # Same treatment of the upper boundary
                        upper_diff = p_isect[dim, pi] - box_array[dim, 1]
                        # Modify coordinates if necessary. This dimension is
                        # simply set to the boundary, while the other should
                        # slide along the tangent vector
                        if upper_diff > 0:
                            p_isect[dim, pi] = box_array[dim, 1]
                            t = tangent[dim] / upper_diff
                            p_isect[other_dim, pi] -= t * tangent[other_dim]

                # Finally, identify points that are outside the face bf
                inside = np.logical_not(outside_box(self.p, bi))
                # Dump points that are outside.
                self.p = self.p[:, inside]
                self.p, _, _ = setmembership.unique_columns_tol(self.p, tol=tol)
                # We have modified the fractures, so re-calculate the centroid
                self.compute_centroid()
            else:
                # No points exists, but the whole point set can still be
                # outside the relevant boundary face.
                outside = outside_box(self.p, bi)
                if np.all(outside):
                    self.p = np.empty((3, 0))
            # There should be at least three points in a fracture.
            if self.p.shape[1] < 3:
                break

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "Points: \n"
        s += str(self.p) + "\n"
        s += "Center: " + str(self.center)
        return s


# --------------------------------------------------------------------


class EllipticFracture(Fracture):
    """
    Subclass of Fractures, representing an elliptic fracture.

    See parent class for description.

    """

    def __init__(
        self,
        center,
        major_axis,
        minor_axis,
        major_axis_angle,
        strike_angle,
        dip_angle,
        num_points=16,
    ):
        """
        Initialize an elliptic shaped fracture, approximated by a polygon.


        The rotation of the plane is calculated using three angles. First, the
        rotation of the major axis from the x-axis. Next, the fracture is
        inclined by specifying the strike angle (which gives the rotation
        axis) measured from the x-axis, and the dip angle. All angles are
        measured in radians.

        Parameters:
            center (np.ndarray, size 3x1): Center coordinates of fracture.
            major_axis (double): Length of major axis (radius-like, not
                diameter).
            minor_axis (double): Length of minor axis. There are no checks on
                whether the minor axis is less or equal the major.
            major_axis_angle (double, radians): Rotation of the major axis from
                the x-axis. Measured before strike-dip rotation, see above.
            strike_angle (double, radians): Line of rotation for the dip.
                Given as angle from the x-direction.
            dip_angle (double, radians): Dip angle, i.e. rotation around the
                strike direction.
            num_points (int, optional): Number of points used to approximate
                the ellipsis. Defaults to 16.

        Example:
            Fracture centered at [0, 1, 0], with a ratio of lengths of 2,
            rotation in xy-plane of 45 degrees, and an incline of 30 degrees
            rotated around the x-axis.
            >>> frac = EllipticFracture(np.array([0, 1, 0]), 10, 5, np.pi/4, 0,
                                        np.pi/6)

        """
        center = np.asarray(center)
        if center.ndim == 1:
            center = center.reshape((-1, 1))
        self.center = center

        # First, populate polygon in the xy-plane
        angs = np.linspace(0, 2 * np.pi, num_points + 1, endpoint=True)[:-1]
        x = major_axis * np.cos(angs)
        y = minor_axis * np.sin(angs)
        z = np.zeros_like(angs)
        ref_pts = np.vstack((x, y, z))

        assert cg.is_planar(ref_pts)

        # Rotate reference points so that the major axis has the right
        # orientation
        major_axis_rot = cg.rot(major_axis_angle, [0, 0, 1])
        rot_ref_pts = major_axis_rot.dot(ref_pts)

        assert cg.is_planar(rot_ref_pts)

        # Then the dip
        # Rotation matrix of the strike angle
        strike_rot = cg.rot(strike_angle, np.array([0, 0, 1]))
        # Compute strike direction
        strike_dir = strike_rot.dot(np.array([1, 0, 0]))
        dip_rot = cg.rot(dip_angle, strike_dir)

        dip_pts = dip_rot.dot(rot_ref_pts)

        assert cg.is_planar(dip_pts)

        # Set the points, and store them in a backup.
        self.p = center + dip_pts
        self.orig_p = self.p.copy()

        # Compute normal vector
        self.normal = cg.compute_normal(self.p)[:, None]

        assert cg.is_planar(self.orig_p, self.normal)


# -------------------------------------------------------------------------


class Intersection(object):
    """ Class representing the intersection between two fractures.

    Fractures are identified by their indexes, which is not robust. Apply with
    care.

    Intersections may be created even for non-intersecting fractures. These are
    identified by an empty coordinate set. This behavior is unfortunate, but
    caused by legacy design choices.

    Attributes:
        first (int): Index of first fracture.
        second (int): Index of second fracture.
        coord (np.array, 3xn_pt): End coordinates of intersection line. Should
            contain up to two points. No points signifies this is an empty
            intersection.
        bound_first (boolean): Whether the intersection is on the boundary of
            the first fracture.
        bound_second (boolean): Whether the intersection is on the boundary of
            the second fracture.

    """

    def __init__(self, first, second, coord, bound_first=False, bound_second=False):
        """ Initialize Intersection object.

        Parameters:
            first (int): Index of first fracture in intersection.
            second (ind): Index of second fracture in intersection.
            coord (np.arary, 3xn_pt): Index of intersection points. May have
                from 0 to 2 points.
            bound_first (boolean, optional). Is intersection on boundary of
                first fracture? Defaults to false.
            bound_second (boolean, optional). Is intersection on boundary of
                second fracture? Defaults to false.

        """
        self.first = first
        self.second = second
        self.coord = coord
        # Information on whether the intersection points lies on the boundaries
        # of the fractures
        self.bound_first = bound_first
        self.bound_second = bound_second

    def __repr__(self):
        s = (
            "Intersection between fractures "
            + str(self.first.index)
            + " and "
            + str(self.second.index)
            + "\n"
        )
        s += "Intersection points: \n"
        for i in range(self.coord.shape[1]):
            s += (
                "("
                + str(self.coord[0, i])
                + ", "
                + str(self.coord[1, i])
                + ", "
                + str(self.coord[2, i])
                + ") \n"
            )
        if self.coord.size > 0:
            s += "On boundary of first fracture " + str(self.bound_first) + "\n"
            s += "On boundary of second fracture " + str(self.bound_second)
            s += "\n"
        return s

    def get_other_fracture(self, i):
        """ Get the other based on index.

        Parameters:
            i (int): Index of a fracture of this intersection.

        Returns:
            int: Index of the other fracture.

        Raises:
            ValueError if fracture with index i does not belong to
                intersection.

        """

        if self.first == i:
            return self.second
        elif self.second == i:
            return self.first
        else:
            raise ValueError("Fracture " + str(i) + " is not in intersection")

    def on_boundary_of_fracture(self, i):
        """ Check if the intersection is on the boundary of a fracture.

        Parameters:
            i (int): Index of a fracture of this intersection.

        Returns:
            boolean: True if intersection is on boundary of this intersection.

        Raises:
            ValueError if fracture with index i does not belong to
                intersection.
        """
        if self.first == i:
            return self.bound_first
        elif self.second == i:
            return self.bound_second
        else:
            raise ValueError("Fracture " + str(i) + " is not in intersection")


# ----------------------------------------------------------------------------


class FractureNetwork(object):
    """
    Collection of Fractures with geometrical information. Facilitates
    computation of intersections of the fracture. Also incorporates the
    bounding box as six boundary fractures and the possibility to impose
    additional gridding constraints through subdomain boundaries. To ensure
    that all fractures lie within the box, call impose_external_boundary()
    _after_ all fractures and subdomains have been specified.

    Attributes:
        _fractures (list of Fracture): All fractures forming the network.
        intersections (list of Intersection): All known intersections in the
            network.
        has_checked_intersections (boolean): If True, the intersection finder
            method has been run. Useful in meshing algorithms to avoid
            recomputing known information.
        verbose (int): Verbosity level. Usage unclear.
        tol (double): Geometric tolerance used in computations.
        domain (dictionary): External bounding box. See
            impose_external_boundary() for details.
        tags (dictionary): Tags used on Fractures and subdomain boundaries.
        mesh_size_min (double): Mesh size parameter, minimum mesh size to be
            sent to gmsh. Set by insert_auxiliary_points().
        mesh_size_frac (double): Mesh size parameter. Ideal mesh size, fed to
            gmsh. Set by insert_auxiliary_points().
        mesh_size_bound (double): Mesh size parameter. Boundary mesh size, fed to
            gmsh. Set by insert_auxiliary_points().
        auxiliary_points_added (boolean): Mesh size parameter. If True,
            extra points have been added to Fracture geometry to facilitate
            mesh size tuning.
        decomposition (dictionary): Splitting of network, accounting for
            fracture intersections etc. Necessary pre-processing before
            meshing. Added by split_intersections().

    """

    def __init__(self, fractures=None, verbose=0, tol=1e-4):
        """ Initialize fracture network.

        Generate network from specified fractures. The fractures will have
        their index set (and existing values overridden), according to the
        ordering in the input list.

        Initialization sets most fields (see attributes) to None.

        Parameters:
            fractures (list of Fracture): Fractures that make up the network.
            verbose (int, optional): Verbosity level. Defaults to 0.
            tol (double, optional): Geometric tolerance. Defaults to 1e-4.

        """

        self._fractures = fractures

        for i, f in enumerate(self._fractures):
            f.set_index(i)

        self.intersections = []

        self.has_checked_intersections = False
        self.tol = tol
        self.verbose = verbose

        # Initialize with an empty domain. Can be modified later by a call to
        # 'impose_external_boundary()'
        self.domain = None

        # Initialize mesh size parameters as empty
        self.mesh_size_min = None
        self.mesh_size_frac = None
        self.mesh_size_bound = None
        # Assign an empty tag dictionary
        self.tags = {}

        # No auxiliary points have been added
        self.auxiliary_points_added = False

        self.bounding_box_imposed = False

    def add(self, f):
        """ Add a fracture to the network.

        The fracture will be assigned a new index, higher than the maximum
        value currently found in the network.

        Parameters:
            f (Fracture): Fracture to be added.

        """
        ind = np.array([f.index for f in self._fractures])

        if ind.size > 0:
            f.set_index(np.max(ind) + 1)
        else:
            f.set_index(0)
        self._fractures.append(f)

    def __getitem__(self, position):
        return self._fractures[position]

    def intersections_of_fracture(self, frac):
        """ Get all known intersections for a fracture.

        If called before find_intersections(), the returned list will be empty.

        Paremeters:
            frac: A fracture in the network

        Returns:
            np.array (Intersection): Array of intersections

        """
        if isinstance(frac, int):
            fi = frac
        else:
            fi = frac.index
        frac_arr = []
        for i in self.intersections:
            if i.coord.size == 0:
                continue
            if i.first.index == fi or i.second.index == fi:
                frac_arr.append(i)
        return frac_arr

    def find_intersections(self, use_orig_points=False):
        """
        Find intersections between fractures in terms of coordinates.

        The intersections are stored in the attribute self.Intersections.

        Handling of the intersections (splitting into non-intersecting
        polygons, paving the way for gridding) is taken care of by the function
        split_intersections().

        Note that find_intersections() should be invoked after external
        boundaries are imposed. If the reverse order is applied, intersections
        outside the domain may be identified, with unknown consequences for the
        reliability of the methods. If intersections outside the bounding box
        are of interest, these can be found by setting the parameter
        use_orig_points to True.

        Parameters:
            use_orig_points (boolean, optional): Whether to use the original
                fracture description in the search for intersections. Defaults
                to False. If True, all fractures will have their attribute p
                reset to their original value.

        """
        self.has_checked_intersections = True
        logger.info("Find intersection between fratures")
        start_time = time.time()

        # If desired, use the original points in the fracture intersection.
        # This will reset the field self._fractures.p, and thus revoke
        # modifications due to boundaries etc.
        if use_orig_points:
            for f in self._fractures:
                f.p = f.orig_p

        for i, first in enumerate(self._fractures):
            for j in range(i + 1, len(self._fractures)):
                second = self._fractures[j]
                logger.debug("Processing fracture %i and %i", i, j)
                isect, bound_first, bound_second = first.intersects(second, self.tol)
                if np.array(isect).size > 0:
                    logger.debug("Found an intersection between %i and %i", i, j)
                    # Let the intersection know whether both intersection
                    # points lies on the boundary of each fracture
                    self.intersections.append(
                        Intersection(
                            first,
                            second,
                            isect,
                            bound_first=bound_first,
                            bound_second=bound_second,
                        )
                    )

        logger.info(
            "Found %i intersections. Ellapsed time: %.5f",
            len(self.intersections),
            time.time() - start_time,
        )

    def intersection_info(self, frac_num=None):
        """ Obtain information on intersections of one or several fractures.

        Specifically, the non-empty intersections are given for each fracture,
        together with aggregated numbers.

        Parameters:
            frac_num (int or np.array, optional): Fractures to be considered.
                Defaults to all fractures in network.

        Returns:
            str: Information on fracture intersections.

        """
        # Number of fractures with some intersection
        num_intersecting_fracs = 0
        # Number of intersections in total
        num_intersections = 0

        if frac_num is None:
            frac_num = np.arange(len(self._fractures))

        s = ""

        for f in np.atleast_1d(np.asarray(frac_num)):
            isects = []
            for i in self.intersections:
                if i.first.index == f and i.coord.shape[1] > 0:
                    isects.append(i.second.index)
                elif i.second.index == f and i.coord.shape[1] > 0:
                    isects.append(i.first.index)
            if len(isects) > 0:
                num_intersecting_fracs += 1
                num_intersections += len(isects)

                if self.verbose > 1:
                    s += (
                        "Fracture " + str(f) + " intersects with"
                        " fractuer(s) " + str(isects) + "\n"
                    )
        # Print aggregate numbers. Note that all intersections are counted
        # twice (from first and second), thus divide by two.
        s += (
            "In total "
            + str(num_intersecting_fracs)
            + " fractures "
            + "intersect in "
            + str(int(num_intersections / 2))
            + " intersections"
        )
        return s

    def split_intersections(self):
        """
        Based on the fracture network, and their known intersections, decompose
        the fractures into non-intersecting sub-polygons. These can
        subsequently be exported to gmsh.

        The method will add an atribute decomposition to self.

        """

        logger.info("Split intersections")
        start_time = time.time()

        # First, collate all points and edges used to describe fracture
        # boundaries and intersections.
        all_p, edges, edges_2_frac, is_boundary_edge = self._point_and_edge_lists()

        if self.verbose > 1:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        # By now, all segments in the grid are defined by a unique set of
        # points and edges. The next task is to identify intersecting edges,
        # and split them.
        all_p, edges, edges_2_frac, is_boundary_edge = self._remove_edge_intersections(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

        if self.verbose > 1:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        # Store the full decomposition.
        self.decomposition = {
            "points": all_p,
            "edges": edges.astype("int"),
            "is_bound": is_boundary_edge,
            "edges_2_frac": edges_2_frac,
        }
        polygons = []
        line_in_frac = []
        for fi, _ in enumerate(self._fractures):
            ei = []
            ei_bound = []
            # Find the edges of this fracture, add to either internal or
            # external fracture list
            for i, e in enumerate(zip(edges_2_frac, is_boundary_edge)):
                # Check that the boundary information matches the fractures
                assert e[0].size == e[1].size
                hit = np.where(e[0] == fi)[0]
                if hit.size == 1:
                    if e[1][hit]:
                        ei_bound.append(i)
                    else:
                        ei.append(i)
                elif hit.size > 1:
                    raise ValueError("Non-unique fracture edge relation")
                else:
                    continue

            poly = sort_points.sort_point_pairs(edges[:2, ei_bound])
            polygons.append(poly)
            line_in_frac.append(ei)

        self.decomposition["polygons"] = polygons
        self.decomposition["line_in_frac"] = line_in_frac

        #                              'polygons': poly_segments,
        #                             'polygon_frac': poly_2_frac}

        logger.info(
            "Finished fracture splitting after %.5f seconds", time.time() - start_time
        )

    def _fracs_2_edges(self, edges_2_frac):
        """ Invert the mapping between edges and fractures.

        Returns:
            List of list: For each fracture, index of all edges that points to
                it.
        """
        f2e = []
        for fi in len(self._fractures):
            f_l = []
            for ei, e in enumerate(edges_2_frac):
                if fi in e:
                    f_l.append(ei)
            f2e.append(f_l)
        return f2e

    def _point_and_edge_lists(self):
        """
        Obtain lists of all points and connections necessary to describe
        fractures and their intersections.

        Returns:
            np.ndarray, 3xn: Unique coordinates of all points used to describe
            the fracture polygons, and their intersections.

            np.ndarray, 2xn_edge: Connections between points, formed either
                by a fracture boundary, or a fracture intersection.
            list: For each edge, index of all fractures that point to the
                edge.
            np.ndarray of bool (size=num_edges): A flag telling whether the
                edge is on the boundary of a fracture.

        """
        logger.info("Compile list of points and edges")
        start_time = time.time()

        # Field for all points in the fracture description
        all_p = np.empty((3, 0))
        # All edges, either as fracture boundary, or fracture intersection
        edges = np.empty((2, 0))
        # For each edge, a list of all fractures pointing to the edge.
        edges_2_frac = []

        # Field to know if an edge is on the boundary of a fracture.
        # Not sure what to do with a T-type intersection here
        is_boundary_edge = []

        # First loop over all fractures. All edges are assumed to be new; we
        # will deal with coinciding points later.
        for fi, frac in enumerate(self._fractures):
            num_p = all_p.shape[1]
            num_p_loc = frac.p.shape[1]
            all_p = np.hstack((all_p, frac.p))

            loc_e = num_p + np.vstack(
                (np.arange(num_p_loc), (np.arange(num_p_loc) + 1) % num_p_loc)
            )
            edges = np.hstack((edges, loc_e))
            for i in range(num_p_loc):
                edges_2_frac.append([fi])
                is_boundary_edge.append([True])

        # Next, loop over all intersections, and define new points and edges
        for i in self.intersections:
            # Only add information if the intersection exists, that is, it has
            # a coordinate.
            if i.coord.size > 0:
                num_p = all_p.shape[1]
                all_p = np.hstack((all_p, i.coord))

                edges = np.hstack((edges, num_p + np.arange(2).reshape((-1, 1))))
                edges_2_frac.append([i.first.index, i.second.index])
                # If the intersection points are on the boundary of both
                # fractures, this is a boundary segment.
                # This does not cover the case of a T-intersection, that will
                # have to come later.
                if i.bound_first and i.bound_second:
                    is_boundary_edge.append([True, True])
                elif i.bound_first and not i.bound_second:
                    is_boundary_edge.append([True, False])
                elif not i.bound_first and i.bound_second:
                    is_boundary_edge.append([False, True])
                else:
                    is_boundary_edge.append([False, False])

        # Ensure that edges are integers
        edges = edges.astype("int")

        logger.info(
            "Points and edges done. Elapsed time %.5f", time.time() - start_time
        )

        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

    def _uniquify_points_and_edges(self, all_p, edges, edges_2_frac, is_boundary_edge):
        # Snap the points to an underlying Cartesian grid. This is the basis
        # for declearing two points equal
        # NOTE: We need to account for dimensions in the tolerance;

        start_time = time.time()
        logger.info(
            """Uniquify points and edges, starting with %i points, %i
                    edges""",
            all_p.shape[1],
            edges.shape[1],
        )

        #        all_p = cg.snap_to_grid(all_p, tol=self.tol)

        # We now need to find points that occur in multiple places
        p_unique, unique_ind_p, all_2_unique_p = setmembership.unique_columns_tol(
            all_p, tol=self.tol * np.sqrt(3)
        )

        # Update edges to work with unique points
        edges = all_2_unique_p[edges]

        # Look for edges that share both nodes. These will be identical, and
        # will form either a L/Y-type intersection (shared boundary segment),
        # or a three fractures meeting in a line.
        # Do a sort of edges before looking for duplicates.
        e_unique, unique_ind_e, all_2_unique_e = setmembership.unique_columns_tol(
            np.sort(edges, axis=0)
        )

        # Update the edges_2_frac map to refer to the new edges
        edges_2_frac_new = e_unique.shape[1] * [np.empty(0)]
        is_boundary_edge_new = e_unique.shape[1] * [np.empty(0)]

        for old_i, new_i in enumerate(all_2_unique_e):
            edges_2_frac_new[new_i], ind = np.unique(
                np.hstack((edges_2_frac_new[new_i], edges_2_frac[old_i])),
                return_index=True,
            )
            tmp = np.hstack((is_boundary_edge_new[new_i], is_boundary_edge[old_i]))
            is_boundary_edge_new[new_i] = tmp[ind]

        edges_2_frac = edges_2_frac_new
        is_boundary_edge = is_boundary_edge_new

        # Represent edges by unique values
        edges = e_unique

        # The uniquification of points may lead to edges with identical start
        # and endpoint. Find and remove these.
        point_edges = np.where(np.squeeze(np.diff(edges, axis=0)) == 0)[0]
        edges = np.delete(edges, point_edges, axis=1)
        unique_ind_e = np.delete(unique_ind_e, point_edges)
        for ri in point_edges[::-1]:
            del edges_2_frac[ri]
            del is_boundary_edge[ri]

        # Ensure that the edge to fracture map to a list of numpy arrays.
        # Use unique so that the same edge only refers to an edge once.
        edges_2_frac = [
            np.unique(np.array(edges_2_frac[i])) for i in range(len(edges_2_frac))
        ]

        # Sanity check, the fractures should still be defined by points in a
        # plane.
        self._verify_fractures_in_plane(p_unique, edges, edges_2_frac)

        logger.info(
            """Uniquify complete. %i points, %i edges. Ellapsed time
                    %.5f""",
            p_unique.shape[1],
            edges.shape[1],
            time.time() - start_time,
        )

        return p_unique, edges, edges_2_frac, is_boundary_edge

    def _remove_edge_intersections(self, all_p, edges, edges_2_frac, is_boundary_edge):
        """
        Remove crossings from the set of fracture intersections.

        Intersecting intersections (yes) are split, and new points are
        introduced.

        Parameters:
            all_p (np.ndarray, 3xn): Coordinates of all points used to describe
                the fracture polygons, and their intersections. Should be
                unique.
            edges (np.ndarray, 2xn): Connections between points, formed either
                by a fracture boundary, or a fracture intersection.
            edges_2_frac (list): For each edge, index of all fractures that
                point to the edge.
            is_boundary_edge (np.ndarray of bool, size=num_edges): A flag
                telling whether the edge is on the boundary of a fracture.

        Returns:
            The same fields, but updated so that all edges are
            non-intersecting.

        """
        logger.info("Remove edge intersections")
        start_time = time.time()

        # The algorithm loops over all fractures, pulls out edges associated
        # with the fracture, project to the local 2D plane, and look for
        # intersections there (direct search in 3D may also work, but this was
        # a simple option). When intersections are found, the global lists of
        # points and edges are updated.
        for fi, frac in enumerate(self._fractures):

            logger.debug("Remove intersections from fracture %i", fi)

            # Identify the edges associated with this fracture
            # It would have been more convenient to use an inverse
            # relationship frac_2_edge here, but that would have made the
            # update for new edges (towards the end of this loop) more
            # cumbersome.
            edges_loc_ind = []
            for ei, e in enumerate(edges_2_frac):
                if np.any(e == fi):
                    edges_loc_ind.append(ei)

            edges_loc = np.vstack((edges[:, edges_loc_ind], np.array(edges_loc_ind)))
            p_ind_loc = np.unique(edges_loc[:2])
            p_loc = all_p[:, p_ind_loc]

            p_2d, edges_2d, p_loc_c, rot = self._points_2_plane(
                p_loc, edges_loc, p_ind_loc
            )

            # Add a tag to trace the edges during splitting
            edges_2d[2] = edges_loc[2]

            # Obtain new points and edges, so that no edges on this fracture
            # are intersecting.
            # It seems necessary to increase the tolerance here somewhat to
            # obtain a more robust algorithm. Not sure about how to do this
            # consistent.
            p_new, edges_new = cg.remove_edge_crossings(
                p_2d, edges_2d, tol=self.tol, verbose=self.verbose, snap=False
            )
            # Then, patch things up by converting new points to 3D,

            # From the design of the functions in cg, we know that new points
            # are attached to the end of the array. A more robust alternative
            # is to find unique points on a combined array of p_loc and p_new.
            p_add = p_new[:, p_ind_loc.size :]
            num_p_add = p_add.shape[1]

            # Add third coordinate, and map back to 3D
            p_add = np.vstack((p_add, np.zeros(num_p_add)))

            # Inverse of rotation matrix is the transpose, add cloud center
            # correction
            p_add_3d = rot.transpose().dot(p_add) + p_loc_c

            # The new points will be added at the end of the global point array
            # (if not, we would need to renumber all global edges).
            # Global index of added points
            ind_p_add = all_p.shape[1] + np.arange(num_p_add)
            # Global index of local points (new and added)
            p_ind_exp = np.hstack((p_ind_loc, ind_p_add))

            # Add the new points towards the end of the list.
            all_p = np.hstack((all_p, p_add_3d))

            new_all_p, _, ia = setmembership.unique_columns_tol(all_p, self.tol)

            # Handle case where the new point is already represented in the
            # global list of points.
            if new_all_p.shape[1] < all_p.shape[1]:
                all_p = new_all_p
                p_ind_exp = ia[p_ind_exp]

            # The ordering of the global edge list bears no significance. We
            # therefore plan to delete all edges (new and old), and add new
            # ones.

            # First add new edges.
            # All local edges in terms of global point indices
            edges_new_glob = p_ind_exp[edges_new[:2]]
            edges = np.hstack((edges, edges_new_glob))

            # Global indices of the local edges
            edges_loc_ind = np.unique(edges_loc_ind)

            # Append fields for edge-fracture map and boundary tags
            for ei in range(edges_new.shape[1]):
                # Find the global edge index. For most edges, this will be
                # correctly identified by edges_new[2], which tracks the
                # original edges under splitting. However, in cases of
                # overlapping segments, in which case the index of the one edge
                # may completely override the index of the other (this is
                # caused by the implementation of remove_edge_crossings).
                # We therefore compare the new edge to the old ones (before
                # splitting). If found, use the old information; if not, use
                # index as tracked by splitting.
                is_old, old_loc_ind = setmembership.ismember_rows(
                    edges_new_glob[:, ei].reshape((-1, 1)), edges[:2, edges_loc_ind]
                )
                if is_old[0]:
                    glob_ei = edges_loc_ind[old_loc_ind[0]]
                else:
                    glob_ei = edges_new[2, ei]
                # Update edge_2_frac and boundary information.
                edges_2_frac.append(edges_2_frac[glob_ei])
                is_boundary_edge.append(is_boundary_edge[glob_ei])

            # Finally, purge the old edges
            edges = np.delete(edges, edges_loc_ind, axis=1)

            # We cannot delete more than one list element at a time. Delete by
            # index in decreasing order, so that we do not disturb the index
            # map.
            edges_loc_ind.sort()
            for ei in edges_loc_ind[::-1]:
                del edges_2_frac[ei]
                del is_boundary_edge[ei]
            # And we are done with this fracture. On to the next one.

        logger.info(
            "Done with intersection removal. Elapsed time %.5f",
            time.time() - start_time,
        )
        self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

        # To
        # define the auxiliary edges, we create a triangulation of the
        # fractures. We then grow polygons forming part of the fracture in a
        # way that ensures that no polygon lies on both sides of an
        # intersection edge.

    def report_on_decomposition(self, do_print=True, verbose=None):
        """
        Compute various statistics on the decomposition.

        The coverage is rudimentary for now, will be expanded when needed.

        Parameters:
            do_print (boolean, optional): Print information. Defaults to True.
            verbose (int, optional): Override the verbosity level of the
                network itself. If not provided, the network value will be
                used.

        Returns:
            str: String representation of the statistics

        """
        if verbose is None:
            verbose = self.verbose
        d = self.decomposition

        s = str(len(self._fractures)) + " fractures are split into "
        s += str(len(d["polygons"])) + " polygons \n"

        s += "Number of points: " + str(d["points"].shape[1]) + "\n"
        s += "Number of edges: " + str(d["edges"].shape[1]) + "\n"

        if verbose > 1:
            # Compute minimum distance between points in point set
            dist = np.inf
            p = d["points"]
            num_points = p.shape[1]
            hit = np.ones(num_points, dtype=np.bool)
            for i in range(num_points):
                hit[i] = False
                dist_loc = cg.dist_point_pointset(p[:, i], p[:, hit])
                dist = np.minimum(dist, dist_loc.min())
                hit[i] = True
            s += "Minimal disance between points " + str(dist) + "\n"

        if do_print:
            logger.info(s)

        return s

    def fractures_of_points(self, pts):
        """
        For a given point, find all fractures that refer to it, either as
        vertex or as internal.

        Returns:
            list of np.int: indices of fractures, one list item per point.

        """
        fracs_of_points = []
        pts = np.atleast_1d(np.asarray(pts))
        for i in pts:
            fracs_loc = []

            # First identify edges that refers to the point
            edge_ind = np.argwhere(
                np.any(self.decomposition["edges"][:2] == i, axis=0)
            ).ravel("F")
            edges_loc = self.decomposition["edges"][:, edge_ind]
            # Loop over all polygons. If their edges are found in edges_loc,
            # store the corresponding fracture index
            for poly_ind, poly in enumerate(self.decomposition["polygons"]):
                ismem, _ = setmembership.ismember_rows(edges_loc, poly)
                if any(ismem):
                    fracs_loc.append(self.decomposition["polygon_frac"][poly_ind])
            fracs_of_points.append(list(np.unique(fracs_loc)))
        return fracs_of_points

    def close_points(self, dist):
        """
        In the set of points used to describe the fractures (after
        decomposition), find pairs that are closer than a certain distance.
        Inteded use: Debugging.

        Parameters:
            dist (double): Threshold distance, all points closer than this will
                be reported.

        Returns:
            List of tuples: Each tuple contain indices of a set of close
                points, and the distance between the points. The list is not
                symmetric; if (a, b) is a member, (b, a) will not be.

        """
        c_points = []

        pt = self.decomposition["points"]
        for pi in range(pt.shape[1]):
            d = cg.dist_point_pointset(pt[:, pi], pt[:, pi + 1 :])
            ind = np.argwhere(d < dist).ravel("F")
            for i in ind:
                # Indices of close points, with an offset to compensate for
                # slicing of the point cloud.
                c_points.append((pi, i + pi + 1, d[i]))

        return c_points

    def _verify_fractures_in_plane(self, p, edges, edges_2_frac):
        """
        Essentially a debugging method that verify that the given set of
        points, edges and edge connections indeed form planes.

        This has turned out to be a common symptom of trouble.

        """
        for fi, _ in enumerate(self._fractures):

            # Identify the edges associated with this fracture
            edges_loc_ind = []
            for ei, e in enumerate(edges_2_frac):
                if np.any(e == fi):
                    edges_loc_ind.append(ei)

            edges_loc = edges[:, edges_loc_ind]
            p_ind_loc = np.unique(edges_loc)
            p_loc = p[:, p_ind_loc]

            # Run through points_2_plane, to check the assertions
            self._points_2_plane(p_loc, edges_loc, p_ind_loc)

    def _points_2_plane(self, p_loc, edges_loc, p_ind_loc):
        """
        Convenience method for rotating a point cloud into its own 2d-plane.
        """

        # Center point cloud around the origin
        p_loc_c = np.mean(p_loc, axis=1).reshape((-1, 1))
        p_loc -= p_loc_c

        # Project the points onto the local plane defined by the fracture
        rot = cg.project_plane_matrix(p_loc, tol=self.tol)
        p_2d = rot.dot(p_loc)

        extent = p_2d.max(axis=1) - p_2d.min(axis=1)
        lateral_extent = np.maximum(np.max(extent[2]), 1)
        assert extent[2] < lateral_extent * self.tol * 30

        # Dump third coordinate
        p_2d = p_2d[:2]

        # The edges must also be redefined to account for the (implicit)
        # local numbering of points
        edges_2d = np.empty_like(edges_loc)
        for ei in range(edges_loc.shape[1]):
            edges_2d[0, ei] = np.argwhere(p_ind_loc == edges_loc[0, ei])
            edges_2d[1, ei] = np.argwhere(p_ind_loc == edges_loc[1, ei])

        assert edges_2d[:2].max() < p_loc.shape[1]

        return p_2d, edges_2d, p_loc_c, rot

    def change_tolerance(self, new_tol):
        """
        Redo the whole configuration based on the new tolerance
        """
        pass

    def __repr__(self):
        s = "Fracture set with " + str(len(self._fractures)) + " fractures"
        return s

    def bounding_box(self):
        """ Obtain bounding box for fracture network.

        The box is defined by the external boundary, if imposed, or if not by
        the maximal extent of the fractures in each direction.

        Returns:
            dictionary with fields 'xmin', 'xmax', 'ymin', 'ymax', 'zmin',
                'zmax'
        """
        min_coord = np.ones(3) * float("inf")
        max_coord = -np.ones(3) * float("inf")

        for f in self._fractures:
            min_coord = np.minimum(np.min(f.p, axis=1), min_coord)
            max_coord = np.maximum(np.max(f.p, axis=1), max_coord)

        return {
            "xmin": min_coord[0],
            "xmax": max_coord[0],
            "ymin": min_coord[1],
            "ymax": max_coord[1],
            "zmin": min_coord[2],
            "zmax": max_coord[2],
        }

    def add_subdomain_boundaries(self, vertexes):
        """
        Adds subdomain boundaries to the fracture network. These are
        intended to ensure the partitioning of the matrix along the planes
        they define.
        Parameters:
            vertexes: either a Fracture or a
            np.array([[x0, x1, x2, x3],
                      [y0, y1, y2, y3],
                      [z0, z1, z2, z3]])
        """
        subdomain_tags = self.tags.get("subdomain", [False] * len(self._fractures))
        for f in vertexes:
            if not hasattr(f, "p") or not isinstance(f.p, np.ndarray):
                # np.array to Fracture:
                f = Fracture(f)
            # Add the fake fracture and its tag to the network:
            self.add(f)
            subdomain_tags.append(True)

        self.tags["subdomain"] = subdomain_tags

    def impose_external_boundary(
        self, box=None, truncate_fractures=True, keep_box=True
    ):
        """
        Set an external boundary for the fracture set.

        The boundary takes the form of a 3D box, described by its minimum and
        maximum coordinates. If no bounding box is provided, a box will be
        fited outside the fracture network.

        If desired, the fratures will be truncated to lay within the bounding
        box; that is, Fracture.p will be modified. The orginal coordinates of
        the fracture boundary can still be recovered from the attribute
        Fracture.orig_points.

        Fractures that are completely outside the bounding box will be deleted
        from the fracture set.

        Parameters:
            box (dictionary): Has fields 'xmin', 'xmax', and similar for y and
                z.
            truncate_fractures (boolean, optional): If True, fractures outside
            the bounding box will be disregarded, while fractures crossing the
            boundary will be truncated.

        """
        self.bounding_box_imposed = True

        if box is None:
            OVERLAP = 0.15
            cmin = np.ones((3, 1)) * float("inf")
            cmax = -np.ones((3, 1)) * float("inf")
            for f in self._fractures:
                cmin = np.min(np.hstack((cmin, f.p)), axis=1).reshape((-1, 1))
                cmax = np.max(np.hstack((cmax, f.p)), axis=1).reshape((-1, 1))
            cmin = cmin[:, 0]
            cmax = cmax[:, 0]

            dx = OVERLAP * (cmax - cmin)

            # If the fractures has no extension along one of the coordinate
            # (a single fracture aligned with one axis), the domain should
            # still have an extension.
            hit = np.where(dx < self.tol)[0]
            dx[hit] = OVERLAP * np.max(dx)

            box = {
                "xmin": cmin[0] - dx[0],
                "xmax": cmax[0] + dx[0],
                "ymin": cmin[1] - dx[1],
                "ymax": cmax[1] + dx[1],
                "zmin": cmin[2] - dx[2],
                "zmax": cmax[2] + dx[2],
            }

        # Insert boundary in the form of a box, and kick out (parts of)
        # fractures outside the box
        self.domain = box

        # Create fractures of box here.
        # Store them self._fractures so that split_intersections work
        # keep track of which fractures are really boundaries - perhaps attribute is_proxy?

        if truncate_fractures:
            # Keep track of fractures that are completely outside the domain.
            # These will be deleted.
            delete_frac = []

            # Loop over all fractures, use method in fractures to truncate if
            # necessary.
            for i, frac in enumerate(self._fractures):
                frac.impose_boundary(box, self.tol)
                if frac.p.shape[1] == 0:
                    delete_frac.append(i)

            # Delete fractures that have all points outside the bounding box
            # There may be some uncovered cases here, with a fracture barely
            # touching the box from the outside, but we leave that for now.
            for i in np.unique(delete_frac)[::-1]:
                del self._fractures[i]

            # Final sanity check: All fractures should have at least three
            # points at the end of the manipulations
            for f in self._fractures:
                assert f.p.shape[1] >= 3

        self._make_bounding_planes(box, keep_box)

    def _make_bounding_planes(self, box, keep_box=True):
        """
        Translate the bounding box into fractures. Tag them as boundaries.
        For now limited to a box consisting of six planes.
        """
        x0 = box["xmin"]
        x1 = box["xmax"]
        y0 = box["ymin"]
        y1 = box["ymax"]
        z0 = box["zmin"]
        z1 = box["zmax"]
        west = Fracture(
            np.array([[x0, x0, x0, x0], [y0, y1, y1, y0], [z0, z0, z1, z1]]),
            check_convexity=False,
        )
        east = Fracture(
            np.array([[x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]]),
            check_convexity=False,
        )
        south = Fracture(
            np.array([[x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]]),
            check_convexity=False,
        )
        north = Fracture(
            np.array([[x0, x1, x1, x0], [y1, y1, y1, y1], [z0, z0, z1, z1]]),
            check_convexity=False,
        )
        bottom = Fracture(
            np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z0, z0, z0, z0]]),
            check_convexity=False,
        )
        top = Fracture(
            np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]]),
            check_convexity=False,
        )
        # Collect in a list to allow iteration
        bound_planes = [west, east, south, north, bottom, top]
        boundary_tags = self.tags.get("boundary", [False] * len(self._fractures))

        # Add the boundaries to the fracture network and tag them.
        if keep_box:
            for f in bound_planes:
                self.add(f)
                boundary_tags.append(True)
        self.tags["boundary"] = boundary_tags

    def _classify_edges(self, polygon_edges):
        """
        Classify the edges into fracture boundary, intersection, or auxiliary.
        Also identify points on intersections between interesctions (fractures
        of co-dimension 3)

        Parameters:
            polygon_edges (list of lists): For each polygon the global edge
                indices that forms the polygon boundary.

        Returns:
            tag: Tag of the fracture, using the values in GmshConstants. Note
                that auxiliary points will not be tagged (these are also
                ignored in gmsh_interface.GmshWriter).
            is_0d_grid: boolean, one for each point. True if the point is
                shared by two or more intersection lines.

        """
        edges = self.decomposition["edges"]
        is_bound = self.decomposition["is_bound"]
        num_edges = edges.shape[1]

        poly_2_frac = np.arange(len(self.decomposition["polygons"]))
        self.decomposition["polygon_frac"] = poly_2_frac

        # Construct a map from edges to polygons
        edge_2_poly = [[] for i in range(num_edges)]
        for pi, poly in enumerate(polygon_edges[0]):
            for ei in np.unique(poly):
                edge_2_poly[ei].append(poly_2_frac[pi])

        # Count the number of referals to the edge from polygons belonging to
        # different fractures (not polygons)
        num_referals = np.zeros(num_edges)
        for ei, ep in enumerate(edge_2_poly):
            num_referals[ei] = np.unique(np.array(ep)).size

        # A 1-d grid is inserted where there is more than one fracture
        # referring.
        has_1d_grid = np.where(num_referals > 1)[0]

        num_constraints = len(is_bound)
        constants = GmshConstants()
        tag = np.zeros(num_edges, dtype="int")

        # Find fractures that are tagged as a boundary
        all_bound = [np.all(is_bound[i]) for i in range(len(is_bound))]
        bound_ind = np.where(all_bound)[0]
        # Remove those that are referred to by more than fracture - this takes
        # care of L-type intersections
        bound_ind = np.setdiff1d(bound_ind, has_1d_grid)

        # Index of lines that should have a 1-d grid. This are all of the first
        # num-constraints, minus those on the boundary.
        # Note that edges with index > num_constraints are known to be of the
        # auxiliary type. These will have tag zero; and treated in a special
        # manner by the interface to gmsh.
        intersection_ind = np.setdiff1d(np.arange(num_constraints), bound_ind)
        tag[bound_ind] = constants.FRACTURE_TIP_TAG
        tag[intersection_ind] = constants.FRACTURE_INTERSECTION_LINE_TAG

        # Count the number of times a point is referred to by an intersection
        # between two fractures. If this is more than one, the point should
        # have a 0-d grid assigned to it.
        isect_p = edges[:, intersection_ind].ravel()
        num_occ_pt = np.bincount(isect_p)
        is_0d_grid = np.where(num_occ_pt > 1)[0]

        return tag, is_0d_grid

    def _poly_2_segment(self):
        """
        Represent the polygons by the global edges, and determine if the lines
        must be reversed (locally) for the polygon to form a closed loop.

        """
        edges = self.decomposition["edges"]
        poly = self.decomposition["polygons"]

        poly_2_line = []
        line_reverse = []
        for p in poly:
            hit, ind = setmembership.ismember_rows(p, edges[:2], sort=False)
            hit_reverse, ind_reverse = setmembership.ismember_rows(
                p[::-1], edges[:2], sort=False
            )
            assert np.all(hit + hit_reverse == 1)

            line_ind = np.zeros(p.shape[1])

            hit_ind = np.where(hit)[0]
            hit_reverse_ind = np.where(hit_reverse)[0]
            line_ind[hit_ind] = ind
            line_ind[hit_reverse_ind] = ind_reverse

            poly_2_line.append(line_ind.astype("int"))
            line_reverse.append(hit_reverse)

        return poly_2_line, line_reverse

    def _determine_mesh_size(self, **kwargs):
        """
        Set the preferred mesh size for geometrical points as specified by
        gmsh.

        Currently, the only option supported is to specify a single value for
        all fracture points, and one value for the boundary.

        See the gmsh manual for further details.

        """

        num_pts = self.decomposition["points"].shape[1]

        if self.mesh_size_min is None or self.mesh_size_frac is None:
            logger.info("Found no information on mesh sizes. Returning")
            return None

        p = self.decomposition["points"]
        num_pts = p.shape[1]
        dist = cg.dist_pointset(p, max_diag=True)
        mesh_size_dist = np.min(dist, axis=1)
        logger.info(
            "Minimal distance between points encountered is " + str(np.min(dist))
        )
        mesh_size_min = np.maximum(
            mesh_size_dist, self.mesh_size_min * np.ones(num_pts)
        )
        mesh_size = np.minimum(mesh_size_min, self.mesh_size_frac * np.ones(num_pts))
        on_boundary = kwargs.get("boundary_point_tags", None)
        if (self.mesh_size_bound is not None) and (on_boundary is not None):
            constants = GmshConstants()
            on_boundary = on_boundary == constants.DOMAIN_BOUNDARY_TAG
            mesh_size_bound = np.minimum(
                mesh_size_min, self.mesh_size_bound * np.ones(num_pts)
            )
            mesh_size[on_boundary] = np.maximum(mesh_size, mesh_size_bound)[on_boundary]
        return mesh_size

    def insert_auxiliary_points(
        self, mesh_size_frac=None, mesh_size_min=None, mesh_size_bound=None
    ):
        """ Insert auxiliary points on fracture edges. Used to guide gmsh mesh
        size parameters.

        The function should only be called once to avoid insertion of multiple
        layers of extra points, this will likely kill gmsh.

        The function is motivated by similar functionality for 2d domains, but
        is considerably less mature.

        The function should be used in conjunction with _determine_mesh_size(),
        called with mode='distance'. The ultimate goal is to set the mesh size
        for geometrical points in Gmsh. To that end, this function inserts
        additional points on the fracture boundaries. The mesh size is then
        determined as the distance between all points in the fracture
        description.

        Parameters:
            mesh_size_frac: Ideal mesh size. Will be added to all points that
                are sufficiently far away from other points.
            mesh_size_min: Minimal mesh size; we will make no attempts to
                enforce even smaller mesh sizes upon Gmsh.
            mesh_size_bound: Boundary mesh size. Will be added to the points
                defining the boundary, unless there are any fractures in the
                immediate vicinity influencing the size. In other words,
                mesh_size_bound is the boundary point equivalent of
                mesh_size_frac.
        """

        if self.auxiliary_points_added:
            logger.info("Auxiliary points already added. Returning.")
        else:
            self.auxiliary_points_added = True

        self.mesh_size_frac = mesh_size_frac
        self.mesh_size_min = mesh_size_min
        self.mesh_size_bound = mesh_size_bound

        def dist_p(a, b):
            a = a.reshape((-1, 1))
            b = b.reshape((-1, 1))
            d = b - a
            return np.sqrt(np.sum(d ** 2))

        intersecting_fracs = []
        # Loop over all fractures
        for f in self._fractures:

            # First compare segments with intersections to this fracture
            nfp = f.p.shape[1]

            # Keep track of which other fractures are intersecting - will be
            # needed later on
            isect_f = []
            for i in self.intersections_of_fracture(f):
                if f is i.first:
                    isect_f.append(i.second.index)
                else:
                    isect_f.append(i.first.index)

                # Assuming the fracture is convex, the closest point for
                dist, cp = cg.dist_points_segments(
                    i.coord, f.p, np.roll(f.p, 1, axis=1)
                )
                # Insert a (candidate) point only at the segment closest to the
                # intersection point. If the intersection line runs parallel
                # with a segment, this may be insufficient, but we will deal
                # with this if necessary.
                closest_segment = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(i.coord.shape[1]), closest_segment]

                for pi, (si, di) in enumerate(zip(closest_segment, min_dist)):
                    if di < mesh_size_frac:
                        d_1 = dist_p(cp[pi, si], f.p[:, si])
                        d_2 = dist_p(cp[pi, si], f.p[:, (si + 1) % nfp])
                        # If the intersection point is not very close to any of
                        # the points on the segment, we split the segment.
                        if d_1 > mesh_size_min and d_2 > mesh_size_min:
                            np.insert(f.p, (si + 1) % nfp, cp[pi, si], axis=1)

            # Take note of the intersecting fractures
            intersecting_fracs.append(isect_f)

        for fi, f in enumerate(self._fractures):
            nfp = f.p.shape[1]
            for of in self._fractures:
                # Can do some box arguments here to avoid computations

                # First, check if we are intersecting, this is covered already
                if of.index in intersecting_fracs[fi]:
                    continue

                f_start = f.p
                f_end = np.roll(f_start, 1, axis=1)
                # Then, compare distance between segments
                # Loop over fracture segments of this fracture
                for si, fs in enumerate(f.segments()):
                    of_start = of.p
                    of_end = np.roll(of_start, 1, axis=1)
                    # Compute distance to all fractures of the other fracture,
                    # and find the closest segment on the other fracture
                    fs = f_start[:, si].squeeze()  # .reshape((-1, 1))
                    fe = f_end[:, si].squeeze()  # .reshape((-1, 1))
                    d, cp_f, _ = cg.dist_segment_segment_set(fs, fe, of_start, of_end)
                    mi = np.argmin(d)
                    # If the distance is smaller than ideal length, but the
                    # closets point is not too close to the segment endpoints,
                    # we add a new point
                    if d[mi] < mesh_size_frac:
                        d_1 = dist_p(cp_f[:, mi], f.p[:, si])
                        d_2 = dist_p(cp_f[:, mi], f.p[:, (si + 1) % nfp])
                        if d_1 > mesh_size_min and d_2 > mesh_size_min:
                            np.insert(f.p, (si + 1) % nfp, cp_f[:, mi], axis=1)

    def to_vtk(self, file_name, data=None, binary=True):
        """
        Export the fracture network to vtk.

        The fractures are treated as polygonal cells, with no special treatment
        of intersections.

        Fracture numbers are always exported (1-offset). In addition, it is
        possible to export additional data, as specified by the
        keyword-argument data.

        Parameters:
            file_name (str): Name of the target file.
            data (dictionary, optional): Data associated with the fractures.
                The values in the dictionary should be numpy arrays. 1d and 3d
                data is supported. Fracture numbers are always exported.
            binary (boolean, optional): Use binary export format. Defaults to
                True.

        """
        network_vtk = vtk.vtkUnstructuredGrid()

        point_counter = 0
        pts_vtk = vtk.vtkPoints()
        for f in self._fractures:
            # Add local points
            [pts_vtk.InsertNextPoint(*p) for p in f.p.T]

            # Indices of local points
            loc_pt_id = point_counter + np.arange(f.p.shape[1], dtype="int")
            # Update offset
            point_counter += f.p.shape[1]

            # Add bounding polygon
            frac_vtk = vtk.vtkIdList()
            [frac_vtk.InsertNextId(p) for p in loc_pt_id]
            # Close polygon
            frac_vtk.InsertNextId(loc_pt_id[0])

            network_vtk.InsertNextCell(vtk.VTK_POLYGON, frac_vtk)

        # Add the points
        network_vtk.SetPoints(pts_vtk)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(network_vtk)
        writer.SetFileName(file_name)

        if not binary:
            writer.SetDataModeToAscii()

        # Cell-data to be exported is at least the fracture numbers
        if data is None:
            data = {}
        # Use offset 1 for fracture numbers (should we rather do 0?)
        data["Fracture_Number"] = 1 + np.arange(len(self._fractures))

        for name, data in data.items():
            data_vtk = vtk_np.numpy_to_vtk(
                data.ravel(order="F"), deep=True, array_type=vtk.VTK_DOUBLE
            )
            data_vtk.SetName(name)
            data_vtk.SetNumberOfComponents(1 if data.ndim == 1 else 3)
            network_vtk.GetCellData().AddArray(data_vtk)

        writer.Update()

    def to_gmsh(self, file_name, in_3d=True, **kwargs):
        """ Write the fracture network as input for mesh generation by gmsh.

        It is assumed that intersections have been found and processed (e.g. by
        the methods find_intersection() and split_intersection()).

        Parameters:
            file_name (str): Path to the .geo file to be written
            in_3d (boolean, optional): Whether to embed the 2d fracture grids
               in 3d. If True (default), the mesh will be DFM-style, False will
               give a DFN-type mesh.

        """
        # Extract geometrical information.
        p = self.decomposition["points"]
        edges = self.decomposition["edges"]
        poly = self._poly_2_segment()
        # Obtain tags, and set default values (corresponding to real fractures)
        # for untagged fractures.
        frac_tags = self.tags
        frac_tags["subdomain"] = frac_tags.get("subdomain", []) + [False] * (
            len(self._fractures) - len(frac_tags.get("subdomain", []))
        )
        frac_tags["boundary"] = frac_tags.get("boundary", []) + [False] * (
            len(self._fractures) - len(frac_tags.get("boundary", []))
        )

        edge_tags, intersection_points = self._classify_edges(poly)

        # All intersection lines and points on boundaries are non-physical in 3d.
        # I.e., they are assigned boundary conditions, but are not gridded. Hence:
        # Remove the points and edges at the boundary
        point_tags, edge_tags = self.on_domain_boundary(edges, edge_tags)
        edges = np.vstack((self.decomposition["edges"], edge_tags))
        int_pts_on_boundary = np.isin(intersection_points, np.where(point_tags))
        intersection_points = intersection_points[np.logical_not(int_pts_on_boundary)]
        self.zero_d_pt = intersection_points

        # Obtain mesh size parameters
        mesh_size = self._determine_mesh_size(boundary_point_tags=point_tags)

        # The tolerance applied in gmsh should be consistent with the tolerance
        # used in the splitting of the fracture network. The documentation of
        # gmsh is not clear, but it seems gmsh scales a given tolerance with
        # the size of the domain - presumably by the largest dimension. To
        # counteract this, we divide our (absolute) tolerance self.tol with the
        # domain size.
        if in_3d:
            dx = np.array(
                [
                    [self.domain["xmax"] - self.domain["xmin"]],
                    [self.domain["ymax"] - self.domain["ymin"]],
                    [self.domain["zmax"] - self.domain["zmin"]],
                ]
            )
            gmsh_tolerance = self.tol / dx.max()
        else:
            gmsh_tolerance = self.tol

        meshing_algorithm = kwargs.get("meshing_algorithm", None)

        # Initialize and run the gmsh writer:
        if in_3d:
            dom = self.domain
        else:
            dom = None
        writer = GmshWriter(
            p,
            edges,
            polygons=poly,
            domain=dom,
            intersection_points=intersection_points,
            mesh_size=mesh_size,
            tolerance=gmsh_tolerance,
            edges_2_frac=self.decomposition["line_in_frac"],
            meshing_algorithm=meshing_algorithm,
            fracture_tags=frac_tags,
        )

        writer.write_geo(file_name)

    def to_csv(self, file_name, domain=None):
        """
        Save the 3d network on a csv file with comma , as separator.
        Note: the file is overwritten if present.
        The format is
        - if domain is given the first line describes the domain as a rectangle
          X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX
        - the other lines descibe the N fractures as a list of points
          P0_X, P0_Y, P0_Z, ...,PN_X, PN_Y, PN_Z

        Parameters:
            file_name: name of the file
            domain: (optional) the bounding box of the problem
        """

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            # if the domain (as bounding box) is defined save it
            if domain is not None:
                order = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                csv_writer.writerow([domain[o] for o in order])

            # write all the fractures
            [csv_writer.writerow(f.p.ravel(order="F")) for f in self._fractures]

    def to_fab(self, file_name):
        """
        Save the 3d network on a fab file, as specified by FracMan.

        The filter is based on the .fab-files needed at the time of writing, and
        may not cover all options available.

        Parameters:
            file_name (str): File name.
        """
        # function to write a numpy matrix as string
        def to_file(p):
            return "\n\t\t".join(" ".join(map(str, x)) for x in p)

        with open(file_name, "w") as f:
            # write the first part of the file, some information are fake
            num_frac = len(self._fractures)
            num_nodes = np.sum([frac.p.shape[1] for frac in self._fractures])
            f.write(
                "BEGIN FORMAT\n\tFormat = Ascii\n\tXAxis = East\n"
                + "\tScale = 100.0\n\tNo_Fractures = "
                + str(num_frac)
                + "\n"
                + "\tNo_TessFractures = 0\n\tNo_Nodes = "
                + str(num_nodes)
                + "\n"
                + "\tNo_Properties = 0\nEND FORMAT\n\n"
            )

            # start to write the fractures
            f.write("BEGIN FRACTURE\n")
            for frac_pos, frac in enumerate(self._fractures):
                f.write("\t" + str(frac_pos) + " " + str(frac.p.shape[1]) + " 1\n\t\t")
                p = np.concatenate(
                    (np.atleast_2d(np.arange(frac.p.shape[1])), frac.p), axis=0
                ).T
                f.write(to_file(p) + "\n")
                f.write("\t0 -1 -1 -1\n")
            f.write("END FRACTURE")

    def fracture_to_plane(self, frac_num):
        """ Project fracture vertexes and intersection points to the natural
        plane of the fracture.

        Parameters:
            frac_num (int): Index of fracture.

        Returns:
            np.ndarray (2xn_pt): 2d coordinates of the fracture vertexes.
            np.ndarray (2xn_isect): 2d coordinates of fracture intersection
                points.
            np.ndarray: Index of intersecting fractures.
            np.ndarray, 3x3: Rotation matrix into the natural plane.
            np.ndarray, 3x1. 3d coordinates of the fracture center.

            The 3d coordinates of the frature can be recovered by
                p_3d = cp + rot.T.dot(np.vstack((p_2d,
                                                 np.zeros(p_2d.shape[1]))))

        """
        isect = self.intersections_of_fracture(frac_num)

        frac = self._fractures[frac_num]
        cp = frac.center.reshape((-1, 1))

        rot = cg.project_plane_matrix(frac.p)

        def rot_translate(pts):
            # Convenience method to translate and rotate a point.
            return rot.dot(pts - cp)

        p = rot_translate(frac.p)
        assert np.max(np.abs(p[2])) < self.tol, (
            str(np.max(np.abs(p[2]))) + " " + str(self.tol)
        )
        p_2d = p[:2]

        # Intersection points, in 2d coordinates
        ip = np.empty((2, 0))

        other_frac = np.empty(0, dtype=np.int)

        for i in isect:
            if i.first.index == frac_num:
                other_frac = np.append(other_frac, i.second.index)
            else:
                other_frac = np.append(other_frac, i.first.index)

            tmp_p = rot_translate(i.coord)
            if tmp_p.shape[1] > 0:
                assert np.max(np.abs(tmp_p[2])) < self.tol
                ip = np.append(ip, tmp_p[:2], axis=1)

        return p_2d, ip, other_frac, rot, cp

    def on_domain_boundary(self, edges, edge_tags):
        """
        Finds edges and points on boundary, to avoid that these
        are gridded. Points  introduced by intersections
        of subdomain boundaries and real fractures remain physical
        (to maintain contact between split fracture lines).
        """

        constants = GmshConstants()
        # Obtain current tags on fractures
        boundary_polygons = np.where(
            self.tags.get("boundary", [False] * len(self._fractures))
        )[0]
        subdomain_polygons = np.where(
            self.tags.get("subdomain"[False] * len(self._fractures))
        )[0]
        # ... on the points...
        point_tags = np.zeros(self.decomposition["points"].shape[1])
        # and the mapping between fractures and edges.
        edges_2_frac = self.decomposition["edges_2_frac"]

        # Loop on edges to tag according to following rules:
        #     Edge is on the boundary:
        #         Tag it and the points it consists of as boundary entities.
        #     Edge is caused by the presence of subdomain boundaries
        #     (Note: one fracture may still be involved):
        #         Tag it as auxiliary
        for e, e2f in enumerate(edges_2_frac):
            if any(np.in1d(e2f, boundary_polygons)):
                edge_tags[e] = constants.DOMAIN_BOUNDARY_TAG
                point_tags[edges[:, e]] = constants.DOMAIN_BOUNDARY_TAG
                continue
            subdomain_parents = np.in1d(e2f, subdomain_polygons)
            if any(subdomain_parents) and e2f.size - sum(subdomain_parents) < 2:
                # Intersection of at most one fracture with one (or more)
                # subdomain boundaries.
                edge_tags[e] = constants.AUXILIARY_TAG

        # Tag points caused solely by subdomain plane intersection as
        # auxiliary. Note that points involving at least one real fracture
        # are kept as physical fracture points. This is to avoid connectivity
        # loss along real fracture intersections.
        for p in np.unique(edges):
            es = np.where(edges == p)[1]
            sds = []
            for e in es:
                sds.append(edges_2_frac[e])
            sds = np.unique(np.concatenate(sds))
            subdomain_parents = np.in1d(sds, subdomain_polygons)
            if all(subdomain_parents) and sum(subdomain_parents) > 2:
                point_tags[p] = constants.AUXILIARY_TAG

        return point_tags, edge_tags
