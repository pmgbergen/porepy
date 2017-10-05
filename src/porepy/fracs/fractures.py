"""
A module for representation and manipulations of fractures and fracture sets.

The model relies heavily on functions in the computational geometry library.

"""
# Import of 'standard' external packages
import warnings
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy

# Imports of external packages that may not be present at the system. The
# module will work without any of these, but with limited functionalbility.
try:
    import triangle
except ImportError:
    warnings.warn('The triangle module is not available. Gridding of fracture'
                  ' networks will not work')
try:
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    warnings.warn('VTK module is not available. Export of fracture network to\
    vtk will not work.')

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    warnings.warn('Matplotlib is not available. Simple plotting will not work')

# Import of internally developed packages.
from porepy.utils import comp_geom as cg
from porepy.utils import setmembership, sort_points
from porepy.grids import simplex
from porepy.grids.gmsh.gmsh_interface import GmshWriter
from porepy.grids.constants import GmshConstants
from porepy.fracs.utils import determine_mesh_size


# Module-wide logger
logger = logging.getLogger(__name__)

class Fracture(object):

    def __init__(self, points, index=None, check_convexity=True):
        self.p = points
        # Ensure the points are ccw
        self.points_2_ccw()
        self.compute_centroid()
        self.normal = cg.compute_normal(points)[:, None]

        self.orig_p = self.p.copy()

        self.index = index

        assert self.is_planar(), 'Points define non-planar fracture'
        if check_convexity:
            assert self.check_convexity(), 'Points form non-convex polygon'

    def set_index(self, i):
        self.index = i

    def __eq__(self, other):
        return self.index == other.index

    def copy(self):
        """ Return a deep copy of the fracture.

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
        up, _, ind = setmembership.unique_columns_tol(ap, tol=tol*np.sqrt(3))
        if up.shape[1] == ap.shape[1]:
            return False, None
        else:
            occurences = np.where(ind == ind[0])[0]
            return True, (occurences[1] - 1)

    def points_2_ccw(self):
        """
        Ensure that the points are sorted in a counter-clockwise order.

        mplementation note:
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

    def add_points(self, p, check_convexity=True, tol=1e-4):
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
            tol (double): Tolerance used to check if the point already exists.

        Return:
            boolean, true if the resulting polygon is convex.

        """
        self.p = np.hstack((self.p, p))
        self.p, _, _ = setmembership.unique_columns_tol(self.p, tol=tol)

        # Sort points to ccw
        self.p = self.p[:, self.points_2_ccw()]

        if check_convexity:
            return self.check_convexity() and self.is_planar(tol)
        else:
            return self.check_convexity()

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
        cc = (p[:, 0].reshape((-1, 1)) + p[:, 1:-1] + p[:, 2:])/3
        # Area of triangles
        area = 0.5 * np.abs(v[0, :-1] * v[1, 1:] - v[1, :-1] * v[0, 1:])

        # The center is found as the area weighted center
        center = np.sum(cc * area, axis=1) / np.sum(area)

        # Project back again.
        self.center = rot.transpose().dot(np.append(center, z))

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

        sp = [sympy.geometry.Point(p[:, i])
              for i in range(p.shape[1])]
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
        scaled_tol = tol * max(1, max(np.max(np.abs(s_2_o)),
                                      np.max(np.abs(o_2_s))))

        # We can terminate only if no vertexes are close to the plane of the
        # other fractures.
        if np.min(np.abs(other_from_self)) > scaled_tol and \
            np.min(np.abs(self_from_other)) > scaled_tol:
            # If one of the fractures has all of its points on a single side of
            # the other, there can be no intersections.
            if np.all(np.sign(other_from_self) == 1) or \
                np.all(np.sign(other_from_self) == -1) or \
                np.all(np.sign(self_from_other) == 1) or \
                np.all(np.sign(self_from_other) == -1):
                return int_points, on_boundary_self, on_boundary_other

        ####
        # Check for intersection between interior of one polygon with
        # segment of the other.

        # Compute intersections, with both polygons as first argument
        isect_self_other = cg.polygon_segment_intersect(self.p, other.p, tol=tol)
        isect_other_self = cg.polygon_segment_intersect(other.p, self.p, tol=tol)

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
            int_points, _, _ \
                = setmembership.unique_columns_tol(int_points, tol=tol)
        # There should be at most two of these points
        assert int_points.shape[1] <= 2

        # There at at the most two intersection points between the fractures
        # (assuming convexity). If two interior points are found, we can simply
        # cut it short here.
        if int_points.shape[1] == 2:
            return int_points, on_boundary_self, on_boundary_other

        ####
        # Next, check for intersections between the polygon boundaries
        bound_sect_self_other = cg.polygon_boundaries_intersect(self.p,
                                                                other.p,
                                                                tol=tol)
        bound_sect_other_self = cg.polygon_boundaries_intersect(other.p,
                                                                self.p,
                                                                tol=tol)


        # Short cut: If no boundary intersections, we return the interior
        # points
        if len(bound_sect_self_other) == 0 and len(bound_sect_other_self) == 0:
            if check_point_contact and int_points.shape[1] == 1:
                raise ValueError('Contact in a single point not allowed')
            # None of the intersection points lay on the boundary
            return int_points, on_boundary_self, on_boundary_other

        # Else, we have boundary intersection, and need to process them
        bound_pt_self, self_segment, _, self_cuts_through = \
                self._process_segment_isect(bound_sect_self_other, self.p, tol)

        bound_pt_other, other_segment, _, other_cuts_through = \
                self._process_segment_isect(bound_sect_other_self, other.p,
                                            tol)

        # Run some sanity checks

        # Convex polygons can intersect each other in at most two points (and a
        # line inbetween)
        if int_points.shape[1] > 1:
            assert bound_pt_self.shape[1] == 0 and bound_pt_other.shape[1] == 0
        elif int_points.shape[1] == 1:
            # There should be exactly one unique boundary point.
            bp = np.hstack((bound_pt_self, bound_pt_other))
            if bp.shape[1] == 0:
                # The other point on the intersection line must be a vertex.
                # Brute force search
                found_vertex = False
                for self_p in self.p.T:
                    diff = self_p.reshape((-1, 1)) - other.p
                    dist = np.sqrt(np.sum(np.power(diff, 2), axis=0))
                    if np.any(np.isclose(dist, 0, rtol=tol)):
                        if found_vertex:
                            raise ValueError('Found > 2 intersection points')
                        found_vertex = True
                        int_points = np.hstack((int_points,
                                                self_p.reshape((-1, 1))))

                # There is not more to say here, simply return
                return int_points, on_boundary_self, on_boundary_other

            u_bound_pt, _, _ = setmembership.unique_columns_tol(bp, tol=tol)
            assert u_bound_pt.shape[1] == 1, 'The fractures should be convex'

        # If a segment runs through the polygon, there should be no interior points.
        # This may be sensitive to tolerances, should be a useful test to gauge
        # that.
        if self_cuts_through or other_cuts_through:
            assert int_points.shape[1] == 0

        # Storage for intersection points located on the boundary
        bound_pt = []

        # Cover the case of a segment - essentially an L-intersection
        if self_segment:
            assert other_segment  # This should be reflexive
            assert bound_pt_self.shape[1] == 2
            assert np.allclose(bound_pt_self, bound_pt_other)
            on_boundary_self = [True, True]
            on_boundary_other = [True, True]
            return bound_pt_self, on_boundary_self, on_boundary_other

        # Case where one boundary segment of one fracture cuts through two
        # boundary segment (thus the surface) of the other fracture. Gives a
        # T-type intersection
        if self_cuts_through:
            assert np.allclose(bound_pt_self, bound_pt_other)
            on_boundary_self = True
            on_boundary_other = False
            return bound_pt_self, on_boundary_self, on_boundary_other
        elif other_cuts_through:
            assert np.allclose(bound_pt_self, bound_pt_other)
            on_boundary_self = False
            on_boundary_other = True
            return bound_pt_self, on_boundary_self, on_boundary_other

        # By now, there should be a single member of bound_pt
        assert bound_pt_self.shape[1] == 1 or bound_pt_self.shape[1] == 2
        assert bound_pt_other.shape[1] == 1 or bound_pt_other.shape[1] == 2
        bound_pt = np.hstack((int_points, bound_pt_self))

        # Check if there are two boundary points on the same segment. If so,
        # this is a boundary segment.
        on_boundary_self = False
        if bound_pt_self.shape[1] == 2:
            if bound_sect_self_other[0][0] == bound_sect_self_other[1][0]:
                on_boundary_self = True
        on_boundary_other = False
        if bound_pt_other.shape[1] == 2:
            if bound_sect_other_self[0][0] == bound_sect_other_self[1][0]:
                on_boundary_other = True
        # Special case if a segment is cut as one interior point (by its
        # vertex), and a boundary point on the same segment, this is a boundary
        # segment
        if int_points.shape[1] == 1 and bound_pt_self.shape[1] == 1:
            is_vert, vi = self.is_vertex(int_points)
            seg_i = bound_sect_self_other[0][0]
            nfp = self.p.shape[1]
            if is_vert and (vi == seg_i or (seg_i+1) % nfp == vi):
                on_boundary_self = True
        if int_points.shape[1] == 1 and bound_pt_other.shape[1] == 1:
            is_vert, vi = other.is_vertex(int_points)
            seg_i = bound_sect_other_self[0][0]
            nfp = other.p.shape[1]
            if is_vert and (vi == seg_i or (seg_i+1) % nfp == vi):
                on_boundary_other = True

        return bound_pt, on_boundary_self, on_boundary_other

    def _process_segment_isect(self, isect_bound, poly, tol):
        """
        Helper function to interpret result from polygon_boundaries_intersect

        Classify an intersection between segments, from the perspective of a
        polygon, as:
        i) Sharing a segment
        ii) Sharing part of a segment
        ii) Sharing a point, which coincides with a polygon vertex
        iii) Sharing a point that splits a segment of the polygn

        Parameters:
            isect_bound: intersections
            poly: First (not second!) polygon sent into intersection finder

        """
        bound_pt = np.empty((3, 0))

        # Boolean indication of a shared segment
        has_segment = False
        # Boolean indication of an intersection in a vertex
        non_vertex = []
        # Counter for how many times each segment is found
        num_occ = np.zeros(poly.shape[1])

        # Loop over all intersections, process information
        for bi in isect_bound:

            # Index of the intersecting segments
            num_occ[bi[0]] += 1

            # Coordinates of the intersections
            ip = bi[2]

            # No intersections should have more than two poitns
            assert ip.shape[1] < 3

            ip_unique, _, _ = setmembership.unique_columns_tol(ip, tol=tol)
            if ip_unique.shape[1] == 2:
                # The polygons share a segment, or a
                bound_pt = np.hstack((bound_pt, ip_unique))
                has_segment = True
                # No need to deal with non_vertex here, there should be no more
                # intersections (convex, non-overlapping polygons).
            elif ip_unique.shape[1] == 1:
                # Either a vertex or single intersection point.
                poly_ext, _, _ = setmembership.unique_columns_tol(
                    np.hstack((self.p, ip_unique)), tol=tol)
                if poly_ext.shape[1] == self.p.shape[1]:
                    # This is a vertex, we skip it
                    pass
                else:
                    # New point contact
                    bound_pt = np.hstack((bound_pt, ip_unique))
                    non_vertex.append(True)
            else:
                # This should never happen for convex polygons
                raise ValueError(
                    'Should not have more than two intersection points')

        # Now, if any segment has been found more than once, it cuts two
        # segments of the other polygon.
        cuts_two = np.any(num_occ > 1)

        # Return a unique version of bound_pt
        # No need to uniquify unless there is more than one point.
        if bound_pt.shape[1] > 1:
            bound_pt, _, _ = setmembership.unique_columns_tol(bound_pt, tol=tol)

        return bound_pt, has_segment, non_vertex, cuts_two


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
        x0_box = box['xmin']
        x1_box = box['xmax']
        y0_box = box['ymin']
        y1_box = box['ymax']
        z0_box = box['zmin']
        z1_box = box['zmax']

        # Gather the box coordinates in an array
        box_array = np.array([[x0_box, x1_box],
                              [y0_box, y1_box],
                              [z0_box, z1_box]])

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
            p = cg.snap_to_grid(p, tol=tol)
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
            outside = np.vstack((west_of, east_of, south_of, north_of,
                                 beneath, above))
            return outside[bound_i]


        # Represent the planes of the bounding box as fractures, to allow
        # us to use the fracture intersection methods.
        # For each plane, we keep the fixed coordinate to the value specified
        # by the box, and extend the two others to cover segments that run
        # through the extension of the plane only.
        west = Fracture(np.array([[x0_box, x0_box, x0_box, x0_box],
                                  [y0, y1, y1, y0], [z0, z0, z1, z1]]))
        east = Fracture(np.array([[x1_box, x1_box, x1_box, x1_box],
                                  [y0, y1, y1, y0],
                                  [z0, z0, z1, z1]]))
        south = Fracture(np.array([[x0, x1, x1, x0],
                                   [y0_box, y0_box, y0_box, y0_box],
                                   [z0, z0, z1, z1]]))
        north = Fracture(np.array([[x0, x1, x1, x0],
                                   [y1_box, y1_box, y1_box, y1_box],
                                   [z0, z0, z1, z1]]))
        bottom = Fracture(np.array([[x0, x1, x1, x0], [y0, y0, y1, y1],
                                    [z0_box, z0_box, z0_box, z0_box]]))
        top = Fracture(np.array([[x0, x1, x1, x0], [y0, y0, y1, y1],
                                 [z1_box, z1_box, z1_box, z1_box]]))
        # Collect in a list to allow iteration
        bound_planes = [west, east, south, north, bottom, top]

        # Loop over all boundary sides and look for intersections
        for bi, bf in enumerate(bound_planes):
            # Dimensions that are not fixed by bf
            active_dims = np.ones(3, dtype=np.bool)
            active_dims[np.floor(bi/2).astype('int')] = 0
            # Convert to indices
            active_dims = np.squeeze(np.argwhere(active_dims))

            # Find intersection points
            isect, _, _ = self.intersects(bf, tol, check_point_contact=False)
            num_isect = isect.shape[1]
            if len(isect) > 0 and num_isect > 0:
                num_pts_orig = self.p.shape[1]
                # Add extra points at the end of the point list.
                self.p, _, _ \
                    = setmembership.unique_columns_tol(np.hstack((self.p,
                                                                  isect)))
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

                # If all intersection points are outside, and on the same side
                # of the box for all of the active dimensions, convexity of the
                # polygon implies that the whole fracture is outside the
                # bounding box. We signify this by setting self.p empty.
                if np.any(np.all(isect[active_dims] < \
                          box_array[active_dims, 0], axis=1), axis=0):
                    self.p = np.empty((3, 0))
                    # No need to consider other boundary planes
                    break
                if np.any(np.all(isect[active_dims] > \
                          box_array[active_dims, 1], axis=1), axis=0):
                    self.p = np.empty((3, 0))
                    # No need to consider other boundary planes
                    break

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
                tangent *= 1./ np.linalg.norm(tangent)

                isect_ind = np.argwhere(sort_ind >= num_pts_orig).ravel('F')
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
        return str(self.as_sp_polygon())

    def __str__(self):
        s = 'Points: \n'
        s += str(self.p) + '\n'
        s += 'Center: ' + str(self.center)
        return s

    def plot_frame(self, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        x = np.append(self.p[0], self.p[0, 0])
        y = np.append(self.p[1], self.p[1, 0])
        z = np.append(self.p[2], self.p[2, 0])
        ax.plot(x, y, z)
        return ax

#--------------------------------------------------------------------


class EllipticFracture(Fracture):
    """
    Subclass of Fractures, representing an elliptic fracture.

    """

    def __init__(self, center, major_axis, minor_axis, major_axis_angle,
                 strike_angle, dip_angle, num_points=16):
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
        self.center = center

        # First, populate polygon in the xy-plane
        angs = np.linspace(0, 2 * np.pi, num_points + 1, endpoint=True)[:-1]
        x = major_axis * np.cos(angs)
        y = minor_axis * np.sin(angs)
        z = np.zeros_like(angs)
        ref_pts = np.vstack((x, y, z))

        # Rotate reference points so that the major axis has the right
        # orientation
        major_axis_rot = cg.rot(major_axis_angle, [0, 0, 1])
        rot_ref_pts = major_axis_rot.dot(ref_pts)

        # Then the dip
        # Rotation matrix of the strike angle
        strike_rot = cg.rot(strike_angle, np.array([0, 0, 1]))
        # Compute strike direction
        strike_dir = strike_rot.dot(np.array([1, 0, 0]))
        dip_rot = cg.rot(dip_angle, strike_dir)

        dip_pts = dip_rot.dot(rot_ref_pts)

        # Set the points, and store them in a backup.
        self.p = center[:, np.newaxis] + dip_pts
        self.orig_p = self.p.copy()

        # Compute normal vector
        self.normal = cg.compute_normal(self.p)[:, None]


#-------------------------------------------------------------------------


class Intersection(object):

    def __init__(self, first, second, coord, bound_first=False, bound_second=False):
        self.first = first
        self.second = second
        self.coord = coord
        # Information on whether the intersection points lies on the boundaries
        # of the fractures
        self.bound_first = bound_first
        self.bound_second = bound_second

    def __repr__(self):
        s = 'Intersection between fractures ' + str(self.first.index) + ' and ' + \
            str(self.second.index) + '\n'
        s += 'Intersection points: \n'
        for i in range(self.coord.shape[1]):
            s += '(' + str(self.coord[0, i]) + ', ' + str(self.coord[1, i]) + ', ' + \
                 str(self.coord[2, i]) + ') \n'
        if self.coord.size > 0:
            s += 'On boundary of first fracture ' + str(self.bound_first) + '\n'
            s += 'On boundary of second fracture ' + str(self.bound_second)
            s += '\n'
        return s

    def get_other_fracture(self, i):
        if self.first == i:
            return self.second
        else:
            return self.first

#----------------------------------------------------------------------------


class FractureNetwork(object):

    def __init__(self, fractures=None, verbose=0, tol=1e-4):

        if fractures is None:
            self._fractures = fractures
        else:
            self._fractures = fractures

        for i, f in enumerate(self._fractures):
            f.set_index(i)

        self.intersections = []

        self.has_checked_intersections = False
        self.tol = tol
        self.verbose = verbose

    def add(self, f):
        # Careful here, if we remove one fracture and then add, we'll have
        # identical indices.
        f.set_index(len(self._fractures))
        self._fractures.append(f)

    def __getitem__(self, position):
        return self._fractures[position]

    def get_normal(self, frac):
        return self._fractures[frac].normal

    def get_center(self, frac):
        return self._fractures[frac].center

    def intersections_of_fracture(self, frac):
        """ Get all known intersections for a fracture.

        If called before find_intersections(), the returned list will be empty.

        Paremeters:
            frac: A fracture in the network

        Returns:
            np.array (Intersection): Array of intersections

        """
        fi = frac.index
        frac_arr = []
        for i in self.intersections:
            if i.coord.size == 0:
                continue
            if i.first == frac or i.second == frac:
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
        logger.info('Find intersection between fratures')
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
                isect, bound_first, bound_second = first.intersects(second,
                                                                    self.tol)
                if len(isect) > 0:
                    # Let the intersection know whether both intersection
                    # points lies on the boundary of each fracture
                    self.intersections.append(Intersection(first, second,
                                                           isect,
                                                           bound_first=bound_first,
                                                           bound_second=bound_second))

        logger.info('Found %i intersections. Ellapsed time: %.5f',
                    len(self.intersections), time.time() - start_time)

    def intersection_info(self, frac_num=None):
        # Number of fractures with some intersection
        num_intersecting_fracs = 0
        # Number of intersections in total
        num_intersections = 0

        if frac_num is None:
            frac_num = np.arange(len(self._fractures))
        for f in frac_num:
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
                    print('  Fracture ' + str(f) + ' intersects with'\
                          ' fractuer(s) ' + str(isects))
        # Print aggregate numbers. Note that all intersections are counted
        # twice (from first and second), thus divide by two.
        print('In total ' + str(num_intersecting_fracs) + ' fractures '
              + 'intersect in ' + str(int(num_intersections/2)) \
              + ' intersections')


    def split_intersections(self):
        """
        Based on the fracture network, and their known intersections, decompose
        the fractures into non-intersecting sub-polygons. These can
        subsequently be exported to gmsh.

        The method will add an atribute decomposition to self.

        """

        logger.info('Split intersections')
        start_time = time.time()

        # First, collate all points and edges used to describe fracture
        # boundaries and intersections.
        all_p, edges,\
            edges_2_frac, is_boundary_edge = self._point_and_edge_lists()

        if self.verbose > 1:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        # By now, all segments in the grid are defined by a unique set of
        # points and edges. The next task is to identify intersecting edges,
        # and split them.
        all_p, edges, edges_2_frac, is_boundary_edge\
            = self._remove_edge_intersections(all_p, edges, edges_2_frac,
                                              is_boundary_edge)

        if self.verbose > 1:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)


        # With all edges being non-intersecting, the next step is to split the
        # fractures into polygons formed of boundary edges, intersection edges
        # and auxiliary edges that connect the boundary and intersections.
#        all_p, edges, is_boundary_edge, poly_segments, poly_2_frac\
#                = self._split_into_polygons(all_p, edges, edges_2_frac,
#                                            is_boundary_edge)

        # Store the full decomposition.
        self.decomposition = {'points': all_p,
                              'edges': edges.astype('int'),
                              'is_bound': is_boundary_edge,
                              'edges_2_frac': edges_2_frac}
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
                    raise ValueError('Non-unique fracture edge relation')
                else:
                    continue

            poly = sort_points.sort_point_pairs(edges[:2, ei_bound])
            polygons.append(poly)
            line_in_frac.append(ei)

        self.decomposition['polygons'] = polygons
        self.decomposition['line_in_frac'] = line_in_frac

#                              'polygons': poly_segments,
#                             'polygon_frac': poly_2_frac}

        logger.info('Finished fracture splitting after %.5f seconds',
                    time.time() - start_time)

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
        logger.info('Compile list of points and edges')
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
            num_p_loc = frac.orig_p.shape[1]
            all_p = np.hstack((all_p, frac.orig_p))

            loc_e = num_p + np.vstack((np.arange(num_p_loc),
                                       (np.arange(num_p_loc) + 1) % num_p_loc))
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

                edges = np.hstack(
                    (edges, num_p + np.arange(2).reshape((-1, 1))))
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
        edges = edges.astype('int')

        logger.info('Points and edges done. Elapsed time %.5f', time.time() -
                    start_time)

        return self._uniquify_points_and_edges(all_p, edges, edges_2_frac,
                                               is_boundary_edge)

    def _uniquify_points_and_edges(self, all_p, edges, edges_2_frac,
                                   is_boundary_edge):
        # Snap the points to an underlying Cartesian grid. This is the basis
        # for declearing two points equal
        # NOTE: We need to account for dimensions in the tolerance;

        start_time = time.time()
        logger.info("""Uniquify points and edges, starting with %i points, %i
                    edges""", all_p.shape[1], edges.shape[1])

        all_p = cg.snap_to_grid(all_p, tol=self.tol)

        # We now need to find points that occur in multiple places
        p_unique, unique_ind_p, \
            all_2_unique_p = setmembership.unique_columns_tol(all_p, tol=
                                                              self.tol *
                                                              np.sqrt(3))

        # Update edges to work with unique points
        edges = all_2_unique_p[edges]

        # Look for edges that share both nodes. These will be identical, and
        # will form either a L/Y-type intersection (shared boundary segment),
        # or a three fractures meeting in a line.
        # Do a sort of edges before looking for duplicates.
        e_unique, unique_ind_e, all_2_unique_e = \
            setmembership.unique_columns_tol(np.sort(edges, axis=0))

        # Update the edges_2_frac map to refer to the new edges
        edges_2_frac_new = e_unique.shape[1] * [np.empty(0)]
        is_boundary_edge_new = e_unique.shape[1] * [np.empty(0)]

        for old_i, new_i in enumerate(all_2_unique_e):
            edges_2_frac_new[new_i], ind =\
                np.unique(np.hstack((edges_2_frac_new[new_i],
                                     edges_2_frac[old_i])),
                          return_index=True)
            tmp = np.hstack((is_boundary_edge_new[new_i],
                             is_boundary_edge[old_i]))
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
        edges_2_frac = [np.unique(np.array(edges_2_frac[i])) for i in
                        range(len(edges_2_frac))]

        # Sanity check, the fractures should still be defined by points in a
        # plane.
        self._verify_fractures_in_plane(p_unique, edges, edges_2_frac)

        logger.info('''Uniquify complete. %i points, %i edges. Ellapsed time
                    %.5f''', p_unique.shape[1], edges.shape[1], time.time() -
                    start_time)

        return p_unique, edges, edges_2_frac, is_boundary_edge

    def _remove_edge_intersections(self, all_p, edges, edges_2_frac,
                                   is_boundary_edge):
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
        logger.info('Remove edge intersections')
        start_time = time.time()

        # The algorithm loops over all fractures, pulls out edges associated
        # with the fracture, project to the local 2D plane, and look for
        # intersections there (direct search in 3D may also work, but this was
        # a simple option). When intersections are found, the global lists of
        # points and edges are updated.
        for fi, frac in enumerate(self._fractures):

            logger.debug('Remove intersections from fracture %i', fi)

            # Identify the edges associated with this fracture
            # It would have been more convenient to use an inverse
            # relationship frac_2_edge here, but that would have made the
            # update for new edges (towards the end of this loop) more
            # cumbersome.
            edges_loc_ind = []
            for ei, e in enumerate(edges_2_frac):
                if np.any(e == fi):
                    edges_loc_ind.append(ei)

            edges_loc = np.vstack((edges[:, edges_loc_ind],
                                   np.array(edges_loc_ind)))
            p_ind_loc = np.unique(edges_loc[:2])
            p_loc = all_p[:, p_ind_loc]

            p_2d, edges_2d, p_loc_c, rot = self._points_2_plane(p_loc,
                                                                edges_loc,
                                                                p_ind_loc)

            # Add a tag to trace the edges during splitting
            edges_2d[2] = edges_loc[2]

            # Obtain new points and edges, so that no edges on this fracture
            # are intersecting.
            # It seems necessary to increase the tolerance here somewhat to
            # obtain a more robust algorithm. Not sure about how to do this
            # consistent.
            p_new, edges_new = cg.remove_edge_crossings(p_2d, edges_2d,
                                                        tol=self.tol*5,
                                                        verbose=self.verbose)
            # Then, patch things up by converting new points to 3D,

            # From the design of the functions in cg, we know that new points
            # are attached to the end of the array. A more robust alternative
            # is to find unique points on a combined array of p_loc and p_new.
            p_add = p_new[:, p_ind_loc.size:]
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
                is_old, old_loc_ind =\
                    setmembership.ismember_rows(edges_new_glob[:, ei]
                                                .reshape((-1, 1)),
                                                edges[:2, edges_loc_ind])
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


        logger.info('Done with intersection removal. Elapsed time %.5f',
                    time.time() - start_time)
        self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        return self._uniquify_points_and_edges(all_p, edges, edges_2_frac,
                                               is_boundary_edge)

        # To
        # define the auxiliary edges, we create a triangulation of the
        # fractures. We then grow polygons forming part of the fracture in a
        # way that ensures that no polygon lies on both sides of an
        # intersection edge.

    def _split_into_polygons(self, all_p, edges, edges_2_frac, is_boundary_edge):
        """
        Split the fracture surfaces into non-intersecting polygons.

        Starting from a description of the fracture polygons and their
        intersection lines (which should be non-intersecting), the fractures
        are split into sub-polygons which they may share a boundary, but no
        sub-polygon intersect the surface of another. This step is necessary
        for the fracture network to be ammenable to gmsh.

        The sub-polygons are defined by drawing auxiliary lines between
        fracture points. This expands the global set of edges, but not the
        points. The auxiliary lines will be sent to gmsh as constraints on the
        gridding algorithm.

        TODO: The restriction that the auxiliary lines can only be drawn
        between existing lines can create unfortunate constraints in the
        gridding, in terms of sharp angles etc. This can be alleviated by
        adding additional points to the global description. However, if a point
        is added to an intersection edge, this will require an update of all
        fractures sharing that intersection. Not sure what the right thing to
        do here is.

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
            np.ndarray (3xn_pt): Coordinates of all points used to describe the
                fractures.
            np.ndarray (3xn_edges): Connections between points, formed by
                fracture boundaries, fracture intersections, or auxiliary lines
                needed to form sub-polygons. The auxiliary lines will be added
                towards the end of the array.
            np.ndarray (boolean, 3xn_edges_orig): Flag telling if an edge is on
                the boundary of a fracture. NOTE: The list is *not* expanded as
                auxiliary lines are added; these are known to not be on the
                boundary. The legth of this output, relative to the second, can
                thus be used to flag auxiliary lines.
            list of np.ndarray: Each item contains a sub-polygon, specified by
                the global indices of its vertices.
            list of int: For each sub-polygon, index of the fracture it forms a
                part of.

            Note that for the moment, the first and third outputs are not
            modified compared to respective input. This will change if the
            splitting is allowed to introduce new additional points.

        """
        logger.info('Split fractures into non-intersecting polygons')
        start_time = time.time()

        # For each polygon, list of segments that make up the polygon
        poly_segments = []
        # Which fracture is the polygon part of
        poly_2_frac = []

        for fi, _ in enumerate(self._fractures):

            if self.verbose > 1:
                print('  Split fracture no ' + str(fi))

            # Identify the edges associated with this fracture
            # It would have been more convenient to use an inverse
            # relationship frac_2_edge here, but that would have made the
            # update for new edges (towards the end of this loop) more
            # cumbersome.
            edges_loc_ind = []
            for ei, e in enumerate(edges_2_frac):
                if np.any(e == fi):
                    edges_loc_ind.append(ei)

            edges_loc = np.vstack((edges[:, edges_loc_ind],
                                   np.array(edges_loc_ind)))
            p_ind_loc = np.unique(edges_loc[:2])
            p_loc = all_p[:, p_ind_loc]

            p_2d, edges_2d, _, _ = self._points_2_plane(p_loc, edges_loc,
                                                        p_ind_loc)


            # Add a tag to trace the edges during splitting
            edges_2d[2] = edges_loc[2]

            # Flag of whether the constraint is an internal boundary (as
            # opposed to boundary of the polygon)
            internal_boundary = np.where([not is_boundary_edge[i]
                                          for i in edges_2d[2]])[0]

            # Create a dictionary of input information to triangle
            triangle_opts = {'vertices': p_2d.transpose(),
                             'segments': edges_2d[:2].transpose(),
                             'segment_markers':
                             edges_2d[2].transpose().astype('int32')
                            }
            # Run a constrained Delaunay triangulation.
            t = triangle.triangulate(triangle_opts, 'p')

            # Pull out information on the triangulation
            segment_markers = \
                np.squeeze(np.asarray(t['segment_markers']).transpose())
            tri = np.sort(np.asarray(t['triangles']).transpose(), axis=0)
            p = np.asarray(t['vertices']).transpose()
            segments = np.sort(np.asarray(t['segments']).transpose(), axis=0)

            p = cg.snap_to_grid(p, tol=self.tol)

            # It turns out that Triangle sometime creates (almost) duplicate
            # points. Our algorithm are based on the points staying, thus these
            # need to go.
            # The problem is characterized by output from Triangle having more
            # points than the input.
            if p.shape[1] > p_2d.shape[1]:

                # Identify duplicate points
                # Not sure about tolerance used here
                p, _, old_2_new = \
                    setmembership.unique_columns_tol(p, tol=np.sqrt(self.tol))
                # Update segment and triangle coordinates to refer to unique
                # points
                segments = old_2_new[segments]
                tri = old_2_new[tri]

                # Remove segments with the same start and endpoint
                segments_remove = np.argwhere(np.diff(segments, axis=0)[0] == 0)
                segments = np.delete(segments, segments_remove, axis=1)
                segment_markers = np.delete(segment_markers, segments_remove)

                # Remove segments that exist in duplicates (a, b) and (b, a)
                segments.sort(axis=0)
                segments, new_2_old_seg, _ = \
                        setmembership.unique_columns_tol(segments)
                segment_markers = segment_markers[new_2_old_seg]

                # Look for degenerate triangles, where the same point occurs
                # twice
                tri.sort(axis=0)
                degenerate_tri = np.any(np.diff(tri, axis=0) == 0, axis=0)
                tri = tri[:, np.logical_not(degenerate_tri)]

                # Remove duplicate triangles, if any
                tri, _, _ = setmembership.unique_columns_tol(tri)


            # We need a connection map between cells. The simplest option was
            # to construct a simplex grid, and use the associated function.
            # While we were at it, we could have used this to find cell
            # centers, cell-face (segment) mappings etc, but for reason, that
            # did not happen.
            g = simplex.TriangleGrid(p, tri)
            g.compute_geometry()
            c2c = g.cell_connection_map()

            def cells_of_segments(cells, s):
                # Find the cells neighboring
                hit_0 = np.logical_or(cells[0] == s[0], cells[0] ==
                                      s[1]).astype('int')
                hit_1 = np.logical_or(
                    cells[1] == s[0], cells[1] == s[1]).astype('int')
                hit_2 = np.logical_or(
                    cells[2] == s[0], cells[2] == s[1]).astype('int')
                hit = np.squeeze(np.where(hit_0 + hit_1 + hit_2 == 2))
                return hit

            px = p[0]
            py = p[1]
            # Cell centers (in 2d coordinates)
            cell_c = np.vstack((np.mean(px[tri], axis=0),
                                np.mean(py[tri], axis=0)))

            # The target sub-polygons should have intersections as an external
            # boundary. The sub-polygons are grown from triangles on each side of
            # the intersection, denoted cw and ccw.
            cw_cells = []
            ccw_cells = []

            # For each internal boundary, find the cells on each side of the
            # boundary; these will be assigned to different polygons. For the
            # moment, there should be a single cell along each side of the
            # boundary, but this may change in the future.

            if internal_boundary.size == 0:
                # For non-intersecting fractures, we just need a single seed
                cw_cells = [0]
            else:
                # Full treatment
                for ib in internal_boundary:
                    segment_match = np.squeeze(np.where(segment_markers ==
                                                        edges_2d[2, ib]))
                    loc_segments = segments[:, segment_match]
                    loc_cells = cells_of_segments(tri, loc_segments)

                    p_0 = p_2d[:, edges_2d[0, ib]]
                    p_1 = p_2d[:, edges_2d[1, ib]]

                    cw_loc = []
                    ccw_loc = []

                    for ci in np.atleast_1d(loc_cells):
                        if cg.is_ccw_polyline(p_0, p_1, cell_c[:, ci]):
                            ccw_loc.append(ci)
                        else:
                            cw_loc.append(ci)
                    cw_cells.append(cw_loc)
                    ccw_cells.append(ccw_loc)

            polygon = np.zeros(tri.shape[1], dtype='int')
            counter = 1
            for c in cw_cells + ccw_cells:
                polygon[c] = counter
                counter += 1

            # A cell in the partitioning may border more than one internal
            # constraint, and thus be assigned two values (one overwritten) in
            # the above loop. This should be fine; the corresponding index
            # will have no cells assigned to it, but that is taken care of
            # below.

            num_iter = 0

            # From the seeds, assign a polygon index to all triangles
            while np.any(polygon == 0):

                num_occurences = np.bincount(polygon)
                # Disregard the 0 (which we know is there, since the while loop
                # took another round).
                order = np.argsort(num_occurences[1:])
                for pi in order:
                    vec = 0 * polygon
                    # We know that the values in polygon are mapped to
                    vec[np.where(polygon == pi + 1)] = 1
                    new_val = c2c * vec
                    not_found = np.where(np.logical_and(polygon == 0,
                                                        new_val > 0))
                    polygon[not_found] = pi + 1

                num_iter += 1
                if num_iter >= tri.shape[1]:
                    # Sanity check, since we're in a while loop. The code
                    # should terminate long before this, but I have not
                    # tried to find an exact maximum number of iterations.
                    raise ValueError('This must be too many iterations')

            logger.debug('Split fracture %i into %i polygons', fi,
                         np.unique(polygon).size)

            # For each cell, find its boundary
            for poly in np.unique(polygon):
                tri_loc = tri[:, np.squeeze(np.where(polygon == poly))]
                # Special case of a single triangle
                if tri_loc.size == 3:
                    tri_loc = tri_loc.reshape((-1, 1))

                seg_loc = np.sort(np.hstack((tri_loc[:2],
                                             tri_loc[1:],
                                             tri_loc[::2])), axis=0)
                # Really need something like accumarray here..
                unique_s, _, all_2_unique = setmembership.unique_columns_tol(
                    seg_loc)
                bound_segment_ind = np.squeeze(np.where(np.bincount(all_2_unique)
                                                        == 1))
                bound_segment = unique_s[:, bound_segment_ind]
                boundary = sort_points.sort_point_pairs(bound_segment)

                # Ensure that the boundary is stored ccw
                b_points = p_2d[:, boundary[:, 0]]
                if not cg.is_ccw_polygon(b_points):
                    boundary = boundary[::-1]

                # The boundary is expressed in terms of the local point
                # numbering. Convert to global numbering.
                boundary_glob = p_ind_loc[boundary].astype('int')
                poly_segments.append(boundary_glob)
                poly_2_frac.append(fi)
                # End of processing this sub-polygon

           # End of processing this fracture.

        # We have now split all fractures into non-intersecting polygons. The
        # polygon boundary consists of fracture boundaries, fracture
        # intersections and auxiliary edges. The latter must be identified, and
        # added to the global edge list.
        all_edges = np.empty((2, 0), dtype='int')
        for p in poly_segments:
            all_edges = np.hstack((all_edges, p))

        # Find edges of the polygons that can be found in the list of existing
        # edges
        edge_exist, _ = setmembership.ismember_rows(all_edges, edges)
        # Also check with the reverse ordering of edge points
        edge_exist_reverse, _ = setmembership.ismember_rows(all_edges[::-1],
                                                            edges)
        # The edge exists if it is found of the orderings
        edge_exist = np.logical_or(edge_exist, edge_exist_reverse)
        # Where in the list of all edges are the new ones located?
        new_edge_ind = np.where([~i for i in edge_exist])[0]
        new_edges = all_edges[:, new_edge_ind]

        # If we have found new edges (will most likely happen if there are
        # intersecting fractures in the network), these should be added to the
        # edge list, in a unique representation.
        if new_edges.size > 0:
            # Sort the new edges to root out the same edge being found twice
            # from different directions
            new_edges = np.sort(new_edges, axis=0)
            # Find unique representation
            unique_new_edges, _, _\
                = setmembership.unique_columns_tol(new_edges)
            edges = np.hstack((edges, unique_new_edges))

        logger.info('Fracture splitting done. Elapsed time %.5f', time.time() -
                    start_time)

        return all_p, edges, is_boundary_edge, poly_segments, poly_2_frac


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

        s = str(len(self._fractures)) + ' fractures are split into '
        s += str(len(d['polygons'])) + ' polygons \n'

        s += 'Number of points: ' + str(d['points'].shape[1]) + '\n'
        s += 'Number of edges: ' + str(d['edges'].shape[1]) + '\n'

        if verbose > 1:
            # Compute minimum distance between points in point set
            dist = np.inf
            p = d['points']
            num_points = p.shape[1]
            hit = np.ones(num_points, dtype=np.bool)
            for i in range(num_points):
                hit[i] = False
                dist_loc = cg.dist_point_pointset(p[:, i], p[:, hit])
                dist = np.minimum(dist, dist_loc.min())
                hit[i] = True
            s += 'Minimal disance between points ' + str(dist) + '\n'

        if do_print:
            print(s)

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
            edge_ind = np.argwhere(np.any(self.decomposition['edges'][:2]\
                                          == i, axis=0)).ravel('F')
            edges_loc = self.decomposition['edges'][:, edge_ind]
            # Loop over all polygons. If their edges are found in edges_loc,
            # store the corresponding fracture index
            for poly_ind, poly in enumerate(self.decomposition['polygons']):
                ismem, _ = setmembership.ismember_rows(edges_loc, poly)
                if any(ismem):
                    fracs_loc.append(self.decomposition['polygon_frac']\
                                     [poly_ind])
            fracs_of_points.append(list(np.unique(fracs_loc)))
        return fracs_of_points

    def close_points(self, dist):
        """
        In the set of points used to describe the fractures (after
        decomposition), find pairs that are closer than a certain distance.

        Parameters:
            dist (double): Threshold distance, all points closer than this will
                be reported.

        Returns:
            List of tuples: Each tuple contain indices of a set of close
                points, and the distance between the points. The list is not
                symmetric; if (a, b) is a member, (b, a) will not be.

        """
        c_points = []

        pt = self.decomposition['points']
        for pi in range(pt.shape[1]):
            d = cg.dist_point_pointset(pt[:, pi], pt[:, pi+1:])
            ind = np.argwhere(d < dist).ravel('F')
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
        rot = cg.project_plane_matrix(p_loc)
        p_2d = rot.dot(p_loc)

        extent = p_2d.max(axis=1) - p_2d.min(axis=1)

        assert extent[2] < np.max(extent[:2]) * self.tol * 5
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
        s = 'Fracture set with ' + str(len(self._fractures)) + ' fractures'
        return s

    def plot_fractures(self, ind=None):
        if ind is None:
            ind = np.arange(len(self._fractures))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for f in self._fractures:
            f.plot_frame(ax)
        return fig


    def impose_external_boundary(self, box=None, truncate_fractures=True):
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
        if box is None:
            OVERLAP = 0.05
            cmin = np.ones((3, 1)) * float('inf')
            cmax = -np.ones((3, 1)) * float('inf')
            for f in self._fractures:
                cmin = np.min(np.hstack((cmin, f.p)), axis=1).reshape((-1, 1))
                cmax = np.max(np.hstack((cmax, f.p)), axis=1).reshape((-1, 1))
            cmin = cmin[:, 0]
            cmax = cmax[:, 0]
            dx = OVERLAP * (cmax - cmin)
            box = {'xmin': cmin[0] - dx[0], 'xmax': cmax[0] + dx[0],
                   'ymin': cmin[1] - dx[1], 'ymax': cmax[1] + dx[1],
                   'zmin': cmin[2] - dx[2], 'zmax': cmax[2] + dx[2]}

        # Insert boundary in the form of a box, and kick out (parts of)
        # fractures outside the box
        self.domain = box

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
        edges = self.decomposition['edges']
        is_bound = self.decomposition['is_bound']
        num_edges = edges.shape[1]

        poly_2_frac = np.arange(len(self.decomposition['polygons']))
        self.decomposition['polygon_frac'] = poly_2_frac

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
        tag = np.zeros(num_edges, dtype='int')

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
        edges = self.decomposition['edges']
        poly = self.decomposition['polygons']

        poly_2_line = []
        line_reverse = []
        for p in poly:
            hit, ind = setmembership.ismember_rows(p, edges[:2], sort=False)
            hit_reverse, ind_reverse = setmembership.ismember_rows(
                p[::-1], edges[:2], sort=False)
            assert np.all(hit + hit_reverse == 1)

            line_ind = np.zeros(p.shape[1])

            hit_ind = np.where(hit)[0]
            hit_reverse_ind = np.where(hit_reverse)[0]
            line_ind[hit_ind] = ind
            line_ind[hit_reverse_ind] = ind_reverse

            poly_2_line.append(line_ind.astype('int'))
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
        mode = kwargs.get('mode', 'constant')

        num_pts = self.decomposition['points'].shape[1]

        if mode == 'constant':
            val = kwargs.get('value', None)
            bound_val = kwargs.get('bound_value', None)
            if val is not None:
                mesh_size = val * np.ones(num_pts)
            else:
                mesh_size = None
            if bound_val is not None:
                mesh_size_bound = bound_val
            else:
                mesh_size_bound = None
            return mesh_size, mesh_size_bound
        else:
            raise ValueError('Unknown mesh size mode ' + mode)

    def compute_distances(self, h_ideal, hmin):

        isect_pt = np.zeros((3, 0), dtype=np.double)

        def dist_p(a, b):
            a = a.reshape((-1, 1))
            b = a.reshape((-1, 1))
            d = b - a
            return np.sqrt(np.sum(d * d))

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
                dist, cp = cg.dist_points_segments(i.coord, f.p,
                                                   np.roll(f.p, 1, axis=1))
                # Insert a (candidate) point only at the segment closest to the
                # intersection point. If the intersection line runs parallel
                # with a segment, this may be insufficient, but we will deal
                # with this if necessary.
                closest_segment = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(i.coord.shape[1]), closest_segment]

                for pi, (si, di) in enumerate(zip(closest_segment, min_dist)):
                    if di < h_ideal:
                        d_1 = dist_p(cp[:, pi], f.p[:, si])
                        d_2 = dist_p(cp[:, pi], f.p[:, (si+1)%nfp])
                        # If the intersection point is not very close to any of
                        # the points on the segment, we split the segment.
                        if d_1 > hmin and d_2 > hmin:
                            np.insert(f.p, (si+1)%nfp, cp[:, pi], axis=1)

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
                    fs = f_start[:, si].squeeze()#.reshape((-1, 1))
                    fe = f_end[:, si].squeeze()#.reshape((-1, 1))
                    d, cp_f, _ = cg.dist_segment_segment_set(fs, fe, of_start,
                                                             of_end)
                    mi = np.argmin(d)
                    # If the distance is smaller than ideal length, but the
                    # closets point is not too close to the segment endpoints,
                    # we add a new point
                    if d[mi] < h_ideal:
                        d_1 = dist_p(cp_f[:, mi], f.p[:, si])
                        d_2 = dist_p(cp_f[:, mi], f.p[:, (si+1)%nfp])
                        if d_1 > hmin and d_2 > hmin:
                            np.insert(f.p, (si+1)%nfp, cp[:, pi], axis=1)

                # Finally, cover the case where the smallest distance is given
                # by a point. Points with false in_point should already be
                # covered by the above iteration over segments.
                d, cp, in_poly = cg.dist_points_polygon(of.p, f.p)
                for di, cpi, ip in zip(d, cp, in_poly):
                    # Closest point on segment is covered above
                    if not ip:
                        continue
                    if di < h_ideal:
                        pass


    def distance_point_segment(self):
        pass

    def distance_segment_segment(self):
        # What happens if two segments are close?
        pass

    def proximate_fractures(self):
        # Find fractures with center distance less than the sum of their major
        # axis + 2 * force length
        return fracture_pairs

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
            loc_pt_id = point_counter + np.arange(f.p.shape[1],
                                                  dtype='int')
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
        data['Fracture_Number'] = 1 + np.arange(len(self._fractures))

        for name, data in data.items():
            data_vtk = vtk_np.numpy_to_vtk(data.ravel(order='F'), deep=True,
                                           array_type=vtk.VTK_DOUBLE)
            data_vtk.SetName(name)
            data_vtk.SetNumberOfComponents(1 if data.ndim == 1 else 3)
            network_vtk.GetCellData().AddArray(data_vtk)

        writer.Update()

    def to_gmsh(self, file_name, **kwargs):
        p = self.decomposition['points']
        edges = self.decomposition['edges']
        edges_2_fracs = self.decomposition['edges_2_frac']

        poly = self._poly_2_segment()

        edge_tags, intersection_points = self._classify_edges(poly)
        edges = np.vstack((self.decomposition['edges'], edge_tags))

        self.zero_d_pt = intersection_points

        if 'mesh_size' in kwargs.keys():
            mesh_size, mesh_size_bound = \
                determine_mesh_size(self.decomposition['points'],
                                    **kwargs['mesh_size'])
        else:
            mesh_size = None
            mesh_size_bound = None

        # The tolerance applied in gmsh should be consistent with the tolerance
        # used in the splitting of the fracture network. The documentation of
        # gmsh is not clear, but it seems gmsh scales a given tolerance with
        # the size of the domain - presumably by the largest dimension. To
        # counteract this, we divide our (absolute) tolerance self.tol with the
        # domain size.
        dx = np.array([[self.domain['xmax'] - self.domain['xmin']],
                       [self.domain['ymax'] - self.domain['ymin']],
                       [self.domain['zmax'] - self.domain['zmin']]])
        gmsh_tolerance = self.tol / dx.max()

        writer = GmshWriter(p, edges, polygons=poly, domain=self.domain,
                            intersection_points=intersection_points,
                            mesh_size_bound=mesh_size_bound,
                            mesh_size=mesh_size, tolerance=gmsh_tolerance,
                            edges_2_frac=self.decomposition['line_in_frac'])

        writer.write_geo(file_name)
