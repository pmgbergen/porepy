"""
A module for representation and manipulations of fractures and fracture sets.

The model relies heavily on functions in the computational geometry library.

"""
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy

from compgeom import basics as cg
from utils import setmembership


class Fracture(object):

    def __init__(self, points, index=None):
        self.p = points
        self.center = np.mean(self.p, axis=1).reshape((-1, 1))

        self.orig_p = self.p.copy()

        self.index = index
#         self.rot = cg.project_plane_matrix(p)

    def set_index(self, i):
        self.index = i


    def __eq__(self, other):
        return self.index == other.index

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
            yield self.p[:, np.array([i, i+1]) % sz]


    def points_2_ccw(self):
        """
        Ensure that the points are sorted in a counter-cyclical order .

        Implementation note:
            For now, the ordering of nodes in based on a simple angle argument.
            This will not be robust for general point clouds, but we expect the
            fractures to be regularly shaped in this sense.

        """
        # First rotate coordinates to the plane
        rotation = cg.project_plane_matrix(self.p - self.center)

        points_2d = rotation.dot(self.p)
        
        theta = np.arctan2(points_2d[1], points_2d[0])
        sort_ind = np.argsort(theta)
        
        
        self.p = self.p[:, sort_ind]


    def add_points(self, p, check_convexity=True):
        """
        Add a point to the polygon, and enforce ccw sorting.

        Parameters:
            p (np.ndarray, 3xn): Points to add
            check_convexity (boolean, optional): Verify that the polygon is
                convex. Defaults to true
        
        Return:
            boolean, true if the resulting polygon is convex.

        """
        self.p = np.hstack((self.p, p))
        self.points_2_ccw()

        return self.check_convexity()
            

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
        return self.as_sp_polygon().is_convex()

    
    def as_sp_polygon(self):
        sp = [sympy.geometry.Point(self.p[:, i]) for  i in range(self.p.shape[1])]
        return sympy.geometry.Polygon(*sp)

    
    def _sympy_2_np(p):
        # Convert sympy points to numpy format. If more than one point, these should be sent as a list
        if isinstance(p, list):
            return np.array(list([i.args for i in p]), dtype='float').transpose()
        else:
            return np.array(list(p.args), dtype='float').reshape((-1, 1))

    def intersects(self, other):
        """
        Find intersections between self and another polygon.

        The function checks for both intersection between the interior of one
        polygon with the boundary of the other, and pure boundary
        intersections. Intersection types supported so far are
            X (full intersection, possibly involving boundaries)
            L (Two fractures intersect at a boundary)

        Not yet supported is T (one fracture ends with a segment, or possibly a
        point in the plane of another). Patterns involving more than two
        fractures need to be delt with by someone else (see FractureSet)


        """
        
        # Find intersections between self and other. Note that the algorithms
        # for intersections are not fully reflexive in terms of argument
        # order, so we need to do two tests.

        ####
        # First, check for intersection between interior of one polygon with
        # segment of the other.
        
        # Array for intersections with the interior of one polygon (as opposed
        # to full boundary intersection, below)
        int_points = np.empty((3, 0))
        
        # Keep track of whether the intersection points are on the boundary of
        # the polygons.
        on_boundary_self = []
        on_boundary_other = []

        # Compute intersections, with both polygons as first argument
        isect_self_other = cg.polygon_segment_intersect(self.p, other.p)
        isect_other_self = cg.polygon_segment_intersect(other.p, self.p)
        
        # Process data
        if isect_self_other is not None:
            int_points = np.hstack((int_points, isect_self_other))

            # An intersection between self and other (in that order) is defined
            # as being between interior of self and boundary of other. See
            # polygon_segment_intersect for details.
            on_boundary_self += [False for i in range(isect_self_other.shape[1])]
            on_boundary_other += [True for i in range(isect_self_other.shape[1])]

        if isect_other_self is not None:
            int_points = np.hstack((int_points, isect_self_other))

            # Interior of other intersected by boundary of self
            on_boundary_self += [True for i in range(isect_other_self.shape[1])]
            on_boundary_other += [False for i in
                                    range(isect_other_self.shape[1])]

        # There should be at most two of these points
        assert int_points.shape[1] <= 2
            
        # Note: If int_points.shape[1] == 2, we can simply cut it short here,
        # as there should be no more than two intersection points for a convex
        # polygen. However, we do not implement this before the method has been
        # thoroughly tested.
        
        ####
        # Next, check for intersections between the polygon boundaries
        bound_sect_self_other = cg.polygon_boundaries_intersect(self.p, other.p)
        bound_sect_other_self = cg.polygon_boundaries_intersect(other.p, self.p)
        
        # Short cut: If no boundary intersections, we return the interior points
        if len(bound_sect_self_other) == 0 and len(bound_sect_other_self) == 0:
            # None of the intersection points lay on the boundary
            return int_points, on_boundary_self, on_boundary_other
        
        # Else, we have boundary intersection, and need to process them
        bound_pt_self, \
            self_segment, \
            self_non_vertex, \
            self_cuts_through = self._process_segment_isect(bound_sect_self_other, self.p)

        bound_pt_other, \
            other_segment, \
            other_non_vertex, \
            other_cuts_through = self._process_segment_isect(bound_sect_other_self, other.p)
        
        # Run some sanity checks        
        
        # Convex polygons can intersect each other in at most two points (and a line inbetween)
        if int_points.shape[1] > 1:
            assert bound_pt_self.shape[1] == 0 and bound_pt_other.shape[1] == 0
        elif int_points.shape[1] == 1:
            assert (bound_pt_self.shape[1] + bound_pt_other.shape[1]) == 1
        
        # If a segment runs through the polygon, there should be no interior points.
        # This may be sensitive to tolerances, should be a useful test to gauge that.
        if self_cuts_through or other_cuts_through:
            assert int_points.shape[1] == 0
            assert not self_segment
            assert not other_segment
        
        # Storage for intersection points located on the boundary
        bound_pt = []
        
        # Cover the case of a segment - essentially an L-intersection
        if self_segment:
            assert other_segment  # This should be reflexive
            assert bound_pt.shape[1] == 2
            assert np.allclose(bound_pt_self, bound_pt_other)
            on_boundary_self == [True, True]
            on_boundary_other = [True, True]
            return bound_pt_self, on_boundary_self, on_boundary_other
        
        # Case of cutting
        if self_cuts_through or other_cuts_through:
            # Do not expect this yet, corresponds to one kind of a a
            # T-intersection (vertexes embedded in plane, but not fully cutting
            # is another possibility).
            raise NotImplemented()
        
        # By now, there should be a single member of bound_pt
        assert bound_pt_self.shape[1] == 1 or bound_pt_self.shape[1] == 2
        assert bound_pt_other.shape[1] == 1 or bound_pt_other.shape[1] == 2
        bound_pt = np.hstack((int_points, bound_pt_self))
        on_boundary_self += [False for i in range(bound_pt_self.shape[1])]
        on_boundary_other += [True for i in range(bound_pt_self.shape[1])]
        
        return bound_pt, on_boundary_self, on_boundary_other


    def _process_segment_isect(self, isect_bound, poly):
        """
        Helper function to interpret result from polygon_boundaries_intersect

        Classify an intersection between segments, from the perspective of a polygon, as
        i) Sharing a segment
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

            ip_unique, *rest = setmembership.unique_columns_tol(ip)
            if ip_unique.shape[1] == 2:
                # The polygons share a segment, or a 
                bound_pt.append(ip_unique)
                has_segment = True
                # No need to deal with non_vertex here, there should be no more intersections (convex, non-overlapping polygons)
                non_vertex = None
            elif ip_unique.shape[1] == 1:
                # Either a vertex or single intersection
                poly_ext, *rest = setmembership.unique_columns_tol(np.hstack((self.p, ip_unique)))
                if poly_ext.shape[1] == self.p.shape[1]:
                    # This is a vertex, 
                    # For L-intersections, we may get a bunch of these, but they will be disgarded elsewhere
                    bound_pt = np.hstack((bound_pt, ip_unique))
                    non_vertex.append(False)
                else:
                    # New point contact
                    bound_pt = np.hstack((bound_pt, ip_unique))
                    non_vertex.append(True)
            else:
                # This should never happen for convex polygons
                raise ValueError('Should not have more than two intersection points')

        # Now, if any segment has been found more than once, it cuts two segments of the other polygon
        cuts_two = np.any(num_occ > 1)

        # Sanity check
        assert num_occ.max() < 3

        return bound_pt, has_segment, non_vertex, cuts_two

    
    def segments(self):
        return self.as_sp_polygon().sides
        
    
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
        ax.plot(x, y ,z)
        return ax
        
#--------------------------------------------------------------------        

class EllipticFracture(Fracture):
    
    def __init__(self, major_axis, minor_axis, center, angles, num_points=8):
        self.center = center 
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        # 
        self.angles = angles
        
        self.num_pt = num_points
        self.p = self.center.transpose()[:, np.newaxis] + self.rot.dot(self._populate())
        self.grid_size = 1
      
    def _populate(self):
        angs = np.linspace(0, 2*pi, self.num_pt+1, endpoint=True)[:-1]
        x = self.major_axis * np.cos(angs)
        y = self.minor_axis * np.sin(angs)
        z = np.zeros_like(angs)
        return np.vstack((x, y, z))
        
    def segments(self):
        return utils.sort_point_lines()


#-------------------------------------------------------------------------------


class Intersection(object):
    
    def __init__(self, first, second, coord):
        self.first = first
        self.second = second
        self.coord = coord


    def __repr__(self):
        s = 'Intersection between fractures ' + str(self.first.index) + ' and ' + \
            str(self.second.index) + '\n'
        s += 'Intersection points: \n'
        for i in range(self.coord.shape[1]):
            s += '(' + str(self.coord[0, i]) + ', ' + str(self.coord[1, i]) + ', ' + \
                 str(self.coord[2, i]) + ') \n'
        return s

    


class FractureSet(object):
    
    def __init__(self, fractures=None):
        
        if fractures is None:
            self._fractures = fractures
        else:
            self._fractures = fractures

        for i, f in enumerate(self._fractures):
            f.set_index(i)
    
        self.intersections = []

        self._has_checked_intersections = False


    def add(self, f):
        f.set_index(len(self._fractures))
        self._fractures.append(f)

    def __getitem__(self, position):
        return self._fractures[position]


    def get_intersections(self, frac):
        if frac is None:
            frac = np.arange(len(self._fractures))
        if isinstance(frac, np.ndarray):
            frac = list(frac)
        elif isinstance(frac, int):
            frac = [frac]
        isect = []
        for fi in frac:
            f = self[fi]
            isect_loc = []
            for i in self.intersections:
                if i.first.index == f.index or i.second.index == f.index:
                    isect_loc.append(i)
            isect.append(isect_loc)
        return isect
        
    def find_intersections(self, use_orig_points=True):

        # Before searching for intersections, reset the fracture polygons to
        # their original state. If this is not done, multiple seacrhes for
        # intersections will find points that are already vertexes (from the
        # previous call). A more robust implementation of the intersection
        # finder may take care of this, but for the moment, we go for this.
        if use_orig_points:
            for f in self._fractures:
                f.p = f.orig_p

        for i, first in enumerate(self._fractures):
            for j in range(i+1, len(self._fractures)):
                second = self._fractures[j]
                isect, bound_first, bound_second = first.intersects(second)
                if len(isect) > 0:
                    self.intersections.append(Intersection(first, second,
                    isect))
                    first.add_points(isect[:, np.where(bound_first)[0]])
                    second.add_points(isect[:, np.where(bound_second)[0]])

    
    def __repr__(self):
        s = 'Fracture set with ' + str(len(self._fractures)) + ' fractures'
        return s
    
    def plot_fractures(self, ind=None):
        if ind is None:
            ind = np.arange(len(self.fractures))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for f in self._fractures:
            f.plot_frame(ax)
        return fig
                    
        
    def close_points():
        for p1, p2 in zip(proximate_fractures):
            pass
            # Check both point-point proximity and point-fracture-plane
            
    def distance_point_segment():
        pass
    
    def distance_segment_segment():
        # What happens if two segments are close?
        pass
        
    def proximate_fractures(self):
        # Find fractures with center distance less than the sum of their major axis + 2 * force length
        return fracture_pairs
    
