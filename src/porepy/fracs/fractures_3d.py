"""
A module for representation and manipulations of fractures and fracture sets.

The model relies heavily on functions in the computational geometry library.

"""
import copy
import csv
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import meshio
import numpy as np

import porepy as pp
from porepy.utils import setmembership, sort_points

from .gmsh_interface import GmshData3d, GmshWriter, Tags

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["gridding"]


class Fracture(object):
    """Class representing a single fracture, as a convex, planar 2D object
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

    @pp.time_logger(sections=module_sections)
    def __init__(self, points, index=None, check_convexity=False):
        """Initialize fractures.

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

    @pp.time_logger(sections=module_sections)
    def set_index(self, i):
        """Set index of this fracture.

        Parameters:
            i (int): Index.

        """
        self.index = i

    @pp.time_logger(sections=module_sections)
    def __eq__(self, other):
        """Equality is defined as two fractures having the same index.

        There are issues with this, behavior may change in the future.

        """
        return self.index == other.index

    @pp.time_logger(sections=module_sections)
    def copy(self):
        """Return a deep copy of the fracture.

        Note that the original points (as given when the fracture was
        initialized) will *not* be preserved.

        Returns:
            Fracture with the same points.

        """
        p = np.copy(self.p)
        return Fracture(p)

    @pp.time_logger(sections=module_sections)
    def points(self):
        """
        Iterator over the vexrtexes of the bounding polygon

        Yields:
            np.array (3 x 1): polygon vertexes

        """
        for i in range(self.p.shape[1]):
            yield self.p[:, i].reshape((-1, 1))

    @pp.time_logger(sections=module_sections)
    def segments(self):
        """
        Iterator over the segments of the bounding polygon.

        Yields:
            np.array (3 x 2): polygon segment
        """

        sz = self.p.shape[1]
        for i in range(sz):
            yield self.p[:, np.array([i, i + 1]) % sz]

    @pp.time_logger(sections=module_sections)
    def is_vertex(self, p, tol=1e-4):
        """Check whether a given point is a vertex of the fracture.

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

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def add_points(self, p, check_convexity=True, tol=1e-4, enforce_pt_tol=None):
        """
        Add a point to the polygon with ccw sorting enforced.

        Always run a test to check that the points are still planar. By
        @pp.time_logger(sections=module_sections)
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
                np.zeros(self.p.shape[1], dtype=bool),
                np.ones(p.shape[1], dtype=bool),
            )
        )
        self.p = np.hstack((self.p, p))
        self.p, _, _ = setmembership.unique_columns_tol(self.p, tol=tol)

        # Sort points to ccw
        mask = self.points_2_ccw()

        if enforce_pt_tol is not None:
            to_enforce = np.where(to_enforce[mask])[0]
            dist = pp.distances.pointset(self.p)
            dist /= np.amax(dist)
            np.fill_diagonal(dist, np.inf)
            mask = np.where(dist < enforce_pt_tol)[0]
            mask = np.setdiff1d(mask, to_enforce, assume_unique=True)
            self.remove_points(mask)

        if check_convexity:
            return self.check_convexity() and self.is_planar(tol)
        else:
            return self.is_planar()

    @pp.time_logger(sections=module_sections)
    def remove_points(self, ind, keep_orig=False):
        """Remove points from the fracture definition

        Parameters:
            ind (np array-like): Indices of points to remove.
            keep_orig (boolean, optional): Whether to keep the original points
                in the attribute orig_p. Defaults to False.

        """
        self.p = np.delete(self.p, ind, axis=1)
        if not keep_orig:
            self.orig_p = self.p

    @pp.time_logger(sections=module_sections)
    def plane_coordinates(self):
        """
        Represent the vertex coordinates in its natural 2d plane.

        The plane does not necessarily have the third coordinate as zero (no
        translation to the origin is made)

        Returns:
            np.array (2xn): The 2d coordinates of the vertexes.

        """
        rotation = pp.map_geometry.project_plane_matrix(self.p)
        points_2d = rotation.dot(self.p)

        return points_2d[:2]

    @pp.time_logger(sections=module_sections)
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
        if self.p.shape[1] == 3:
            # A triangle is always convex
            return True

        p_2d = self.plane_coordinates()
        return self.as_sp_polygon(p_2d).is_convex()

    @pp.time_logger(sections=module_sections)
    def is_planar(self, tol=1e-4):
        """Check if the points forming this fracture lies in a plane.

        Parameters:
            tol (double): Tolerance for non-planarity. Treated as an absolute
                quantity (no scaling with fracture extent)

        Returns:
            boolean, True if the polygon is planar. False if not.
        """
        p = self.p - np.mean(self.p, axis=1).reshape((-1, 1))
        rot = pp.map_geometry.project_plane_matrix(p)
        p_2d = rot.dot(p)
        return np.max(np.abs(p_2d[2])) < tol

    @pp.time_logger(sections=module_sections)
    def compute_centroid(self):
        """
        Compute, and redefine, center of the fracture in the form of the
        centroid.

        The method assumes the polygon is convex.

        """
        # Rotate to 2d coordinates
        rot = pp.map_geometry.project_plane_matrix(self.p)
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

    @pp.time_logger(sections=module_sections)
    def compute_normal(self):
        """Compute normal to the polygon."""
        self.normal = pp.map_geometry.compute_normal(self.p)[:, None]

    @pp.time_logger(sections=module_sections)
    def as_sp_polygon(self, p=None):
        """Represent polygon as a sympy object.

        Parameters:
            p (np.array, nd x npt, optional): Points for the polygon. Defaults
                to None, in which case self.p is used.

        Returns:
            sympy.geometry.Polygon: Representation of the polygon formed by p.

        """
        from sympy.geometry import Point, Polygon

        if p is None:
            p = self.p

        sp = [Point(p[:, i]) for i in range(p.shape[1])]
        return Polygon(*sp)

    @pp.time_logger(sections=module_sections)
    def __repr__(self):
        return self.__str__()

    @pp.time_logger(sections=module_sections)
    def __str__(self):
        s = "Points: \n"
        s += str(self.p) + "\n"
        s += "Center: " + str(self.center)
        return s


class EllipticFracture(Fracture):
    """
    Subclass of Fractures, representing an elliptic fracture.

    See parent class for description.

    """

    @pp.time_logger(sections=module_sections)
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

        assert pp.geometry_property_checks.points_are_planar(ref_pts)

        # Rotate reference points so that the major axis has the right
        # orientation
        major_axis_rot = pp.map_geometry.rotation_matrix(major_axis_angle, [0, 0, 1])
        rot_ref_pts = major_axis_rot.dot(ref_pts)

        assert pp.geometry_property_checks.points_are_planar(rot_ref_pts)

        # Then the dip
        # Rotation matrix of the strike angle
        strike_rot = pp.map_geometry.rotation_matrix(strike_angle, np.array([0, 0, 1]))
        # Compute strike direction
        strike_dir = strike_rot.dot(np.array([1, 0, 0]))
        dip_rot = pp.map_geometry.rotation_matrix(dip_angle, strike_dir)

        dip_pts = dip_rot.dot(rot_ref_pts)

        assert pp.geometry_property_checks.points_are_planar(dip_pts)

        # Set the points, and store them in a backup.
        self.p = center + dip_pts
        self.orig_p = self.p.copy()

        # Compute normal vector
        self.normal = pp.map_geometry.compute_normal(self.p)[:, None]

        assert pp.geometry_property_checks.points_are_planar(self.orig_p, self.normal)


class FractureNetwork3d(object):
    """
    Collection of Fractures with geometrical information. Facilitates
    computation of intersections of the fracture. Also incorporates the
    bounding box of the domain. To ensure
    that all fractures lie within the box, call impose_external_boundary()
    _after_ all fractures have been specified.

    Attributes:
        _fractures (list of Fracture): All fractures forming the network.
        intersections (list of Intersection): All known intersections in the
            network.
        has_checked_intersections (boolean): If True, the intersection finder
            method has been run. Useful in meshing algorithms to avoid
            recomputing known information.
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

    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        fractures: Optional[List[Fracture]] = None,
        domain: Optional[Union[Dict[str, float], List[np.ndarray]]] = None,
        tol: float = 1e-8,
        run_checks: bool = False,
    ) -> None:
        """Initialize fracture network.

        Generate network from specified fractures. The fractures will have
        their index set (and existing values overridden), according to the
        ordering in the input list.

        Initialization sets most fields (see attributes) to None.

        Parameters:
            fractures (list of Fracture, optional): Fractures that make up the network.
                Defaults to None, which will create a domain empty of fractures.
            domain (either dictionary or list of np.arrays): Domain specification. See
                self.impose_external_boundary() for details.
            tol (double, optional): Tolerance used in geometric tolerations. Defaults
                to 1e-8.
            run_checks (boolean, optional): Run consistency checks during the network
                processing. Can be considered a limited debug mode. Defaults to False.

        """
        self._fractures = []

        if fractures is not None:
            for f in fractures:
                self._fractures.append(f.copy())

        for i, f in enumerate(self._fractures):
            f.set_index(i)

        # Store intersection information as a dictionary. Keep track of
        # the intersecting fractures, the start and end point of the intersection line,
        # and whether the intersection is on the boundary of the fractures.
        # Note that a Y-type intersection between three fractures is represented
        # as three intersections.
        self.intersections: Dict[str, np.ndarray] = {
            "first": np.array([], dtype=object),
            "second": np.array([], dtype=object),
            "start": np.zeros((3, 0)),
            "end": np.zeros((3, 0)),
            "bound_first": np.array([], dtype=bool),
            "bound_second": np.array([], dtype=bool),
        }

        self.has_checked_intersections = False
        self.tol = tol
        self.run_checks = run_checks

        # Initialize with an empty domain. Can be modified later by a call to
        # 'impose_external_boundary()'
        self.domain = domain

        # Initialize mesh size parameters as empty
        self.mesh_size_min = None
        self.mesh_size_frac = None
        self.mesh_size_bound = None
        # Assign an empty tag dictionary
        self.tags: Dict[str, List[bool]] = {}

        # No auxiliary points have been added
        self.auxiliary_points_added = False

        self.bounding_box_imposed = False

    @pp.time_logger(sections=module_sections)
    def add(self, f):
        """Add a fracture to the network.

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

    @pp.time_logger(sections=module_sections)
    def copy(self):
        """Create deep copy of the network.

        The method will create a deep copy of all fractures, as well as the domain, of
        the network. Note that if the fractures have had extra points imposed as part
        of a meshing procedure, these will included in the copied fractures.

        Returns:
            pp.FractureNetwork3d.

        """
        fracs = [f.copy() for f in self._fractures]

        domain = self.domain
        if domain is not None:
            # Get a deep copy of domain, but no need to do that if domain is None
            domain = copy.deepcopy(domain)

        return FractureNetwork3d(fracs, domain, self.tol)

    @pp.time_logger(sections=module_sections)
    def mesh(
        self,
        mesh_args,
        dfn=False,
        file_name=None,
        constraints=None,
        write_geo=False,
        **kwargs,
    ):
        """Mesh the fracture network, and generate a mixed-dimensional grid.

        The mesh itself is generated by Gmsh.

        Parameters:
            mesh_args (dict): Should contain fields 'mesh_size_frac', 'mesh_size_min',
                which represent the ideal mesh size at the fracture, and the
                minimum mesh size passed to gmsh. Can also contain
                'mesh_size_bound', wihch gives the far-field (boundary) mehs
                size.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            file_name (str, optional): Name of file used to communicate with gmsh.
                @pp.time_logger(sections=module_sections)
                defaults to gmsh_frac_file. The gmsh configuration file will be
                file_name.geo, while the mesh is dumped to file_name.msh.
            constraints (np.array): Index list of elements in the fracture list that
                should be treated as constraints in meshing, but not added as separate
                fracture grids (no splitting of nodes etc).

        Returns:
            GridBucket: Mixed-dimensional mesh.

        """
        if file_name is None:
            file_name = "gmsh_frac_file.msh"

        gmsh_repr = self.prepare_for_gmsh(mesh_args, dfn, constraints)

        gmsh_writer = GmshWriter(gmsh_repr)

        if dfn:
            dim_meshing = 2
        else:
            dim_meshing = 3

        gmsh_writer.generate(file_name, dim_meshing, write_geo=write_geo)

        if dfn:
            grid_list = pp.fracs.simplex.triangle_grid_embedded(file_name)
        else:
            # Process the gmsh .msh output file, to make a list of grids
            grid_list = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
                file_name, constraints
            )

        # Merge the grids into a mixed-dimensional GridBucket
        gb = pp.meshing.grid_list_to_grid_bucket(grid_list, **kwargs)
        return gb

    @pp.time_logger(sections=module_sections)
    def prepare_for_gmsh(self, mesh_args, dfn=False, constraints=None) -> GmshData3d:
        """Process network intersections and write a gmsh .geo configuration file,
        ready to be processed by gmsh.

        NOTE: Consider to use the mesh() function instead to get a ready GridBucket.

        Parameters:
            mesh_args (dict): Should contain fields 'mesh_size_frac', 'mesh_size_min',
                which represent the ideal mesh size at the fracture, and the
                minimum mesh size passed to gmsh. Can also contain
                'mesh_size_bound', wihch gives the far-field (boundary) mehs
                size.
            dfn (boolean, optional): If True, a DFN mesh (of the network, but not
                the surrounding matrix) is created.
            file_name (str, optional): Name of file used to communicate with gmsh.
                @pp.time_logger(sections=module_sections)
                defaults to gmsh_frac_file. The gmsh configuration file will be
                file_name.geo, while the mesh is dumped to file_name.msh.
            constraints (np.array): Index list of elements in the fracture list that
                should be treated as constraints in meshing, but not added as separate
                fracture grids (no splitting of nodes etc).

        Returns:
            str: Name of .geo file.

        """

        # The implementation in this function is fairly straightforward, all
        # technical difficulties are hidden in other functions.

        # Impose the boundary of the domain. This may cut and split fractures.
        # FIXME: When fractures are split or deleted, there will likely be
        # mapping errors between fracture indices.
        if not dfn and not self.bounding_box_imposed:
            self.impose_external_boundary(self.domain)

        if constraints is None:
            constraints = np.array([], dtype=int)

        # Find intersections between fractures
        if not self.has_checked_intersections:
            self.find_intersections()
        else:
            logger.info("Use existing intersections")

        if "mesh_size_frac" not in mesh_args.keys():
            raise ValueError("Meshing algorithm needs argument mesh_size_frac")
        if "mesh_size_min" not in mesh_args.keys():
            raise ValueError("Meshing algorithm needs argument mesh_size_min")

        # Insert auxiliary points for mesh size control. This is done after the
        # intersections are found, but before the fractures are collected into a
        # set of edges (see split_intersecions).
        mesh_size_frac = mesh_args.get("mesh_size_frac", None)
        mesh_size_min = mesh_args.get("mesh_size_min", None)
        mesh_size_bound = mesh_args.get("mesh_size_bound", None)
        self._insert_auxiliary_points(mesh_size_frac, mesh_size_min, mesh_size_bound)

        # Process intersections to get a description of the geometry in non-
        # intersecting lines and polygons
        self.split_intersections()

        # Having found all intersections etc., the next step is to classify the geometric
        # objects before representing them in the data format expected by the Gmsh interface.
        # The classification is somewhat complex, since, for certain applications, it is
        # necessary with a detailed description of different objects.

        # Extract geometrical information.
        p = self.decomposition["points"]
        edges = self.decomposition["edges"]
        poly = self._poly_2_segment()

        has_boundary = "boundary" in self.tags

        # Get preliminary set of tags for the edges. Also find which edges are
        # interior to all or only some edges
        edge_tags, not_boundary_edge, some_boundary_edge = self._classify_edges(
            poly, constraints
        )

        # All intersection lines and points on boundaries are non-physical in 3d.
        # I.e., they are assigned boundary conditions, but are not gridded. Hence:
        # Remove the points and edges at the boundary
        point_tags, edge_tags = self._on_domain_boundary(edges, edge_tags)

        # To ensure that polygons that are constraints, but not fractures, do not
        # trigger the generation of 1d fracture intersections, or 0d point grids, some
        # steps are needed.

        # Count the number of times a lines is defined as 'inside' a fracture, and the
        # the number of times this is casued by a constraint
        in_frac_occurences = np.zeros(edges.shape[1], dtype=int)
        in_frac_occurences_by_constraints = np.zeros(edges.shape[1], dtype=int)

        for poly_ind, poly_edges in enumerate(self.decomposition["line_in_frac"]):
            for edge_ind in poly_edges:
                in_frac_occurences[edge_ind] += 1
                if poly_ind in constraints:
                    in_frac_occurences_by_constraints[edge_ind] += 1

        # Count the number of occurences that are not caused by a constraint
        num_occ_not_by_constraints = (
            in_frac_occurences - in_frac_occurences_by_constraints
        )

        # If all but one occurence of a line internal to a polygon is caused by
        # constraints, this is likely a fracture.
        auxiliary_line = num_occ_not_by_constraints == 1

        # .. However, the line may also be caused by a T- or L-intersection
        auxiliary_line[some_boundary_edge] = False
        # The edge tags for internal lines were set accordingly in self._classify_edges.
        # Update to auxiliray line if this was really what we had.
        edge_tags[auxiliary_line] = Tags.AUXILIARY_LINE.value

        # .. and we're done with edges (until someone defines a new special case)
        # Next, find intersection points.

        # Count the number of times a point is referred to by an intersection
        # between two fractures. If this is more than one, the point should
        # have a 0-d grid assigned to it.
        # Here, we must account for lines that are internal, but not those that have
        # been found to be triggered by constraints.
        intersection_edge = np.logical_and(
            not_boundary_edge, np.logical_not(auxiliary_line)
        )

        # All intersection points should occur at least twice
        isect_p = edges[:2, intersection_edge].ravel()
        num_occ_pt = np.bincount(isect_p)
        intersection_point_candidates = np.where(num_occ_pt > 1)[0]

        # .. however, this is not enough: If a fracture and a constraint intersect at
        # a domain boundary, the candidate is not a true intersection point.
        # Loop over all candidates, find all polygons that have this as part of an edge,
        # and count the number of those polygons that are fractures (e.g. not boundary
        # or constraint). If there is more than one, this is indeed  a fracture intersection
        # and an intersection point grid should be assigned.
        # (Reason for more than one, not two, I believe is: Two fractures crossing will not
        # make a 0d point by themselves). Point must then come either from a third fracture,
        # or a constraint. In the latter case, we anyhow need the intersection point as a
        # grid to get dynamics between the intersection line correctly represented.
        intersection_points = []
        # Points that are both on a boundary, and on a fracture. This may represent one of
        # a few cases: A fracture-constraint intersection on a boundary, and/or a fracture
        # crossing a domain surface boundary.
        fracture_and_boundary_points = []

        for pi in intersection_point_candidates:
            # edges of this point
            _, edge_ind = np.where(edges == pi)

            # Fractures of the edge
            frac_arr = np.array([], dtype=int)
            for e in edge_ind:
                frac_arr = np.append(frac_arr, self.decomposition["edges_2_frac"][e])
            # Uniquify.
            unique_fracs = np.unique(frac_arr)

            # If the domain has a boundary, classify the polygons of this point as
            # fracture, fracture or boundary (and tacitly, constraints)
            if has_boundary:
                is_boundary = np.array(self.tags["boundary"])[unique_fracs]
                is_frac = np.logical_not(
                    np.logical_or(np.in1d(unique_fracs, constraints), is_boundary)
                )
                is_frac_or_boundary = np.logical_not(
                    np.in1d(unique_fracs, constraints),
                )
            else:
                is_frac = np.logical_not(np.in1d(unique_fracs, constraints))
                is_frac_or_boundary = is_frac

            # If more than one fracture share this point, it is an intersection point.
            # See comment above on why not > 2.
            if is_frac.sum() > 1:
                intersection_points.append(pi)
            # If the point is on a fracture, and on a boundary surface, it will be
            # classified otherwise.
            if is_frac.sum() > 0 and (is_frac_or_boundary.sum() - is_frac.sum()) > 0:
                fracture_and_boundary_points.append(pi)

        # Finally, we have the full set of intersection points (take a good laugh when finding
        # this line in the next round of debugging).

        # Candidates that were not intersections
        fracture_constraint_intersection = np.setdiff1d(
            intersection_point_candidates, fracture_and_boundary_points
        )
        # Special tag for intersection between fracture and constraint.
        # These are not needed in the gmsh postprocessing (will not produce 0d grids),
        # but it can be useful to mark them for other purposes (EK: DFM upscaling)
        point_tags[
            fracture_constraint_intersection
        ] = Tags.FRACTURE_CONSTRAINT_INTERSECTION_POINT.value

        point_tags[fracture_and_boundary_points] = Tags.FRACTURE_BOUNDARY_POINT.value

        # We're done! Hurah!

        # Find points tagged as on the domain boundary
        boundary_points = np.where(point_tags == Tags.DOMAIN_BOUNDARY_POINT.value)[0]

        fracture_boundary_points = np.where(
            point_tags == Tags.FRACTURE_BOUNDARY_POINT.value
        )[0]

        # Intersections on the boundary should not have a 0d grid assigned
        true_intersection_points = np.setdiff1d(
            intersection_points, np.hstack((boundary_points, fracture_boundary_points))
        )

        edges = np.vstack((self.decomposition["edges"], edge_tags))

        # Obtain mesh size parameters
        mesh_size = self._determine_mesh_size(point_tags=point_tags)

        line_in_poly = self.decomposition["line_in_frac"]

        physical_points: Dict[int, Tags] = {}
        for pi in fracture_boundary_points:
            physical_points[pi] = Tags.FRACTURE_BOUNDARY_POINT

        for pi in fracture_constraint_intersection:
            physical_points[pi] = Tags.FRACTURE_CONSTRAINT_INTERSECTION_POINT

        for pi in boundary_points:
            physical_points[pi] = Tags.DOMAIN_BOUNDARY_POINT

        for pi in true_intersection_points:
            physical_points[pi] = Tags.FRACTURE_INTERSECTION_POINT

        # Use separate structures to store tags and physical names for the polygons.
        # The former is used for feeding information into gmsh, while the latter is
        # used to tag information in the output from gmsh. They are kept as separate
        # variables since a user may want to modify the physical surfaces before
        # generating the .msh file.
        physical_surfaces: Dict[int, Tags] = {}
        polygon_tags = np.zeros(len(self._fractures), dtype=int)

        for fi, _ in enumerate(self._fractures):
            if has_boundary and self.tags["boundary"][fi]:
                physical_surfaces[fi] = Tags.DOMAIN_BOUNDARY_SURFACE
                polygon_tags[fi] = Tags.DOMAIN_BOUNDARY_SURFACE.value
            elif fi in constraints:
                physical_surfaces[fi] = Tags.AUXILIARY_PLANE
                polygon_tags[fi] = Tags.AUXILIARY_PLANE.value
            else:
                physical_surfaces[fi] = Tags.FRACTURE
                polygon_tags[fi] = Tags.FRACTURE.value

        physical_lines = {}
        for ei in range(edges.shape[1]):
            if edges[2, ei] in (
                Tags.FRACTURE_TIP.value,
                Tags.FRACTURE_INTERSECTION_LINE.value,
                Tags.DOMAIN_BOUNDARY_LINE.value,
                Tags.FRACTURE_BOUNDARY_LINE.value,
                Tags.AUXILIARY_LINE.value,
            ):
                physical_lines[ei] = edges[2, ei]

        gmsh_repr = GmshData3d(
            dim=3,
            pts=p,
            lines=edges,
            mesh_size=mesh_size,
            polygons=poly,
            physical_surfaces=physical_surfaces,
            polygon_tags=polygon_tags,
            lines_in_surface=line_in_poly,
            physical_points=physical_points,
            physical_lines=physical_lines,
        )

        self.decomposition["edge_tags"] = edges[2]
        self.decomposition["domain_boundary_points"] = boundary_points
        self.decomposition["point_tags"] = point_tags

        return gmsh_repr

    @pp.time_logger(sections=module_sections)
    def __getitem__(self, position):
        return self._fractures[position]

    @pp.time_logger(sections=module_sections)
    def intersections_of_fracture(
        self, frac: Union[int, Fracture]
    ) -> Tuple[List[int], List[bool]]:
        """Get all known intersections for a fracture.

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

        isects = []
        is_first = []
        for i in range(len(self.intersections["first"])):
            if self.intersections["first"][i].index == fi:
                isects.append(i)
                is_first.append(True)
            elif self.intersections["second"][i].index == fi:
                isects.append(i)
                is_first.append((False))

        return isects, is_first

    @pp.time_logger(sections=module_sections)
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

        # Intersections are found using a method in the comp_geom module, which requires
        # the fractures to be represented as a list of polygons.
        polys = [f.p for f in self._fractures]
        # Obtain intersection points, indexes of intersection points for each fracture
        # information on whether the fracture is on the boundary, and pairs of fractures
        # that intersect.
        isect, point_ind, bound_info, frac_pairs, _ = pp.intersections.polygons_3d(
            polys
        )

        # Loop over all pairs of intersection pairs, add the intersections to the
        # internal list.
        for pair in frac_pairs:
            # Indices of the relevant pairs.
            ind_0, ind_1 = pair
            # Find the common indices of intersection points (referring to isect)
            # and find where they are located in the array point_ind[ind_0]
            common_ind, i0 = pp.utils.setmembership.ismember_rows(
                point_ind[ind_1], point_ind[ind_0]
            )
            # Convert from boolean indices to indices
            common_ind = np.where(common_ind)[0]
            # There should be exactly two intersection points
            assert common_ind.size == 2

            # Do the same exercise with the second fracture
            common_ind_2, i1 = pp.utils.setmembership.ismember_rows(
                point_ind[ind_0], point_ind[ind_1]
            )
            common_ind_2 = np.where(common_ind_2)[0]
            assert common_ind_2.size == 2

            # Check if the intersection points form a boundary edge of each of the
            # fractures. The indexing is a bit involved, but is based on there being
            # two intersection points for each segment - thus the indices in i0 and i1
            # must be divided by two.
            on_bound_0 = bound_info[ind_0][np.floor(i0[0] / 2).astype(int)]
            on_bound_1 = bound_info[ind_1][np.floor(i1[0] / 2).astype(int)]

            # Add the intersection to the internal storage
            self._add_intersection(
                self._fractures[ind_0],
                self._fractures[ind_1],
                isect[:, point_ind[ind_1][common_ind[0]]],
                isect[:, point_ind[ind_1][common_ind[1]]],
                bound_first=on_bound_0,
                bound_second=on_bound_1,
            )
        logger.info(
            "Found %i intersections. Ellapsed time: %.5f",
            len(self.intersections),
            time.time() - start_time,
        )

    @pp.time_logger(sections=module_sections)
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

        # By now, all segments in the grid are defined by a unique set of
        # points and edges. The next task is to identify intersecting edges,
        # and split them.
        all_p, edges, edges_2_frac, is_boundary_edge = self._remove_edge_intersections(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

        if self.run_checks:
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

            poly, _ = sort_points.sort_point_pairs(edges[:2, ei_bound])
            polygons.append(poly)
            line_in_frac.append(ei)

        self.decomposition["polygons"] = polygons
        self.decomposition["line_in_frac"] = line_in_frac

        #                              'polygons': poly_segments,
        #                             'polygon_frac': poly_2_frac}

        logger.info(
            "Finished fracture splitting after %.5f seconds", time.time() - start_time
        )

    @pp.time_logger(sections=module_sections)
    def _fracs_2_edges(self, edges_2_frac):
        """Invert the mapping between edges and fractures.

        Returns:
            List of list: For each fracture, index of all edges that points to
                it.
        """
        f2e = []
        for fi in range(len(self._fractures)):
            f_l = []
            for ei, e in enumerate(edges_2_frac):
                if fi in e:
                    f_l.append(ei)
            f2e.append(f_l)
        return f2e

    @pp.time_logger(sections=module_sections)
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
        # The workflow is based on first collecting information for all fractures,
        # next get information for all intersections between fractures.

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

        # Next, add points relating to the intersections between fractures.
        # Since the intersections are already defined as numpy arrays, this is
        # relatively straightforward.
        num_isect = self.intersections["start"].shape[1]
        num_p = all_p.shape[1]
        # Intersection points are added by first all starts, then all ends
        isect_pt = np.hstack((self.intersections["start"], self.intersections["end"]))
        # Intersection edges are offset by the number of fracture edges
        intersection_edges = num_p + np.vstack(
            (np.arange(num_isect), num_isect + np.arange(num_isect))
        )
        # Merge fields
        all_p = np.hstack((all_p, isect_pt))
        edges = np.hstack((edges, intersection_edges))

        # Mapping from intersections to their fractures
        isect_2_frac = [
            [
                self.intersections["first"][i].index,
                self.intersections["second"][i].index,
            ]
            for i in range(num_isect)
        ]
        edges_2_frac += isect_2_frac

        # Boolean for intersections being on fracture boundaries
        isect_is_boundary = [
            [
                self.intersections["bound_first"][i],
                self.intersections["bound_second"][i],
            ]
            for i in range(num_isect)
        ]
        is_boundary_edge += isect_is_boundary

        # Ensure that edges are integers
        edges = edges.astype("int")

        logger.info(
            "Points and edges done. Elapsed time %.5f", time.time() - start_time
        )

        # Uniquify the points and edges before returning.
        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

    @pp.time_logger(sections=module_sections)
    def _uniquify_points_and_edges(self, all_p, edges, edges_2_frac, is_boundary_edge):

        start_time = time.time()
        logger.info(
            """Uniquify points and edges, starting with %i points, %i
                    edges""",
            all_p.shape[1],
            edges.shape[1],
        )

        # We now need to find points that occur in multiple places
        p_unique, _, all_2_unique_p = setmembership.unique_columns_tol(
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
        edges_2_frac_new = e_unique.shape[1] * [np.empty(0, dtype=int)]
        is_boundary_edge_new = e_unique.shape[1] * [np.empty(0, dtype=int)]

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
        if self.run_checks:
            self._verify_fractures_in_plane(p_unique, edges, edges_2_frac)

        logger.info(
            """Uniquify complete. %i points, %i edges. Ellapsed time
                    %.5f""",
            p_unique.shape[1],
            edges.shape[1],
            time.time() - start_time,
        )

        return p_unique, edges, edges_2_frac, is_boundary_edge

    @pp.time_logger(sections=module_sections)
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
            edges_2_frac (list of np.arrays): One list item per edge, each item is an
                np.array with indices of all fractures sharing this edge.
            is_boundary_edge (list of np.arrays): One list item per edge. Each item is
                an np.arary with values 0 or 1, according to whether the edge is on the
                boundary of a fracture. Ordering of the inner list corresponds to the
                ordering in edges_2_frac.

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
        for fi in range(len(self._fractures)):

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
            p_new, edges_new, tags = pp.intersections.split_intersecting_segments_2d(
                p_2d, edges_2d, tol=self.tol
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
                # caused by the implementation of split_intersection_segments_2d.
                # We therefore compare the new edge to the old ones (before
                # splitting). If found, use the old information; if not, use
                # index as tracked by splitting.
                is_old, old_loc_ind = setmembership.ismember_rows(
                    edges_new_glob[:, ei].reshape((-1, 1)), edges[:2, edges_loc_ind]
                )
                if is_old[0]:
                    # The edge was preserved, with the same indices for start and end
                    # during splitting. Information on edge-to-fracs, and on whether this
                    # is a boundary edge is identical to that of the edge before the
                    # splitting routine. The global index is that of the local edges
                    # that had the same nodes.
                    glob_ei = [edges_loc_ind[old_loc_ind[0]]]
                else:
                    # This edge is formed by splitting of old edges. To recover all,
                    # exploit information from the splitting of segments.
                    cols_mapped_to_glob_ei = tags[1] == ei
                    glob_ei = tags[0][cols_mapped_to_glob_ei]

                # Update edge_2_frac and boundary information.
                e2f = np.array([], dtype=int)
                ib = np.array([], dtype=bool)
                for gi in glob_ei:
                    e2f = np.hstack((e2f, edges_2_frac[gi]))
                    ib = np.hstack((ib, is_boundary_edge[gi]))

                # There may de duplicates in e2f (if the size of glob_ei is larger
                # than 1), but these are removed in the below call to uniquify points
                # and edges.
                edges_2_frac.append(e2f)
                is_boundary_edge.append(ib)

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
        if self.run_checks:
            self._verify_fractures_in_plane(all_p, edges, edges_2_frac)

        return self._uniquify_points_and_edges(
            all_p, edges, edges_2_frac, is_boundary_edge
        )

    @pp.time_logger(sections=module_sections)
    def fractures_of_points(self, pts):
        """
        For a given point, find all fractures that refer to it, either as
        vertex or as internal.

        Returns:
            list of int: indices of fractures, one list item per point.

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

    @pp.time_logger(sections=module_sections)
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
            d = pp.distances.point_pointset(pt[:, pi], pt[:, pi + 1 :])
            ind = np.argwhere(d < dist).ravel("F")
            for i in ind:
                # Indices of close points, with an offset to compensate for
                # slicing of the point cloud.
                c_points.append((pi, i + pi + 1, d[i]))

        return c_points

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _points_2_plane(self, p_loc, edges_loc, p_ind_loc):
        """
        Convenience method for rotating a point cloud into its own 2d-plane.
        """
        # Center point cloud around the origin
        p_loc_c = np.mean(p_loc, axis=1).reshape((-1, 1))
        p_loc -= p_loc_c

        # Project the points onto the local plane defined by the fracture
        rot = pp.map_geometry.project_plane_matrix(p_loc, tol=self.tol)
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

    def __repr__(self) -> str:
        s = "Fracture set with " + str(len(self._fractures)) + " planes\n"
        if "boundary" in self.tags:
            bnd = "boundary"
            s += f"{sum(self.tags[bnd])} of the fractures are domain boundaries"
        else:
            s += "No boundary information is given"
        return s

    def _reindex_fractures(self):
        for fi, f in enumerate(self._fractures):
            f.index = fi

    def bounding_box(self):
        """Obtain bounding box for fracture network.

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

    @pp.time_logger(sections=module_sections)
    def impose_external_boundary(
        self,
        domain: Optional[Union[Dict[str, float], List[np.ndarray]]] = None,
        keep_box: bool = True,
    ) -> np.ndarray:
        """
        Set an external boundary for the fracture set.

        There are two permissible data formats for the domain boundary:
            1) A 3D box, described by its minimum and maximum coordinates.
            2) A list of polygons (each specified as a np.array, 3xn), that together
                form a closed polyhedron.

        If no bounding box is provided, a box will be fited outside the fracture
        network.

        The fratures will be truncated to lay within the bounding
        box; that is, Fracture.p will be modified. The orginal coordinates of
        the fracture boundary can still be recovered from the attribute
        Fracture.orig_points.

        Fractures that are completely outside the bounding box will be deleted
        from the fracture set.

        Parameters:
            box (dictionary or list of np.ndarray): See above for description.
            keep_box (bool, optional): If True (default), the bounding surfaces will be
                added to the end of the fracture list, and tagged as boundary.

        Returns:
            np.array: Mapping from old to new fractures, referring to the fractures in
                self._fractures before and after imposing the external boundary.
                The mapping does not account for the boundary fractures added to the
                end of the fracture array (if keep_box) is True.

        Raises:
            ValueError
                If the FractureNetwork contains no fractures and no domain was passed
                to this method.

        """
        if domain is None and not self._fractures:
            # Cannot automatically calculate external boundary for non-fractured grids.
            raise ValueError(
                "A domain must be supplied to constrain non-fractured media."
            )
        self.bounding_box_imposed = True

        if domain is not None:
            if isinstance(domain, dict):
                polyhedron = self._make_bounding_planes_from_box(domain)
            else:
                polyhedron = domain
            self.domain = domain
        else:
            # Compute a bounding box from the extension of the fractures.
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
            polyhedron = self._make_bounding_planes_from_box(box)
            self.domain = polyhedron

        # Constrain the fractures to lie within the bounding polyhedron
        polys = [f.p for f in self._fractures]
        constrained_polys, inds = pp.constrain_geometry.polygons_by_polyhedron(
            polys, polyhedron
        )

        # Delete fractures that are not member of any constrained fracture
        old_frac_ind = np.arange(len(self._fractures))
        delete_frac = np.setdiff1d(old_frac_ind, inds)
        if inds.size > 0:
            split_frac = np.where(np.bincount(inds) > 1)[0]
        else:
            split_frac = np.zeros(0, dtype=int)

        ind_map = np.delete(old_frac_ind, delete_frac)

        # Update the fractures with the new data format
        for poly, ind in zip(constrained_polys, inds):
            if ind not in split_frac:
                self._fractures[ind].p = poly
        # Special handling of fractures that are split in two
        for fi in split_frac:
            hit = np.where(inds == fi)[0]
            for sub_i in hit:
                self.add(Fracture(constrained_polys[sub_i]))
                ind_map = np.hstack((ind_map, fi))

        # Delete fractures that have all points outside the bounding box
        # There may be some uncovered cases here, with a fracture barely
        # touching the box from the outside, but we leave that for now.
        for i in np.unique(np.hstack((delete_frac, split_frac)))[::-1]:
            del self._fractures[i]

        # Final sanity check: All fractures should have at least three
        # points at the end of the manipulations
        for f in self._fractures:
            assert f.p.shape[1] >= 3

        boundary_tags = self.tags.get("boundary", [False] * len(self._fractures))
        if keep_box:
            for f in polyhedron:
                self.add(Fracture(f))
                boundary_tags.append(True)
        self.tags["boundary"] = boundary_tags

        self._reindex_fractures()
        return ind_map

    def _make_bounding_planes_from_box(
        self, box: Dict[str, float], keep_box: bool = True
    ) -> List[np.ndarray]:
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
        west = np.array([[x0, x0, x0, x0], [y0, y1, y1, y0], [z0, z0, z1, z1]])
        east = np.array([[x1, x1, x1, x1], [y0, y1, y1, y0], [z0, z0, z1, z1]])
        south = np.array([[x0, x1, x1, x0], [y0, y0, y0, y0], [z0, z0, z1, z1]])
        north = np.array([[x0, x1, x1, x0], [y1, y1, y1, y1], [z0, z0, z1, z1]])
        bottom = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z0, z0, z0, z0]])
        top = np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [z1, z1, z1, z1]])

        bound_planes = [west, east, south, north, bottom, top]
        return bound_planes

    @pp.time_logger(sections=module_sections)
    def _classify_edges(self, polygon_edges, constraints):
        """
        Classify the edges into fracture boundary, intersection, or auxiliary.
        Also identify points on intersections between interesctions (fractures
        of co-dimension 3)

        Parameters:
            polygon_edges (list of lists): For each polygon the global edge
                indices that forms the polygon boundary.
            constraints (list or np.array): Which polygons are constraints, and should
                not cause the generation of 1d grids.

        Returns:
            tag: Tag of the edges, using the values in GmshConstants. Note
                that auxiliary points will not be tagged (these are also
                ignored in gmsh_interface.GmshWriter).
            np.array, boolean: True for edges which are internal to at least one
                polygon.
            np.array, boolean: True for edges that are internal to at least one polygon,
                and on the boundary of another polygon.

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

        # Count the number of referals to the edge from different polygons
        # Only do this if the polygon is not a constraint
        num_referals = np.zeros(num_edges)
        for ei, ep in enumerate(edge_2_poly):
            ep_not_constraint = np.setdiff1d(ep, constraints)
            num_referals[ei] = np.unique(ep_not_constraint).size

        # A 1-d grid will be inserted where there is more than one fracture
        # referring.
        has_1d_grid = np.where(num_referals > 1)[0]

        num_is_bound = len(is_bound)
        tag = np.zeros(num_edges, dtype="int")
        # Find edges that are tagged as a boundary of all its fractures
        all_bound = [np.all(is_bound[i]) for i in range(num_is_bound)]

        # We also need to find edges that are on the boundary of some, but not all
        # the fractuers
        any_bound = [np.any(is_bound[i]) for i in range(num_is_bound)]
        some_bound = np.logical_and(any_bound, np.logical_not(all_bound))

        bound_ind = np.where(all_bound)[0]
        # Remove boundary  that are referred to by more than fracture - this takes
        # care of L-type intersections
        bound_ind = np.setdiff1d(bound_ind, has_1d_grid)

        # Index of lines that should have a 1-d grid. This are all of the first
        # num-constraints, minus those on the boundary.
        # Note that edges with index > num_is_bound are known to be of the
        # auxiliary type. These will have tag zero; and treated in a special
        # manner by the interface to gmsh.
        not_boundary_ind = np.setdiff1d(np.arange(num_is_bound), bound_ind)
        tag[bound_ind] = Tags.FRACTURE_TIP.value

        tag[not_boundary_ind] = Tags.FRACTURE_INTERSECTION_LINE.value

        return tag, np.logical_not(all_bound), some_bound

    @pp.time_logger(sections=module_sections)
    def _on_domain_boundary(self, edges, edge_tags):
        """
        Finds edges and points on boundary, to avoid that these
        are gridded.

        Returns:
            np.array: For all points in the decomposition, the value is 0 if the point is
                in the interior, constants.FRACTURE_TAG if the point is on a fracture
                that extends to the boundary, and constants.DOMAIN_BOUNDARY_TAG if the
                point is part of the boundary specification.
            np.array: For all edges in the decomposition, tags identifying the edge
                as on a fracture or boundary.

        """
        # Obtain current tags on fractures
        boundary_polygons = np.where(
            self.tags.get("boundary", [False] * len(self._fractures))
        )[0]

        # ... on the points...
        point_tags = Tags.NEUTRAL.value * np.ones(
            self.decomposition["points"].shape[1], dtype=int
        )
        # and the mapping between fractures and edges.
        edges_2_frac = self.decomposition["edges_2_frac"]

        # Loop over edges, and the polygons to which the edge belongs
        for e, e2f in enumerate(edges_2_frac):

            # Check if the polygons are on the boundary
            edge_of_domain_boundary = np.in1d(e2f, boundary_polygons)

            if any(edge_of_domain_boundary):
                # If all associated polygons are boundary, this is simple
                if all(edge_of_domain_boundary):
                    # The point is not associated with a fracture extending to the
                    # boundary
                    edge_tags[e] = Tags.DOMAIN_BOUNDARY_LINE.value
                    # The points of this edge are also associated with the boundary
                    point_tags[edges[:, e]] = Tags.DOMAIN_BOUNDARY_POINT.value
                else:
                    # The edge is associated with at least one fracture. Still, if it is
                    # also the edge of at least one boundary point, we will consider it
                    # a domain boundary edge
                    on_one_domain_edge = False
                    for pi in np.where(edge_of_domain_boundary)[0]:
                        if e not in self.decomposition["line_in_frac"][e2f[pi]]:
                            on_one_domain_edge = True
                            break

                    # The line is on the boundary
                    if on_one_domain_edge:
                        edge_tags[e] = Tags.DOMAIN_BOUNDARY_LINE.value
                        point_tags[edges[:, e]] = Tags.DOMAIN_BOUNDARY_POINT.value
                    else:
                        # The edge is an intersection between a fracture and a boundary
                        # polygon
                        edge_tags[e] = Tags.FRACTURE_BOUNDARY_LINE.value
                        point_tags[edges[:, e]] = Tags.FRACTURE_BOUNDARY_POINT.value
            else:
                # This is not an edge on the domain boundary, and the tag assigned in
                # in self._classify_edges() is still valid: It is either a fracture tip
                # or a fracture intersection line
                # The point tag is still neutral; it may be adjusted later depending on
                # how many intersection lines refer to the point
                continue

        return point_tags, edge_tags

    @pp.time_logger(sections=module_sections)
    def _poly_2_segment(self):
        """
        Represent the polygons by the global edges, and determine if the lines
        must be reversed (locally) for the polygon to form a closed loop.

        Returns:
            list of np.array: Each list item gives the indices of the edges need to
                form a polygon (fracture, boundary) in the decomposition.
            list of np.array: For each polygon, and each edge of the polygon, True if
                the edge must be reversed (relative to its description in
                self.decomposition['edges']).

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

    @pp.time_logger(sections=module_sections)
    def _determine_mesh_size(self, point_tags, **kwargs):
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
        # Compute distance between the points forming the network
        # This will also account for points close to segments on other fractures, as
        # an auxiliary point will already have been inserted there
        dist = pp.distances.pointset(p, max_diag=True)
        point_dist = np.min(dist, axis=1)
        logger.info(
            "Minimal distance between points encountered is " + str(np.min(dist))
        )
        # The mesh size should not be smaller than the prescribed minimum distance
        mesh_size_min = np.maximum(point_dist, self.mesh_size_min * np.ones(num_pts))
        # The mesh size should not be larger than the prescribed fracture mesh size.
        # If the boundary mesh size is also presribed, that will be enforced below.
        mesh_size = np.minimum(mesh_size_min, self.mesh_size_frac * np.ones(num_pts))

        if self.mesh_size_bound is not None:
            on_boundary = point_tags == Tags.DOMAIN_BOUNDARY_POINT.value
            mesh_size_bound = np.minimum(
                mesh_size_min, self.mesh_size_bound * np.ones(num_pts)
            )
            mesh_size[on_boundary] = np.maximum(mesh_size, mesh_size_bound)[on_boundary]
        return mesh_size

    @pp.time_logger(sections=module_sections)
    def _insert_auxiliary_points(
        self, mesh_size_frac=None, mesh_size_min=None, mesh_size_bound=None
    ):
        """Insert auxiliary points on fracture edges. Used to guide gmsh mesh
        size parameters.

        The function should only be called once to avoid insertion of multiple
        layers of extra points, this will likely kill gmsh.

        The function is motivated by similar functionality for 2d domains, but
        is considerably less mature.

        The function should be used in conjunction with _determine_mesh_size().
        The ultimate goal is to set the mesh size
        for geometrical points in Gmsh. To that end, this function inserts
        additional points on the fracture boundaries. The mesh size is then
        determined as the distance between all points in the fracture
        description.

        Parameters:
            mesh_size_frac: Ideal mesh size. Will be added to all points that
                are sufficiently far away from other points.
            mesh_size_min: Minimal mesh size; we will make no attempts to
                enforce even smaller mesh sizes upon Gmsh.
            mesh_size_bound (optional): Boundary mesh size. Will be added to the points
                @pp.time_logger(sections=module_sections)
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

        # Auxiliary points may be added for two reasons: For fractures that do intersect
        # the points are added on the segments on the fracture boundaries, close to the
        # endpoints of the intersection segments. This is tailored to the way gmsh
        # creates grids, with meshing of 1d lines before 2d surffaces.
        # For non-intersecting fractures, auxiliary points may be added on the fracture
        # segments if they are sufficiently close.

        # FIXME: If a fracture is close to the interior of another fracture, this
        # will not be seen by the current algorithm. The solution should be to add
        # points to the fracture surface, but make sure that this is not decleared
        # physical. The operation should only be done for fracture pairs where no
        # auxiliary points were added because of close segments.

        def dist_p(a, b):
            # Helper funciton to get the distance from a set of points (a) to a single point
            # (b).
            if a.size == 3:
                a = a.reshape((-1, 1))
            b = b.reshape((-1, 1))
            return np.sqrt(np.sum(np.power(b - a, 2), axis=0))

        # Dictionary that for each fracture maps the index of all other fractures.
        intersecting_fracs: Dict[List[int]] = {}

        # Loop over all fractures
        for fi, f in enumerate(self._fractures):

            # Do not insert auxiliary points on domain boundaries
            if "boundary" in self.tags.keys() and self.tags["boundary"][fi]:
                continue

            # First compare segments with intersections to this fracture
            nfp = f.p.shape[1]

            # Keep track of which other fractures are intersecting - will be
            # needed later on
            intersections_this_fracture: List[int] = []

            # Loop over all intersections of the fracture
            isects, is_first_isect = self.intersections_of_fracture(f)
            for (i, is_first) in zip(isects, is_first_isect):

                # Register this intersection
                if is_first:
                    intersections_this_fracture.append(
                        self.intersections["second"][i].index
                    )
                else:
                    intersections_this_fracture.append(
                        self.intersections["first"][i].index
                    )

                # Find the point on the fracture segments that is closest to the
                # intersection point
                isect_coord = np.vstack(
                    (self.intersections["start"][:, i], self.intersections["end"][:, i])
                ).T

                dist, cp = pp.distances.points_segments(
                    isect_coord, f.p, np.roll(f.p, -1, axis=1)
                )
                # Insert a (candidate) point only at the segment closest to the
                # intersection point. If the intersection line runs parallel
                # with a segment, this may be insufficient, but we will deal
                # with this if it turns out to be a problem.
                closest_segment = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(2), closest_segment]

                inserted_points = 0

                for pi, (si, di) in enumerate(zip(closest_segment, min_dist)):
                    if di < mesh_size_frac and di > mesh_size_min:
                        # Distance between the closest point of the intersection segment
                        # and the points of this fracture.
                        d_1 = dist_p(cp[pi, si], f.p[:, si])
                        d_2 = dist_p(cp[pi, si], f.p[:, (si + 1) % nfp])
                        # If the intersection point is not very close to any of
                        # the points on the segment, we split the segment.
                        # NB: It is critical that this is done before a call to
                        # self.split_intersections(), or else the export to gmsh
                        # will go wrong.
                        if d_1 > mesh_size_min and d_2 > mesh_size_min:
                            f.p = np.insert(
                                f.p,
                                (si + 1 + inserted_points) % nfp,
                                cp[pi, si],
                                axis=1,
                            )

                            inserted_points += 1
                            nfp += 1
                            # Check if some of the intersections of the fracture should
                            # be split when the new point is added.
                            self._split_intersections_of_fracture(cp[pi, si])

            # Take note of the intersecting fractures
            intersecting_fracs[fi] = intersections_this_fracture

        # Next, insert points along segments that are close to other fractures,
        # which are not touching. The insertion is not symmetric, so that when
        # a point is inserted on one fracture, it is not inserted on the close
        # fracture which triggered the insertion (the corresponding point on the
        # second fracture may beinserted on later).
        for fi, f in enumerate(self._fractures):

            # Do not insert auxiliary points on domain boundaries
            if "boundary" in self.tags.keys() and self.tags["boundary"][fi]:
                continue

            # Get the segments of all other fractures which do no intersect with
            # the main one.
            # First the indices
            other_fractures = []
            for of in self._fractures:
                if not (of.index in intersecting_fracs[fi] or of.index == f.index):
                    other_fractures.append(of)

            # If no other fractures were found, we go on.
            if len(other_fractures) == 0:
                continue

            # Next, arrays of start and end points. This allows us to use the
            # vectorized segment-segment distance computation.
            start_all = np.zeros((3, 0))
            end_all = np.zeros((3, 0))
            for of in other_fractures:
                start_all = np.hstack((start_all, of.p))
                end_all = np.hstack((end_all, np.roll(of.p, -1, axis=1)))

            # Now, loop over all segments in the main fracture, look for close
            # segments on other fractures, and insert points along the main
            # segment if some are close enough. Points may be inserted if the
            # other fracture is closer than mesh_size_frac, however, the
            # auxiliary points should not be too dense along the main segment.
            # Specifically, points are only inserted if the distance form existing
            # points on the segment (endpoint or auxiliary) is larger than the
            # minimum distance. This implies that if several other fractures
            # have nearby closest points along the segment, only some of the
            # possible auxiliary points wil be inserted, however, the mesh
            # size in that part of the segment should still be fairly small.

            # Since the number of points in the fracture description changes by
            # insertion of auxiliary points, we use a counter and while loop
            # to loop over the segments.
            start_index = 0

            while start_index < f.p.shape[1]:
                # Start and end of this segment. These should be vertexes in the
                # original fracture (prior to insertion of auxiliary points)
                seg_start = f.p[:, start_index].reshape((-1, 1))
                # Modulus to finish the final segment of the fracture.
                seg_end = f.p[:, (start_index + 1) % f.p.shape[1]].reshape((-1, 1))

                # Find the distance from this segment to all other segments
                # (not intersecting).
                dist, cp_f, _ = pp.distances.segment_segment_set(
                    seg_start, seg_end, start_all, end_all
                )

                # Sort the points according to distances, so that we try to insert
                # the most needed auxiliary points first
                sort_ind = np.argsort(dist)
                dist = dist[sort_ind]
                cp_f = cp_f[:, sort_ind]

                # Chances are, many of the other segments will be closest to an end
                # of this segment. These can be disregarded (below).
                dist_segment_ends = np.minimum(
                    dist_p(cp_f, seg_start), dist_p(cp_f, seg_end)
                )
                not_closest_on_ends = dist_segment_ends > mesh_size_min

                # Also cut all points corresponding to segments that are further
                # away than the prescribed fracture mesh size
                not_far_away = dist < mesh_size_frac

                is_candidate = np.logical_and(not_closest_on_ends, not_far_away)

                if not np.any(is_candidate):
                    # If no candidate point was found for this edge, we can move on:
                    # Increase the start index to hit the next segment along the main
                    # fracture.
                    start_index += 1
                else:
                    # First pick out all condidate auxiliary points, then reduce
                    # to a unique subset (close candidate points may occur e.g.
                    # for fractures close to complex domain boundaries.)
                    candidates = cp_f[:, is_candidate]
                    (
                        unique_candidates,
                        _,
                        o2n,
                    ) = pp.utils.setmembership.unique_columns_tol(candidates, self.tol)

                    # Make arrays for points along the segment (originally the
                    # endpoints), and for the auxiliary points
                    segment_points = np.hstack((seg_start, seg_end))
                    new_points = np.zeros((3, 0))

                    # Loop over all candidate points (due to the sorting on
                    # distances, the candidates corresponds to increasing distance
                    # from the nearby fracture).
                    for point_counter in range(unique_candidates.shape[1]):
                        pt = unique_candidates[:, point_counter].reshape((-1, 1))

                        # Insert the candidate point if it is sufficiently far
                        # away from existing points along the line.
                        if np.min(dist_p(segment_points, pt)) > mesh_size_min:
                            # No sorting of points along the line, take care
                            # of this below.
                            segment_points = np.hstack((segment_points, pt))
                            # Register new point.
                            new_points = np.hstack((new_points, pt))

                    # Compute distance between new points and segment start
                    dist_from_start = dist_p(new_points, seg_start)

                    # Sort the new points according to distance. This will make
                    # the new sub-segments a line along the main segment. Then insert.
                    sorted_new = new_points[:, np.argsort(dist_from_start)]

                    f.p = np.insert(f.p, [start_index + 1], sorted_new, axis=1)

                    # If the new points sit on top of intersections, these must be split
                    for pi in range(sorted_new.shape[1]):
                        self._split_intersections_of_fracture(
                            sorted_new[:, pi].reshape((-1, 1))
                        )

                    # Move the start index to the next segment, compensating for
                    # inserted auxiliary points.
                    start_index += 1 + sorted_new.shape[1]

    @pp.time_logger(sections=module_sections)
    def _split_intersections_of_fracture(self, cp: np.ndarray) -> None:
        """Check if a point lies on intersections of a given fracture, and if so,
        split the intersection into two.

        The intended usage is when auxiliary points are added to a fracture's vertexes
        after the intersections have been identified.

        Parameters:
            cp (np.array, size 3): Coordinate of the added point. The intersection will
                be split if the point lies on the intersection.

        """
        # Intersections must be split for all fractures, not only those where the
        # fracture is involved, or else Y-type intersections may not be treated
        # correctly.

        if cp.size < 4:
            cp = cp.reshape((-1, 1))

        # Compute distacne from the point to all other points
        isect_start = self.intersections["start"]
        isect_end = self.intersections["end"]
        d_isect, _ = pp.distances.points_segments(cp, isect_start, isect_end)

        # We are only interested in segments that are  close to the point
        close_to_segment = d_isect[0] < self.tol
        # If there are no close segments, there is nothing more to do.
        if not np.any(close_to_segment):
            return

        # If the close point is the endpoint of the intersection segment, we
        # should not split it
        dist_start = pp.distances.point_pointset(cp, isect_start)
        dist_end = pp.distances.point_pointset(cp, isect_end)
        close_to_endpoint = np.logical_or(dist_start < self.tol, dist_end < self.tol)

        # replace intersections that are cut in two pieces, both with non-zero length.
        to_replace = np.where(
            np.logical_and(close_to_segment, np.logical_not(close_to_endpoint))
        )[0]

        if to_replace.size == 0:
            return

        # Duplicate intersections, and split into two.
        first = self.intersections["first"][to_replace]
        second = self.intersections["second"][to_replace]

        bound_first = self.intersections["bound_first"][to_replace]
        bound_second = self.intersections["bound_second"][to_replace]

        start = self.intersections["start"][:, to_replace]
        end = self.intersections["end"][:, to_replace]
        if start.size < 4:
            start = start.reshape((-1, 1))
            end = end.reshape((-1, 1))

        new_first = np.tile(first, 2)
        new_second = np.tile(second, 2)
        new_bound_first = np.tile(bound_first, 2)
        new_bound_second = np.tile(bound_second, 2)

        new_start = np.hstack((start, end))
        new_end = np.tile(cp, 2 * to_replace.size)

        self._add_intersection(
            new_first, new_second, new_start, new_end, new_bound_first, new_bound_second
        )
        # Delete the old intersections.
        for key in self.intersections:
            self.intersections[key] = np.delete(
                self.intersections[key], to_replace, axis=-1
            )

    def _add_intersection(
        self,
        first: Union[Fracture, np.ndarray],
        second: Union[Fracture, np.ndarray],
        start: np.ndarray,
        end: np.ndarray,
        bound_first: Union[bool, np.ndarray],
        bound_second: Union[bool, np.ndarray],
    ) -> None:
        # Add one or several fracture pairs to the intersections.
        # If several fractures, these should be wrapped in a numpy array.
        self.intersections["first"] = np.hstack(
            (self.intersections["first"], first)  # type: ignore
        )
        self.intersections["second"] = np.hstack(
            (self.intersections["second"], second)  # type: ignore
        )

        if start.size < 4:
            start = start.reshape((-1, 1))
            end = end.reshape((-1, 1))

        self.intersections["start"] = np.hstack((self.intersections["start"], start))
        self.intersections["end"] = np.hstack((self.intersections["end"], end))
        self.intersections["bound_first"] = np.hstack(
            (self.intersections["bound_first"], bound_first)
        )
        self.intersections["bound_second"] = np.hstack(
            (self.intersections["bound_second"], bound_second)
        )

    @pp.time_logger(sections=module_sections)
    def to_file(
        self, file_name: str, data: Dict[str, np.ndarray] = None, **kwargs
    ) -> None:
        """
        Export the fracture network to file.

        The file format is given as an kwarg, by default vtu will be used. The writing is
        outsourced to meshio, thus the file format should be supported by that package.

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

        Optional arguments in kwargs:
            binary (boolean): Use binary export format. Default to
                True.
            fracture_offset (int): Use to define the offset for a
                fracture id. Default to 1.
            folder_name (string): Path to save the file. Default to "./".
            extension (string): File extension. Default to ".vtu".

        See also the functions self.to_fab() and self.to_csv().

        """
        if data is None:
            data = {}

        binary: bool = kwargs.pop("binary", True)
        fracture_offset: int = kwargs.pop("fracture_offset", 1)
        extension: str = kwargs.pop("extension", ".vtu")
        folder_name: str = kwargs.pop("folder_name", "")

        if kwargs:
            msg = "Got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        if not file_name.endswith(extension):
            file_name += extension

        # fracture points
        meshio_pts = np.empty((0, 3))
        # map cell->nodes
        cell_to_nodes = {}
        # cell id map
        cell_id = {}
        # counter for the points
        pts_pos = 0

        # we operate fracture by fracture
        for fid, frac in enumerate(self._fractures):
            # save the points of the fracture
            meshio_pts = np.vstack((meshio_pts, frac.p.T))
            num_pts = frac.p.shape[1]
            # determine the fracture type
            cell_type = "polygon" + str(num_pts)
            # just stack all the nodes after the others
            nodes = pts_pos + np.arange(num_pts)
            pts_pos += num_pts
            # polygon should be stored uniformly
            if cell_type not in cell_to_nodes:
                cell_to_nodes[cell_type] = np.atleast_2d(nodes)
                cell_id[cell_type] = [fid]
            else:
                cell_to_nodes[cell_type] = np.vstack((cell_to_nodes[cell_type], nodes))
                cell_id[cell_type] += [fid]

        # construct the meshio data structure
        num_block = len(cell_to_nodes)
        meshio_cells = np.empty(num_block, dtype=object)
        meshio_cell_id = np.empty(num_block, dtype=object)
        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block)
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # add also the fracture number to the data to export
        data.update(
            {"fracture_number": fracture_offset + np.arange(len(self._fractures))}
        )

        meshio_data = {}
        # store also the data
        for key, val in data.items():
            # for each field create a sub-vector for each geometrically uniform group of cells
            meshio_data[key] = np.empty(num_block, dtype=object)
            # fill up the data
            for block, ids in enumerate(meshio_cell_id):
                if val.ndim == 1:
                    meshio_data[key][block] = val[ids]
                elif val.ndim == 2:
                    meshio_data[key][block] = val[:, ids].T
                else:
                    raise ValueError

        meshio_grid_to_export = meshio.Mesh(
            meshio_pts, meshio_cells, cell_data=meshio_data
        )
        meshio.write(folder_name + file_name, meshio_grid_to_export, binary=binary)

    @pp.time_logger(sections=module_sections)
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
            for f in self._fractures:
                csv_writer.writerow(f.p.ravel(order="F"))

    @pp.time_logger(sections=module_sections)
    def to_fab(self, file_name):
        """
        Save the 3d network on a fab file, as specified by FracMan.

        The filter is based on the .fab-files needed at the time of writing, and
        may not cover all options available.

        Parameters:
            file_name (str): File name.
        """
        # function to write a numpy matrix as string
        @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def fracture_to_plane(self, frac_num):
        """Project fracture vertexes and intersection points to the natural
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

        rot = pp.map_geometry.project_plane_matrix(frac.p)

        @pp.time_logger(sections=module_sections)
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

        other_frac = np.empty(0, dtype=int)

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
