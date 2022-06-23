"""Contains classes representing fractures in 2D, i.e. manifolds of dimension 2 embedded in 3D.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sympy.geometry import Point, Polygon

import porepy as pp

from .fracture import Fracture


class Fracture3d(Fracture):
    """A class representing planar fracture in 3D in form of a (bounded) plane.

    Non-convex fractures can be initialized, but geometry processing (computation
    of intersections etc.) is not supported for non-convex geometries.
    """

    def sort_points(self) -> np.ndarray:
        """Ensure that the points are sorted in a counter-clockwise (CCW) order.

        Notes
        -----
            For now, the ordering of nodes in based on a simple angle argument.
            This will not be robust for general point clouds, but we expect the
            fractures to be regularly shaped in this sense. In particular, we
            will be safe if the cell is convex.

        Returns
        -------
        numpy.ndarray(dtype=int)
            The indices corresponding to the sorting.
        """
        # First rotate coordinates to the plane
        points_2d = self.local_coordinates()
        # Center around the 2d origin
        points_2d -= np.mean(points_2d, axis=1).reshape((-1, 1))

        theta = np.arctan2(points_2d[1], points_2d[0])
        sort_ind = np.argsort(theta)

        self.pts = self.pts[:, sort_ind]

        return sort_ind

    def local_coordinates(self) -> np.ndarray:
        """
        Represent the vertex coordinates in the natural 2D plane.

        The plane has constant, but not necessarily zero, third coordinate.
        I.e., no translation to the origin is made.

        Returns
        -------
        numpy.ndarray(2 x n)
            The 2d coordinates of the vertices.
        """
        rotation = pp.map_geometry.project_plane_matrix(self.pts)
        points_2d = rotation.dot(self.pts)

        return points_2d[:2]

    def is_convex(self) -> bool:
        """See parent class docs

        TODO
        ----
            If a hanging node is inserted at a segment, this may slightly
            violate convexity due to rounding errors. It should be possible to
            write an algorithm that accounts for this. First idea: Project
            point onto line between points before and after, if the projection
            is less than a tolerance, it is okay.
        """
        if self.pts.shape[1] == 3:
            # A triangle is always convex
            return True

        p_2d = self.local_coordinates()
        return self.as_sympy_polygon(p_2d).is_convex()

    def is_planar(self, tol: float = 1e-4) -> bool:
        """Check if the points representing this fracture lie within a 2D-plane.

        Parameters
        ----------
            tol : float
                Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns
        -------
        bool
            True if the polygon is planar, False otherwise.
        """
        p = self.pts - np.mean(self.pts, axis=1).reshape((-1, 1))
        rot = pp.map_geometry.project_plane_matrix(p)
        p_2d = rot.dot(p)
        return np.max(np.abs(p_2d[2])) < tol

    def compute_centroid(self) -> np.ndarray:
        """See parent class docs.
        The method assumes the polygon is convex.
        """
        # Rotate to 2d coordinates
        rot = pp.map_geometry.project_plane_matrix(self.pts)
        p = rot.dot(self.pts)
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
        return rot.transpose().dot(np.append(center, z)).reshape((3, 1))

    def compute_normal(self) -> np.ndarray:
        """See parent class docs."""
        return pp.map_geometry.compute_normal(self.pts)[:, None]

    def as_sympy_polygon(self, pts: Optional[np.ndarray] = None) -> Polygon:
        """Represent polygon as a ``sympy`` object.

        Parameters
        ----------
            pts : numpy.array(nd x npt) , optional
                Points for the polygon. Defaults to None, in which case self.pts is used.

        Returns
        -------
        sympy.geometry.Polygon
            Representation of the polygon formed by p.
        """

        if pts is None:
            pts = self.pts

        sp = [Point(pts[:, i]) for i in range(pts.shape[1])]
        return Polygon(*sp)


def create_elliptic_fracture(
    center: np.ndarray,
    major_axis: float,
    minor_axis: float,
    major_axis_angle: float,
    strike_angle: float,
    dip_angle: float,
    num_points: int = 16,
    index: Optional[int] = None,
):
    """
    Initialize an elliptic shaped fracture, approximated by a polygon.

    The rotation of the plane is calculated using three angles. First, the
    rotation of the major axis from the x-axis. Next, the fracture is
    inclined by specifying the strike angle (which gives the rotation
    axis) measured from the x-axis, and the dip angle. All angles are
    measured in radians.

    Notes
    -----
        This child class constructor does not call the parent class constructor.
        Ergo parent methods like `copy` are not available

    Parameters
    ----------
        center : numpy.ndarray(3x1)
            Center coordinates of fracture.
        major_axis : float
            Length of major axis (radius-like, not diameter).
        minor_axis : float
            Length of minor axis.
            There are no checks on whether the minor axis is less or equal the major.
        major_axis_angle : float , radians
            Rotation of the major axis from the x-axis.
            Measured before strike-dip rotation, see above.
        strike_angle : float , radians
            Line of rotation for the dip.
            Given as angle from the x-direction.
        dip_angle : float , radians
            Dip angle, i.e. rotation around the strike direction.
        num_points : int , optional
            Defaults to 16.
            Number of points used to approximate the ellipsis.
        index  : int , optional
            Index of fracture. Defaults to None.

    Examples
    --------
        Fracture centered at [0, 1, 0], with a ratio of lengths of 2,
        rotation in xy-plane of 45 degrees, and an incline of 30 degrees
        rotated around the x-axis.
        >>> frac = create_elliptic_fracture(np.array([0, 1, 0]), 10, 5,
                                            np.pi/4, 0, np.pi/6)

    """
    center = np.asarray(center)
    if center.ndim == 1:
        center = center.reshape((-1, 1))

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
    pts = center + dip_pts
    return Fracture3d(pts, index, sort_points=False)
