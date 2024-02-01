"""Contains classes representing two-dimensional fractures.

That is, manifolds of dimension 2 embedded in 3D.

"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.utils import setmembership

from .fracture import Fracture


class PlaneFracture(Fracture):
    """A class representing planar fractures in 3D in form of a (bounded) plane,
    i.e. manifolds of dimension 2 embedded in 3D.

    Non-convex fractures can be initialized, but geometry processing (computation of
    intersections etc.) is not supported for non-convex geometries.

    For a description of constructor arguments, see base class.

    Raises:
        AssertionError: If the resulting fracture is not planar or convex
            (if additional check requested).

    """

    def __init__(
        self,
        points: ArrayLike,
        index: Optional[int] = None,
        check_convexity: bool = False,
        sort_points: bool = True,
    ):
        super().__init__(points=points, index=index, sort_points=sort_points)

        assert self.is_planar(), "Points define non-planar fracture"
        if check_convexity:
            assert self.is_convex(), "Points form non-convex polygon"

    def sort_points(self) -> np.ndarray:
        """Sort the points in a counter-clockwise (CCW) order.

        Note:
            For now, the ordering of nodes in based on a simple angle argument. This
            will not be robust for general point clouds, but we expect the fractures
            to be regularly shaped in this sense. In particular, we will be safe if
            the cell is convex.

        Returns:
            An integer array with containing the sorted indices.

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
        """Represent the vertex coordinates in the natural 2D plane.

        The plane has constant, but not necessarily zero, third coordinate. I.e.,
        no translation to the origin is made.

        Returns:
            An array with ``shape=(2, n)`` representing the 2D coordinates of the
            vertices.

        """
        rotation = pp.map_geometry.project_plane_matrix(self.pts)
        points_2d = rotation.dot(self.pts)

        return points_2d[:2]

    def add_points(
        self,
        p: np.ndarray,
        check_convexity: bool = True,
        tol: float = 1e-4,
        enforce_pt_tol: Optional[float] = None,
    ) -> bool:
        """Add points to the polygon with the implemented sorting enforced.

        Always run a test to check that the points are still planar. By default,
        a check of convexity is also performed, however, this can be turned off to
        speed up simulations (the test uses sympy, which turns out to be slow in many
        cases).

        Parameters:
            p: ``shape=(nd, num_pts)``

                Points to be added.
            check_convexity: ``default=True``

                Verify that the polygon is convex.
            tol: ``default=1e-4``

                Tolerance used to check if the point already exists.
            enforce_pt_tol: ``default=None``

                If not ``None``, enforces a maximal distance between points by removing
                points failing that criterion.

        Returns:
            ``True``, if the resulting polygon is planar, ``False`` otherwise.

            If the argument ``check_convexity`` is ``True``,
            checks for planarity **and** convexity.

        """
        to_enforce = np.hstack(
            (
                np.zeros(self.pts.shape[1], dtype=bool),
                np.ones(p.shape[1], dtype=bool),
            )
        )
        self.pts = np.hstack((self.pts, p))
        self.pts, _, _ = setmembership.unique_columns_tol(self.pts, tol=tol)

        # Sort points to counter-clockwise
        mask = self.sort_points()

        if enforce_pt_tol is not None:
            to_enforce = np.where(to_enforce[mask])[0]
            dist = pp.distances.pointset(self.pts)
            dist /= np.amax(dist)
            np.fill_diagonal(dist, np.inf)
            mask = np.where(dist < enforce_pt_tol)[0]
            mask = np.setdiff1d(mask, to_enforce, assume_unique=True)
            self.remove_points(mask)

        if check_convexity:
            return self.is_convex() and self.is_planar(tol)
        else:
            return self.is_planar()

    def remove_points(
        self, ind: Union[np.ndarray, list[int]], keep_orig: bool = False
    ) -> None:
        """Remove points from the fracture definition.

        Parameters:
            ind: Array or list of indices of points to be removed.
            keep_orig: ``default=False``

                If ``True``, keeps the original points in the attribute
                :attr:`orig_pts`.

        """
        self.pts = np.delete(self.pts, ind, axis=1)

        if not keep_orig:
            self.orig_pts = self.pts

    def is_convex(self) -> bool:
        """See parent class docs.

        Todo:
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

        Parameters:
            tol: ``default=1e-4``

                Tolerance for non-planarity. Treated as an absolute quantity (no
                scaling with fracture extent).

        Returns:
            ``True`` if the polygon is planar, ``False`` otherwise.

        """
        p = self.pts - np.mean(self.pts, axis=1).reshape((-1, 1))
        rot = pp.map_geometry.project_plane_matrix(p)
        p_2d = rot.dot(p)
        return np.max(np.abs(p_2d[2])) < tol

    def compute_centroid(self) -> np.ndarray:
        """See parent class docs.

        Note:
            This method assumes the polygon is convex.

        """
        # Rotate to 2d coordinates
        rot = pp.map_geometry.project_plane_matrix(self.pts)
        p = rot.dot(self.pts)
        z = p[2, 0]
        p = p[:2]

        # Vectors from the first point to all other points. Subsequent pairs of these
        # will span triangles which, assuming convexity, will cover the polygon.
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
        return pp.map_geometry.compute_normal(self.pts)[:, None]

    # EK: Removed typing of return type, since this would necessitate an import of the
    # full sympy module during import of PorePy, which is not desirable.
    def as_sympy_polygon(self, pts: Optional[np.ndarray] = None):
        """Represent polygon as a sympy object.

        Parameters:
            pts: ``shape=(nd, num_pts)``

                Points for the polygon. If ``None``, :attr:`pts` is used.

        Returns:
            Sympy Representation of the polygon formed by ``pts``.

        """
        from sympy.geometry import Point, Polygon

        if pts is None:
            pts = self.pts

        sp = [Point(pts[:, i]) for i in range(pts.shape[1])]
        return Polygon(*sp)

    def _check_pts(self) -> None:
        """Check the shape of ``self.pts``.

        Raises:
            ValueError: If ``self.pts`` does not have the expected shape.

        """
        if self.pts.shape[0] != 3:
            raise ValueError(
                "First dimension of pts defining a PlaneFracture should be 3."
            )
        if self.pts.shape[1] < 3:
            raise ValueError("At least 3 points are needed to define a PlaneFracture.")


def create_elliptic_fracture(
    center: np.ndarray,
    major_axis: float,
    minor_axis: float,
    major_axis_angle: float,
    strike_angle: float,
    dip_angle: float,
    num_points: int = 16,
    index: Optional[int] = None,
) -> PlaneFracture:
    """Initialize an elliptically shaped fracture, approximated by a polygon.

    The rotation of the plane is calculated using three angles. First, the rotation
    of the major axis from the x-axis. Next, the fracture is inclined by specifying
    the strike angle (which gives the rotation axis) measured from the x-axis,
    and the dip angle. All angles are measured in radians.

    Example:
        Fracture centered at ``[0, 1, 0]``, with a ratio of lengths of 2,
        rotation in xy-plane of 45 degrees, and an incline of 30 degrees
        rotated around the x-axis:

        >>> import numpy as np
        >>> import porepy as pp
        >>> frac = pp.create_elliptic_fracture(np.array([0, 1, 0]), 10, 5,
        >>> ... np.pi/4, 0, np.pi/6)

    Parameters:
        center: ``shape=(3, 1)``

            Center coordinates of fracture.
        major_axis: Length of major axis (radius-like, not diameter).
        minor_axis: Length of minor axis.

            There are no checks on whether the minor axis is less or equal the major.
        major_axis_angle: Rotation of the major axis from the x-axis in radians.
            Measured before strike-dip rotation, see above.
        strike_angle: Line of rotation for the dip. Given as angle in radians from the
            x-direction.
        dip_angle: Dip angle in radians, i.e., rotation around the strike direction.
        num_points: ``default=16``

            Number of points used to approximate the ellipsis.
        index: ``default=None``

            Index to be assigned to the fracture.

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

    # Rotate reference points so that the major axis has the right orientation
    major_axis_rot = pp.map_geometry.rotation_matrix(
        major_axis_angle, np.array([0, 0, 1])
    )
    rot_ref_pts = major_axis_rot.dot(ref_pts)

    assert pp.geometry_property_checks.points_are_planar(rot_ref_pts)

    # Then the dip.
    # Rotation matrix of the strike angle.
    strike_rot = pp.map_geometry.rotation_matrix(strike_angle, np.array([0, 0, 1]))
    # Compute strike direction
    strike_dir = strike_rot.dot(np.array([1, 0, 0]))
    dip_rot = pp.map_geometry.rotation_matrix(dip_angle, strike_dir)

    dip_pts = dip_rot.dot(rot_ref_pts)

    assert pp.geometry_property_checks.points_are_planar(dip_pts)

    # Set the points, and store them in a backup.
    pts = center + dip_pts
    return PlaneFracture(pts, index, sort_points=False)
