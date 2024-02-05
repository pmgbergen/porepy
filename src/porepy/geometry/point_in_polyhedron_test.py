"""This module implements a robust point in polyhedron test supporting non-convex
polyhedra.

"""

from __future__ import annotations

from typing import Any

import numpy as np


class PointInPolyhedronTest:
    """This class implements a robust point in polyhedron test supporting non-convex
    polyhedra.

    The implementation is based on the reference: Robust inside-outside segmentation
    using generalized winding numbers (https://doi.org/10.1145/2461912.2461916)

    Implementation requires consistent orientation of the triangulated surface.

    Parameters:
        vertices: ``shape=(num_pt, 3)``

            Triangulation vertices.
        connectivity: ``shape=(num_triangles, 3)``

            Triangulation connectivity map.
        tol: ``default=1e-10``

            Geometric tolerance, used in comparison of points, areas and
            volumes.

    """

    def __init__(
        self,
        vertices: np.ndarray[Any, np.dtype[np.float64]],
        connectivity: np.ndarray[Any, np.dtype[np.int64]],
        tol: float = 1e-10,
    ):
        self.vertices: np.ndarray[Any, np.dtype[np.float64]] = vertices
        self.connectivity: np.ndarray[Any, np.dtype[np.int64]] = connectivity
        self.tol: float = tol

    def solid_angle(self, R: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        """Implements an analytical expression for the solid angle.

        The solid angle is described by a plane triangle surface S at some arbitrary
        point P. It is the measure of the intersection of the three-dimensional unit
        sphere centered at P and the conic hull defined by the vectors in R.

        The implementation is based on equation (6) in reference: Robust inside-outside
        segmentation using generalized winding numbers
        (https://doi.org/10.1145/2461912.2461916)

        Raises:
            ValueError: If the origin ``[0,0,0]`` point coincides with a vertex, is
                collinear with the vertices, or is coplanar with the vertices.

        Parameters:
            R: ``shape=(num_pt, 3)``

                Translated triangle's points at origin ``(0,0,0)``.
                The original triangle's points need to be translated by subtracting the
                arbitrary point P.

        Returns:
            The solid angle measured in steradians.

        """

        r = np.linalg.norm(R, axis=1)

        check_overlapping_vertices = np.any(np.absolute(r) < self.tol)
        if check_overlapping_vertices:
            raise ValueError("Origin point coincides with a vertex")

        # Check collinearity by computing areas formed by pairs of points in R and
        # the origin [0,0,0]. Example: half norm of np.cross(R[0]-[0,0,0], R[1]-[0,0,0])
        # represents the area associated with R[0], R[1], and [0,0,0], if [0,0,0] is
        # collinear with R[0] and R[1] the formed area is zero.
        r_areas = 0.5 * np.array(
            [
                np.linalg.norm(np.cross(R[0], R[1])),
                np.linalg.norm(np.cross(R[1], R[2])),
                np.linalg.norm(np.cross(R[2], R[0])),
            ]
        )
        check_collinearity = np.any(np.absolute(r_areas) < self.tol)
        if check_collinearity:
            raise ValueError("Origin point is collinear with the vertices")

        # Check coplanarity by computing the volume formed by the points in R and the
        # origin [0,0,0]. If [0,0,0] is coplanar with R[0], R[1], and R[2], the formed
        # volume is zero.
        r_volume = np.linalg.norm(np.dot(R[1], np.cross(R[0] - R[1], R[2] - R[1])))
        check_coplanarity = np.any(np.absolute(r_volume) < self.tol)
        if check_coplanarity:
            raise ValueError("Origin point is coplanar with the vertices")

        nv = np.dot(R[0], np.cross(R[1], R[2]))
        dv = (
            np.prod(r)
            + np.dot(R[0], R[1]) * r[2]
            + np.dot(R[0], R[2]) * r[1]
            + np.dot(R[1], R[2]) * r[0]
        )
        angle = 2.0 * np.arctan2(nv, dv)
        return angle

    def winding_number(self, point: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        """Computes the winding number of a closed triangulated surface at given point.

        Parameters:
            point: ``shape=(1, 3)``

                The point being tested.

        Returns:
            The winding number generalized to R^3. Its absolute value ``|wn|`` is 0 for
            points outside the polyhedron, 1 for points inside non-convex polyhedron and
            ``|wn| > 1`` for points inside overlapping polyhedron.

        """
        R = self.vertices - point
        R_triangles = [R[indexes] for indexes in self.connectivity]
        angles = np.array(list(map(self.solid_angle, R_triangles)))
        wn = np.sum(angles) / (4.0 * np.pi)
        return wn
