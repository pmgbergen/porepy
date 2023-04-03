"""
Contains classes representing 1D fractures, i.e., manifolds of dimension 1 embedded
in 2d.

"""

import numpy as np

from .fracture import Fracture


class LineFracture(Fracture):
    """A class representing linear fracture in 2D."""

    def sort_points(self) -> np.ndarray:
        """Sort the vertices.

        For :class:~`LineFracture`s defined by a two-vertex line segment, sorting is
        trivial.

        Returns:
            :obj:`~numpy.ndarray`: ``(dtype=int)``
                The sorted vertex indices.

        """
        return np.arange(2)

    def local_coordinates(self) -> np.ndarray:
        """Compute the 1d vertex coordinates of the fracture in a local system.

        The first coordinate is set to ``x=0``, the second to ``x=self.length()``.

        Returns:
            :obj:`~numpy.ndarray`: ``(shape=(1, 2))``
                The coordinates of the two vertices in the 1d local dimension.

        """
        return np.reshape([0, self.length()], (1, 2))

    def compute_centroid(self) -> np.ndarray:
        """Compute the centroid of the fracture.

        Returns:
            :obj:`~numpy.ndarray`: Array containing the 2D coordinates of the centroid.

        """
        return np.mean(self.pts, axis=1)

    def compute_normal(self) -> np.ndarray:
        """Compute the normal vector of the fracture

        Returns:
            :obj:`~numpy.ndarray`: ``(shape=(2, 1))``
                Normal vector of the fracture.

        """

        diff = np.diff(self.pts, axis=1)
        normal = np.array([diff[0], -diff[1]])
        return normal / np.linalg.norm(normal)

    def _check_pts(self):
        """Consistency check for ``self.pts``.

        Defining a LineFracture with more than two points is not supported, as this
        violates assumptions underlying som class methods.

        Raises:
            ValueError: If ``self.pts`` does not have the expected shape.
            ValueError: If the two ``self.pts`` are identical.

        """
        if self.pts.shape != (2, 2):
            raise ValueError(
                "pts defining a LineFracture should have dimensions 2 x 2."
            )
        if np.all(np.isclose(self.pts[:, 0], self.pts[:, 1])):
            raise ValueError("Need two distinct pts to define a LineFracture.")

    def length(self) -> float:
        """Compute length of the fracture.

        Returns:
            length: Fracture length.

        """
        diff = np.diff(self.pts, axis=1)
        return float(np.linalg.norm(diff))
