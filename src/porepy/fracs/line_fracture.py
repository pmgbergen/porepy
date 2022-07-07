"""Contains classes representing fractures of 1D, i.e. manifolds of dimension 1 embedded in 2D.
"""

import numpy as np

from .fracture import Fracture


class LineFracture(Fracture):
    """A class representing linear fracture in 2D."""

    def sort_points(self) -> np.ndarray:
        """Sort the vertices.

        For fractures defined by a two-vertex line segment, sorting is trivial.
        Returns
        -------
        numpy.ndarray(dtype=int)
            The indices corresponding to the sorting.
        """
        return np.arange(2)

    def local_coordinates(self) -> np.ndarray:
        """Method for computing the 1d vertex coordinates in a local system.

        The first coordinate is set at x=0, the second at x=self.length().

        Returns
        -------
        numpy.ndarray(1 x 2)
            The coordinates of the two vertices in the single local dimension.
        """
        return np.reshape([0, self.length()], (1, 2))

    def compute_centroid(self) -> np.ndarray:
        """Method for computing the centroid of the fracture.

        Returns:
            Array containing the 2D coordinates of the centroid.
        """
        return np.mean(self.pts, axis=1)

    def compute_normal(self) -> np.ndarray:
        """Method computing normal vectors of the fracture

        Returns
        -------
        numpy.ndarray(2 x 1)
            Normal vector of the fracture.
        """

        diff = np.diff(self.pts, axis=1)
        normal = np.array([diff[0], -diff[1]])
        return normal / np.linalg.norm(normal)

    def _check_pts(self):
        """Consistency check for self.pts.

        Defining a LineFracture with more than two points is not supported, as this violates
        assumptions underlying som class methods.

        Raises:
            ValueError if self.pts does not have the expected shape.
            ValueError if the two self.pts are identical.
        """
        if self.pts.shape != (2, 2):
            raise ValueError(
                "pts defining a LineFracture should have dimensions 2 x 2."
            )
        if np.all(np.isclose(self.pts[:, 0], self.pts[:, 1])):
            raise ValueError("Need two distinct pts to define a LineFracture.")

    def length(self):
        """Compute length of the fracture.

        Returns:
            Fracture length as a float.

        """
        diff = np.diff(self.pts, axis=1)
        return np.linalg.norm(diff)
