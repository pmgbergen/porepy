"""Contains classes representing fractures of 1D, i.e. manifolds of dimension 1 embedded in 2D.
"""

import numpy as np

import porepy as pp

from .fracture import Fracture


class Fracture2d(Fracture):
    """A class representing linear fracture in 2D."""

    def sort_points(self) -> np.ndarray:
        """Abstract method to sort the vertices as needed for geometric algorithms.

        Returns
        -------
        numpy.ndarray(dtype=int)
            The indices corresponding to the sorting.
        """
        pass

    def local_coordinates(self) -> np.ndarray:
        """Abstract method for computing the vertex coordinates in a local system and its
        local dimension `d`.
        For now, `d` is expected to be `nd` - 1, i.e. the fracture to be of co-dimension 1.

        Returns
        -------
        numpy.ndarray((nd-1) x npt)
            The coordinates of the vertices in local dimensions.
        """
        pass

    def is_convex(self) -> bool:
        """Method for affirming convexity. 1D fractures are made up by two points
        and are thus always convex.

        Returns
        -------
        bool
            True
        """
        return True

    def is_planar(self, tol: float = 1e-4) -> bool:
        """Method for affirming planar fracture. 1D fractures are made up by two
        points and are thus always planar.

        Parameters:
            tol:
                Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns:
            True
        """
        return True

    def compute_centroid(self) -> np.ndarray:
        """Method for computing and returning the centroid of the fracture."""
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

        Defining a Fracture2d with more than two points is not supported, as this violates
        assumptions underlying som class methods.

        Raises:
            ValueError if self.pts does not have the expected shape.
            ValueError if the two self.pts are identical.
        """
        if self.pts.shape != (2, 2):
            raise ValueError("pts defining a Fracture2d should have dimensions 2 x 2.")
        if np.all(np.isclose(self.pts[:, 0], self.pts[:, 1])):
            raise ValueError("Need two distinct pts to define a Fracture2d.")
