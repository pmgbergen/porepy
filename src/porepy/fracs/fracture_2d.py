"""Contains classes representing fractures in 1D, i.e. manifolds of dimension 1 embedded in 2D.
"""

from typing import Optional

import numpy as np

import porepy as pp
from fracture import Fracture

class Fracture2d(Fracture):
    """A class representing linear fracture in 2D.
    
    """
    
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
        rotation = pp.map_geometry.project_line_matrix(self.p)
        points_2d = rotation.dot(self.p)

        return points_2d[:2]

    def is_convex(self) -> bool:
        """Abstract method for implementing a convexity check.

        Returns
        -------
        bool
            True if the fracture is convex, False otherwise.
        """
        return True

    def is_planar(self, tol: float = 1e-4) -> bool:
        """Abstract method to check if the fracture is planar.

        Parameters
        ----------
            tol : float
                Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns
        -------
        bool
            True if the fracture is planar, False otherwise.
        """
        return True

    def compute_centroid(self) -> np.ndarray:
        """Method for computing and returning the centroid of the fracture.
        """
        return np.array([sum(self.p[i])/2 for i in range(len(self.p))])

    def compute_normal(self) -> np.ndarray:
        """See parent class docs."""
        return pp.map_geometry.compute_normals_1d(self.p)[:,0]