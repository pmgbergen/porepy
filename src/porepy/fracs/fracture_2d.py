"""Contains classes representing fractures of 1D, i.e. manifolds of dimension 1 embedded in 2D.
"""

import numpy as np

import porepy as pp
from .fracture import Fracture

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

        Parameters
        ----------
            tol : float
                Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns
        -------
        bool
            True
        """
        return True

    def compute_centroid(self) -> np.ndarray:
        """Method for computing and returning the centroid of the fracture.
        """
        return np.array([sum(self.pts[i])/2 for i in range(len(self.pts))])

    def compute_normal(self) -> np.ndarray:
        """Method computing normal vectors of the fracture
        
        Returns
        -------
            numpy.ndarray(3 x 2)
                Two normal vectors of the fracture.       
        """
        return pp.map_geometry.compute_normals_1d(self.pts)