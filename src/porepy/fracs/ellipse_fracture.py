"""Contains classes representing two-dimensional fractures.

That is, manifolds of dimension 2 embedded in 3D.

"""

from __future__ import annotations

from typing import Optional, Union

import gmsh
import numpy as np
from numpy.typing import ArrayLike

import porepy as pp

from .fracture import Fracture


import math
import numpy as np
import gmsh

class EllipticFracture:
    """
    Class representing an elliptic fracture embedded in a 3D domain.

    The fracture is represented directly as an OpenCASCADE surface in Gmsh,
    avoiding polygonal approximation. The generated entity is a 2D OCC disk
    (elliptic face) that can be used for meshing and geometric Boolean operations.

    The fracture is defined by its center position, major and minor axes,
    and its spatial orientation given by three rotation angles in radians. 

    Example:
        >>> import numpy as np
        >>> import gmsh
        >>> gmsh.initialize()
        >>> frac = EllipticFracture(
        ...     center=np.array([0.0, 0.0, 0.0]),
        ...     major_axis=5.0,
        ...     minor_axis=2.0,
        ...     major_axis_angle=np.pi / 6,
        ...     strike_angle=np.pi / 4,
        ...     dip_angle=np.pi / 8,
        ... )
        >>> tag = frac.fracture_to_gmsh_3D()
        >>> gmsh.finalize()

    """

    def __init__(
        self,
        center: np.ndarray,
        major_axis: float,
        minor_axis: float,
        major_axis_angle: float,
        strike_angle: float,
        dip_angle: float,
        index: int | None = None,
    ):
        """
        Initialize an elliptic fracture in 3D.

        Parameters:
            center: Array of ``shape=(3, 1)``
                Coordinates of the fracture center in 3D space.
            major_axis: Length of the major semi-axis (radius-like, not diameter).
            minor_axis: Length of the minor semi-axis.
            major_axis_angle: rotation of the major axis in radians from the x-axis 
                before strike-dip rotation.
            strike_angle: the direction of the strike line (rotation axis for dip)
                in radians, measured from the x-axis in the xy-plane.
            dip_angle: rotation of the fracture plane around the strike line in 
                radians, defining the inclination of the fracture.
            index: Optional integer index to be assigned to the fracture.
        """
        self.center = np.asarray(center)
        self.r1 = float(major_axis)
        self.r2 = float(minor_axis)
        self.major_axis_angle = float(major_axis_angle)
        self.strike_angle = float(strike_angle)
        self.dip_angle = float(dip_angle)
        self.index = index

    def set_index(self, index: int) -> None:
        """Set the index of this fracture.

        Parameters:
            index: Index.

        """
        self.index = index

    def fracture_to_gmsh_3D(self) -> int:
        """
        Create the elliptic fracture as an OpenCASCADE entity in Gmsh
        and return the corresponding 2D surface tag.

        The procedure follows the same geometric logic as in
        `create_elliptic_fracture` (polygonal version), but uses the
        OpenCASCADE kernel for exact geometry definition.

        Steps:
            1. Create an elliptic disk at the origin in the XY-plane
               with semi-axes (r1, r2).
            2. Rotate around the Z-axis by `major_axis_angle`
               to set the in-plane orientation.
            3. Rotate around the strike axis (defined by `strike_angle`)
               by `dip_angle` to impose the dip inclination.
            4. Translate the fracture to its target center position.

        Returns:
            int: Tag of the generated 2D OCC surface.

        Notes:
            The rotation order and axis definitions match those 
            in PorePy’s `create_elliptic_fracture`, ensuring consistent geometry.

        """
        # 1) Create an elliptic disk centered at the origin in the XY-plane
        surface_tag = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, self.r1, self.r2)
        dimTags = [(2, surface_tag)]

        # 2) Rotate around the Z-axis by the in-plane major axis angle
        gmsh.model.occ.rotate(
            dimTags,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            self.major_axis_angle,
        )

        # 3) Rotate around the strike direction by the dip angle
        strike_x = math.cos(self.strike_angle)
        strike_y = math.sin(self.strike_angle)
        strike_z = 0.0

        gmsh.model.occ.rotate(
            dimTags,
            0.0, 0.0, 0.0,
            strike_x,
            strike_y,
            strike_z,
            self.dip_angle,
        )
        
        # 4) Translate the surface to the specified center
        gmsh.model.occ.translate(dimTags, self.center[0], self.center[1], self.center[2])

        # gmsh.model.occ.synchronize()

        return surface_tag


    