"""Contains classes representing two-dimensional fractures in 3D."""

from __future__ import annotations

import math
from typing import Optional, Union

import gmsh
import numpy as np
from numpy.typing import ArrayLike

import porepy as pp

from .fracture import Fracture


class EllipticFracture(Fracture):
    """
    Class representing an elliptic fracture embedded in a 3D domain.

    The fracture is represented directly as an OpenCASCADE surface in Gmsh, avoiding
    polygonal approximation. The generated entity is a 2D OCC disk (elliptic face) that
    can be used for meshing and geometric Boolean operations.

    The fracture is defined by its center position, major and minor axes, and its
    spatial orientation given by three rotation angles in radians.

    Coordinate convention:
        The global coordinate system is right-handed in 3D with +x = east, +y = north,
        and +z = up.

    Orientation convention:
        The orientation of the fracture is described using a strike-dip representation.

        The strike angle is defined as the azimuth, i.e., the angle measured clockwise
        from the geographic north (y-axis) in the horizontal plane. This definition is
        widely adopted in geosciences, although alternative notation conventions also
        exist.

        The strike-dip convention follows the right-hand rule in geology: when moving
        along the strike direction, the fracture plane dips downward to the right.

    References:
        - [1] Fossen, H.: Structural Geology (2016).

    Example:
        Fracture centered at ``[0, 1, 0]``, with a ratio of lengths of 2, a rotation
        in the xy-plane of 45 degrees, and an incline of 30 degrees rotated around the
        y-axis (north), due to the strike angle of 0 radians:


        >>> import numpy as np
        >>> frac = EllipticFracture(
        ...     center=np.array([0.0, 0.0, 0.0]),
        ...     major_axis=4.0,
        ...     minor_axis=2.0,
        ...     major_axis_angle=np.pi / 4,
        ...     strike_angle=0,
        ...     dip_angle=np.pi / 6,
        ... )

    Parameters:
        center: ``shape=(3, 1)``
            Center coordinates of fracture.
        major_axis: Length of major axis (radius-like, not diameter).
        minor_axis: Length of minor axis. There are no checks on whether the minor axis
            is less than or equal to the major axis.
        major_axis_angle: Rotation of the major axis from the x-axis in radians.
            Measured before the strike-dip rotation, see below.
        strike_angle: Geological strike angle in radians, measured clockwise from the
            y-axis (north) in the horizontal plane.
        dip_angle: Dip angle in radians, i.e., rotation around the strike direction.
        index: ``default=None``
            Index to be assigned to the fracture.

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
        """Initialize an elliptic fracture in 3D."""
        self.center = center.reshape((-1, 1))
        self.r1 = major_axis
        self.r2 = minor_axis
        self.major_axis_angle = major_axis_angle
        self.strike_angle = strike_angle
        self.dip_angle = dip_angle
        self.index: Optional[int] = index

    def fracture_to_gmsh(self) -> int:
        """
        Create the elliptic fracture as an OpenCASCADE entity in Gmsh and return the
        corresponding 2D surface tag.

        Returns:
            int: Tag of the generated 2D OCC surface.

        """
        # 1) Create an elliptic disk centered at the origin in the XY-plane.
        surface_tag = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, self.r1, self.r2)
        dimTags = [(2, surface_tag)]

        # 2) Rotate around the Z-axis by the in-plane major axis angle.
        gmsh.model.occ.rotate(
            dimTags, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, self.major_axis_angle
        )

        # 3) Rotate around the geological strike direction by the dip angle.
        strike_x = math.sin(self.strike_angle)
        strike_y = math.cos(self.strike_angle)
        strike_z = 0.0

        gmsh.model.occ.rotate(
            dimTags, 0.0, 0.0, 0.0, strike_x, strike_y, strike_z, self.dip_angle
        )

        # 4) Translate the surface to the specified center.
        gmsh.model.occ.translate(
            dimTags, self.center[0][0], self.center[1][0], self.center[2][0]
        )

        return surface_tag

    def __str__(self) -> str:
        """Represent principal axes, normal and centroid."""
        s = f"Elliptic fracture with major axis {self.r1} and minor axis {self.r2}\n"
        s += "Center: \n" + str(self.center) + "\n"
        return s

    def __repr__(self) -> str:
        """Representation of the fracture."""
        return (
            f"EllipticFracture(center={self.center}, major_axis={self.r1}, "
            f"minor_axis={self.r2}, major_axis_angle={self.major_axis_angle}, "
            f"strike_angle={self.strike_angle}, dip_angle={self.dip_angle}, "
            f"index={self.index})"
        )

    def copy(self) -> Fracture:
        """Return a copy of the fracture.

        Returns:
            EllipticFracture: A copy of the fracture.

        """
        return EllipticFracture(
            center=self.center.copy(),
            major_axis=self.r1,
            minor_axis=self.r2,
            major_axis_angle=self.major_axis_angle,
            strike_angle=self.strike_angle,
            dip_angle=self.dip_angle,
            index=self.index,
        )
