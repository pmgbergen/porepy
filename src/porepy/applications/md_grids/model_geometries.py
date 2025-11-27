from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import porepy as pp

from . import domains, fracture_sets


class SquareDomainOrthogonalFractures(pp.PorePyModel):
    """Create a mixed-dimensional grid for a square domain with up to two
    orthogonal fractures.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the square domain.

        The domain size is controlled by the parameter ``domain_size`` in the model
        parameter dictionary.

        """
        # Scale by length unit.
        return self.units.convert_units(self.params.get("domain_size", 1.0), "m")

    def set_fractures(self) -> None:
        """Assigns 0 to 2 fractures to the domain.

        The fractures are defined in
        :meth:`porepy.applications.md_grids.fracture_sets.orthogonal_fractures_2d`, see
        that method for a further description.

        To control the number of fractures, the parameter ``fracture_indices`` can be
        passed to the model, as a list of integers between 0 and 1.

        """
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = fracture_sets.orthogonal_fractures_2d(self.domain_size)
        self._fractures = [all_fractures[i] for i in fracture_indices]

    def set_domain(self) -> None:
        """Set the square domain.

        To control the size of the domain, the parameter ``domain_size`` can be passed
        in the model parameter dictionary.

        """
        self._domain = domains.nd_cube_domain(2, self.domain_size)


class CubeDomainOrthogonalFractures(pp.PorePyModel):
    """Create a mixed-dimensional grid for a cube domain with up to three
    orthogonal fractures.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the cube domain."""
        # Scale by length unit.
        return self.units.convert_units(self.params.get("domain_size", 1.0), "m")

    def set_fractures(self) -> None:
        """Assigns 0 to 3 fractures."""
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = fracture_sets.orthogonal_fractures_3d(self.domain_size)
        self._fractures = [all_fractures[i] for i in fracture_indices]

    def set_domain(self) -> None:
        """Set the cube domain."""
        self._domain = domains.nd_cube_domain(3, self.domain_size)


class RectangularDomainThreeFractures(pp.PorePyModel):
    """A rectangular domain with up to three fractures.

    The domain is `[0, 2] x [0, 1]`.

    The first two fractures are orthogonal, with `x` and `y` coordinates equal to
    0.5, respectively. The third fracture is tilted. The number of fractures is
    controlled by the parameter ``fracture_indices``, which can be any subset of
    [0, 1, 2].

    """

    def set_fractures(self) -> None:
        # Length scale:
        ls = self.units.convert_units(1, "m")

        fracture_indices = self.params.get("fracture_indices", [0])
        fractures = [
            pp.LineFracture(np.array([[0, 2], [0.5, 0.5]]) * ls),
            pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]) * ls),
            pp.LineFracture(np.array([[0.3, 0.7], [0.3, 0.7]]) * ls),
        ]
        self._fractures = [fractures[i] for i in fracture_indices]

    def meshing_arguments(self) -> dict:
        # Divide by length scale:
        ls = self.units.convert_units(1, "m")

        mesh_sizes = {
            # Cartesian: 2 by 8 cells.
            "cell_size_x": 0.25 * ls,
            "cell_size_y": 0.5 * ls,
            # Simplex. Whatever gmsh decides.
            "cell_size_fracture": 0.5 * ls,
            "cell_size_boundary": 0.5 * ls,
            "cell_size_min": 0.2 * ls,
        }
        return mesh_sizes

    def set_domain(self) -> None:
        if not self.params.get("cartesian", False):
            self.params["grid_type"] = "simplex"
        else:
            self.params["grid_type"] = "cartesian"

        # Length scale:
        ls = self.units.convert_units(1, "m")

        # Mono-dimensional grid by default
        phys_dims = np.array([2, 1]) * ls
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self._domain = pp.Domain(box)


class OrthogonalFractures3d(CubeDomainOrthogonalFractures):
    """A 3d domain of the unit cube with up to three orthogonal fractures.

    The fractures have constant `x`, `y` and `z` coordinates equal to 0.5, respectively,
    and are situated in a unit cube domain. The number of fractures is controlled by
    the parameter ``num_fracs``, which can be 0, 1, 2 or 3.

    """

    params: dict
    """Model parameters."""

    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.units.convert_units(1, "m")

        mesh_sizes = {
            "cell_size": 0.5 * ls,
            "cell_size_fracture": 0.5 * ls,
            "cell_size_boundary": 0.5 * ls,
            "cell_size_min": 0.2 * ls,
        }
        return mesh_sizes


class NonMatchingSquareDomainOrthogonalFractures(SquareDomainOrthogonalFractures):
    """Create a non-matching mixed-dimensional grid of a square domain with up to two
    orthogonal fractures.

    The setup is similar to :class:`SquareDomainOrthogonalFractures`, but the
    geometry allows for non-matching grids and different resolution for each grid.
    """

    def create_mdg(self) -> None:
        """Create a non-matching grid.

        The actual grid is created by the mdg_library function for orthogonal fractures.

        """

        # Create a non-matching mixed-dimensional grid. The parameters below are picked
        # from the model, with default values set to mirror those applied in
        # SquareDomainOrthogonalFractures.
        self.mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            grid_type=self.grid_type(),
            meshing_args=self.meshing_arguments(),
            fracture_indices=self.params.get("fracture_indices", [0]),
            fracture_endpoints=self.params.get("fracture_endpoints", None),
            size=self.domain_size,
            non_matching=True,
            **(
                {
                    "fracture_refinement_ratio": self.params.get(
                        "fracture_refinement_ratio", 2
                    ),
                    "interface_refinement_ratio": self.params.get(
                        "interface_refinement_ratio", 2
                    ),
                }
            ),
        )

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)


class SubsurfaceCuboidDomain:
    """Mixin class for cuboid subsurface domains.

    Provides method for setting domain, defining its side lengths and depth calculation.
    The resulting domain extends from surface to bottom in negative z direction. The
    depth calculation can be extended by adding an offset representing the depth of the
    top boundary if needed.

    """

    units: pp.Units
    """Model units."""
    params: dict
    """Model parameters."""

    def domain_sizes(self) -> NDArray[np.float64]:
        """Return the size of the domain in each coordinate direction."""
        return self.units.convert_units(
            self.params.get("domain_sizes", np.array([1.0, 1.0, 1.0])), "m"
        )

    def set_domain(self) -> None:
        """Set the cubic domain."""
        x_size, y_size, z_size = self.domain_sizes()
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": 0.0,
            "ymax": y_size,
            "zmin": -z_size,
            "zmax": 0.0,
        }
        self._domain = pp.Domain(box)

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Calculate depth of points.

        Parameters:
            points: Array of points where depth is to be calculated.

        Returns:
            Depth values for the provided points.

        """
        return self.domain.bounding_box["zmax"] - points[2, :]


class TwoWells3d(SubsurfaceCuboidDomain):
    """A mixin adding two wells to a 3d model.

    By default, one straight vertical well and one kinked well are added to a cubic
    domain. The domain size and well mesh size can be controlled by the parameters
    ``domain_sizes`` and ``well_mesh_size``, respectively.

    A sketch of the setup in the x-z plane is provided in the comments of the method
    :meth:`set_well_network`.
    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    @property
    def well_names(self) -> list[str]:
        """Return the names of the two wells.

        By default, the names are "injection_well" and "production_well". In this class,
        these names are used to tag the wells when creating them. If used e.g. for
        setting boundary conditions or source terms, the user should ensure consistency
        with these names, and may override this property to provide custom names or
        switch the roles of the wells.

        """
        return ["injection_well", "production_well"]

    def set_well_network(self) -> None:
        """Set the two wells.

        See below comment for a sketch of the setup.
        """
        # With constant y coordinates for both wells, the projection in the x-z plane
        # looks roughly as follows, using double lines to indicate the domain
        # boundaries:

        #               w1      w2
        #     ==============================
        #     ||        |        |         ||
        #     ||        |        |         ||
        #     ||        |        |         ||
        #     ||        |        \         ||
        #     ||        |         \        ||
        #     ||        |          \       ||
        #     ||                           ||
        #     ==============================

        # Side lengths of the domain:
        dx, dy, dz = self.domain_sizes()
        # One straight vertical well at (0.35dx, 0.35dy) extending from z=0 to z=-0.8dz.
        well_1 = pp.Well(
            np.array([[0.35 * dx, 0.35 * dx], [0.35 * dy, 0.35 * dy], [0, -0.8 * dz]]),
            tags={"well_name": self.well_names[0]},
        )
        # One kinked well at (0.6dx, 0.65dy), with a kink at z=-0.4dz and terminating at
        # (0.7dx, 0.65dy, -0.8dz).
        well_2 = pp.Well(
            np.array(
                [
                    [0.6 * dx, 0.6 * dx, 0.7 * dx],
                    [0.65 * dy, 0.65 * dy, 0.65 * dy],
                    [0, -0.4 * dz, -0.8 * dz],
                ]
            ),
            tags={"well_name": self.well_names[1]},
        )
        self._wells = [well_1, well_2]

        mesh_size = self.params.get("well_mesh_size", {"mesh_size": 0.1 * dz})
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=self._wells, parameters=mesh_size
        )


class TwoEllipticFractures3d(SubsurfaceCuboidDomain):
    """A mixin adding two elliptic fractures to a 3d model.

    The fractures are defined by their centers, major and minor axes, strike and dip
    angles, and major axis angles. The parameters can be controlled by passing a
    dictionary ``fracture_params`` to the model parameter dictionary. See the property
    :meth:`fracture_params` for details on the available parameters and their default
    values.

    If extending to more than two fractures, the user should override all properties
    defining fracture parameters to return arrays of size (at least) self.num_fractures.
    The case num_fractures < (size of arrays) is allowed, in which case only the first
    num_fractures entries are used.

    TODO: Decide whether to replace properties with a defualt dictionary, i.e.,
    def fracture_params(self) -> dict:
        default_params = {
            "num_fractures": 2,
            "num_points": (10, 10),
            "fracture_major_axes": (0.2, 0.2),
            "fracture_minor_axes": (None, None),
            "strike_angles": (np.pi / 4, np.pi / 4),
            "dip_angles": (np.pi / 2, np.pi / 2),
            "major_axis_angles": (0.0, 0.0),
        }
        user_params = self.params.get("fracture_params", {})
        default_params.update(user_params)
        return default_params
    The parameters could then be retrieved as self.fracture_params["num_points"], etc.
    This would reduce the number of properties, but possibly make the code less
    explicit.
    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    @property
    def fracture_params(self) -> dict:
        """Return fracture parameters."""
        return self.params.get("fracture_params", {})

    @property
    def num_fracture_points(self) -> NDArray[np.int32]:
        """Return the number of points per fracture."""
        return self.fracture_params.get("num_points", np.array((10, 10)))

    @property
    def fracture_major_axes(self) -> NDArray[np.float64]:
        """Return the major axes of the two fractures.

        Returns:
            Array with the major axes of the two fractures.
        """
        default_axes = np.array([0.2, 0.2])
        axes = self.fracture_params.get("fracture_major_axes", default_axes)
        axes = self.units.convert_units(axes, "m")
        return axes

    @property
    def fracture_minor_axes(self) -> NDArray[np.float64]:
        """Return the minor axes of the two fractures.

        If not specified, the minor axes are set equal to the major axes, resulting in
        disk-shaped fractures.

        Returns:
            Array with the minor axes of the two fractures.
        """
        default_axes = self.fracture_major_axes
        axes = self.fracture_params.get("fracture_minor_axes", default_axes)
        if axes is None:
            axes = default_axes
        else:
            axes = self.units.convert_units(axes, "m")
        return axes

    @property
    def strike_angles(self) -> NDArray[np.float64]:
        """Return the strike angles of the two fractures.

        If not specified, both strike angles are set to `pi/4`.

        Returns:
            Array with the strike angles of the two fractures.
        """
        default_strikes = np.array((np.pi / 4, np.pi / 4))
        strikes = self.fracture_params.get("strike_angles", default_strikes)
        return strikes

    @property
    def dip_angles(self) -> NDArray[np.float64]:
        """Return the dip angles of the two fractures.

        If not specified, both dip angles are set to `pi/2`.

        Returns:
            Array with the dip angles of the two fractures.
        """
        default_dips = np.array((np.pi / 2, np.pi / 2))
        dips = self.fracture_params.get("dip_angles", default_dips)
        return dips

    @property
    def major_axis_angles(self) -> NDArray[np.float64]:
        """Return the major axis angles of the two fractures.

        If not specified, both major axis angles are set to `0`.

        Returns:
            Array with the major axis angles of the two fractures.
        """
        default_angles = np.array((0.0, 0.0))
        angles = self.fracture_params.get("major_axis_angles", default_angles)
        return angles

    @property
    def fracture_centers(self) -> tuple[np.ndarray, np.ndarray]:
        dx, dy, dz = self.domain_sizes()
        center_1 = np.array([0.35 * dx, 0.35 * dy, -0.6 * dz])
        center_2 = np.array([0.65 * dx, 0.65 * dy, -0.6 * dz])
        return center_1, center_2

    @property
    def num_fractures(self) -> int:
        """Return the number of fractures."""
        return self.fracture_params.get("num_fractures", 2)

    def set_fractures(self):
        """Set the two elliptic fractures."""
        self._fractures = []
        for i in range(self.num_fractures):
            f = pp.create_elliptic_fracture(
                center=self.fracture_centers[i],
                strike_angle=self.strike_angles[i],
                dip_angle=self.dip_angles[i],
                major_axis=self.fracture_major_axes[i],
                minor_axis=self.fracture_minor_axes[i],
                major_axis_angle=self.major_axis_angles[i],
                num_points=self.num_fracture_points[i],
            )
            self._fractures.append(f)
