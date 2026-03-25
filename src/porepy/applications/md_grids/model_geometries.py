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
        """A default grid type ('simplex') is set if none is provided."""

        self.params["grid_type"] = self.params.get("grid_type", "simplex")

        # Length scale:
        ls = self.units.convert_units(1, "m")

        # Mono-dimensional grid by default
        phys_dims = np.array([2, 1]) * ls
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self._domain = pp.Domain(box)


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

    domain: pp.Domain
    """Model domain."""
    nd: int
    """Number of spatial dimensions."""
    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    def domain_sizes(self) -> NDArray[np.float64]:
        """Return the size of the domain in each of the three coordinate directions."""
        # Hard-coded to 3 instead of self.nd since nd is not necessarily set when this
        # method is (first) called. Justified since this is a *cuboid* domain.
        return self.units.convert_units(
            self.params.get("domain_sizes", np.ones(3, dtype=float)), "m"
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
        # TODO: Revert to kinked wells once well geometry processing is reworked. The
        # sketch below is kept for reference.

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
        # One straight vertical well at (0.4dx, 0.4dy) extending from z=0 to z=-0.8dz.
        well_1 = pp.Well(
            np.array([[0.4 * dx, 0.4 * dx], [0.4 * dy, 0.4 * dy], [0, -0.8 * dz]]),
            tags={"well_name": self.well_names[0]},
        )
        # One well at (0.6dx, 0.6dy).
        well_2 = pp.Well(
            np.array(
                [
                    [0.6 * dx, 0.6 * dx],
                    [0.6 * dy, 0.6 * dy],
                    [0, -0.8 * dz],
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

    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    def fracture_params(self) -> dict:
        """Return fracture parameters with defaults.

        The available parameters are:
            - num_fractures: Number of fractures (default 2)
            - fracture_major_axes: Major axes of the fractures (default [0.2, 0.2])
            - fracture_minor_axes: Minor axes of the fractures (default equal to
                major axes, whether explicitly provided or not)
            - strike_angles: Strike angles of the fractures (default [pi/4, pi/4])
            - dip_angles: Dip angles of the fractures (default [pi/2, pi/2])
            - major_axis_angles: Major axis angles of the fractures (default [0.0, 0.0])

        The fracture axes are scaled by the minimum of the domain sizes. For adjusting
        the fracture centers, the user should override the property
        :meth:`fracture_centers`.

        Returns:
            A dictionary with fracture parameters.

        """
        default_params = {
            "num_fractures": 2,
            "fracture_major_axes": np.array([0.2, 0.2]),
            "strike_angles": np.array([np.pi / 4, np.pi / 4]),
            "dip_angles": np.array([np.pi / 2, np.pi / 2]),
            "major_axis_angles": np.array([0.0, 0.0]),
        }
        user_params = self.params.get("fracture_params", {})
        default_params.update(user_params)
        if "fracture_minor_axes" not in default_params:
            default_params["fracture_minor_axes"] = default_params[
                "fracture_major_axes"
            ]
        return default_params

    @property
    def fracture_minor_axes(self) -> np.ndarray:
        params = self.fracture_params()
        # Scale minor axes by the minimum domain size.
        size = min(self.domain_sizes())
        return params["fracture_minor_axes"] * size

    @property
    def fracture_major_axes(self) -> np.ndarray:
        params = self.fracture_params()
        # Scale major axes by the minimum domain size.
        size = min(self.domain_sizes())
        return params["fracture_major_axes"] * size

    @property
    def fracture_centers(self) -> tuple[np.ndarray, np.ndarray]:
        dx, dy, dz = self.domain_sizes()
        center_1 = np.array([0.4 * dx, 0.4 * dy, -0.6 * dz])
        center_2 = np.array([0.6 * dx, 0.6 * dy, -0.6 * dz])
        return center_1, center_2

    def set_fractures(self):
        """Set the elliptic fractures as defined in the fracture parameters and the
        fracture centers method."""
        self._fractures = []
        params = self.fracture_params()
        for i in range(params["num_fractures"]):
            f = pp.EllipticFracture(
                center=self.fracture_centers[i],
                strike_angle=params["strike_angles"][i],
                dip_angle=params["dip_angles"][i],
                major_axis=self.fracture_major_axes[i],
                minor_axis=self.fracture_minor_axes[i],
                major_axis_angle=params["major_axis_angles"][i],
            )
            self._fractures.append(f)
