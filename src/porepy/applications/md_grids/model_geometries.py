from __future__ import annotations

import numpy as np

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
