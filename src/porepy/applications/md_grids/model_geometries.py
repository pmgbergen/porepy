from __future__ import annotations

import numpy as np

import porepy as pp

from . import domains, fracture_sets


class SquareDomainOrthogonalFractures(pp.ModelGeometry):
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


class CubeDomainOrthogonalFractures(pp.ModelGeometry):
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


class RectangularDomainThreeFractures(pp.ModelGeometry):
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

    def set_geometry(self) -> None:
        """Define geometry and create a non-matching mixed-dimensional grid.

        We here make a non-matching mixed-dimensional grid in the sense that neither of
        the fracture grids, interface grids or the rock grid are matching. This is done
        by refining the fracture and interface grids.

        """

        # The fracture grids are replaced by a refined version of the fractures which
        # are already present in the grid. The interface grids are replaced by first
        # creating a new mixed-dimensional grid with a higher resolution. The new
        # mixed-dimensional grid will "donate" its interface grids to the original
        # mixed-dimensional grid.

        # There are a few things to be aware of when creating a non-matching
        # mixed-dimensional grid:
        #     * If _both_ the interface and fracture grids are to be refined/coarsened,
        #       you must be aware of:
        #         * If you first refine (and replace) the fracture grids, you must also
        #           update the fracture numbering such that the new fracture grids have
        #           the same fracture number as the old ones. This is because the fracture
        #           numbering is used to identify the correct interface grids later.
        #           TODO: Make sure that the refine-grid-functions also copy frac_num
        #           attribute such that the abovementioned is no longer a problem.
        #     * Ensure that the "donor" mixed-dimensional grid is physically the same as
        #       the "recipient" mixed-dimensional grid. Fractures must be located at the
        #       same place and the physical dimension of the grids must be the same.

        # First set the geometry and create a matching mixed-dimensional grid.
        super().set_geometry()

        # Refine and replace fracture grids. First we fetch the old fracture grids:
        old_fracture_grids = self.mdg.subdomains(dim=1)

        # Ratios which we want to refine the fracture grids with. Default, if no ratios
        # are provided, is to not refine at all.
        ratios = self.params.get(
            "fracture_refinement_ratios", np.ones(len(old_fracture_grids))
        )

        # The actual refinement of the fracture grids.
        new_fracture_grids = [
            pp.refinement.refine_grid_1d(g=old_grid, ratio=ratio)
            for old_grid, ratio in zip(old_fracture_grids, ratios)
        ]

        # Create a mapping between the old and new fracture grids.
        grid_map = dict(zip(old_fracture_grids, new_fracture_grids))

        # Refine and replace interface grids. This is more complicated, since directly
        # refining the interface grids is not straightforward. Instead, we create a new
        # mixed-dimensional grid and donate the interface grids to the original
        # mixed-dimensional grid.
        # We first create a new and more refined mixed-dimensional grid.
        def mdg_func(self, cell_size) -> pp.MixedDimensionalGrid:
            """Generate a refined version of an already existing mixed-dimensional grid.

            Parameters:
                cell_size: Cell size of the refined grid.
            """
            # Construct a new grid with the same fractures as the original grid.
            # Directly call the meshing function instead of using higher-level methods
            # (e.g., those being called in self.set_geometry), since we don't want to
            # interfere with the original grid at this stage.
            fracs = [self.fractures[i].pts for i in range(len(self.fractures))]

            domain = self.domain.bounding_box
            nx = int(domain["xmax"] / cell_size)
            ny = int(domain["ymax"] / cell_size)

            md_grid = pp.meshing.cart_grid(
                fracs, np.array([nx, ny]), physdims=[domain["xmax"], domain["ymax"]]
            )
            return md_grid

        interface_refinement_ratio = self.params.get("interface_refinement_ratio", 1)
        cell_size_old_grid = self.meshing_arguments()["cell_size"]
        cell_size_new_grid = cell_size_old_grid / interface_refinement_ratio

        mdg_new = mdg_func(self, cell_size_new_grid)

        intf_map = {}
        # First we loop through all the interfaces in the new mixed-dimensional grid
        # (donor).
        for intf in mdg_new.interfaces(dim=1):
            # Then, for each interface, we fetch the secondary grid which belongs to it.
            _, g_sec = mdg_new.interface_to_subdomain_pair(intf)

            # We then loop through all the interfaces in the original grid (recipient).
            for intf_coarse in self.mdg.interfaces(dim=1):
                # Fetch the secondary grid of the interface in the original grid.
                _, g_sec_coarse = self.mdg.interface_to_subdomain_pair(intf_coarse)

                # Checkinc the fracture number of the secondary grid in the recipient
                # mdg. If they are the same, i.e., the fractures are the same ones, we
                # update the interface map.
                if g_sec_coarse.frac_num == g_sec.frac_num:
                    intf_map.update({intf_coarse: intf})

        # Finally replace the subdomains and interfaces in the original
        # mixed-dimensional grid. Both can be done at the same time:
        self.mdg.replace_subdomains_and_interfaces(sd_map=grid_map, intf_map=intf_map)

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)
