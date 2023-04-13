from __future__ import annotations

from . import domains, fracture_sets

import porepy as pp


class SquareDomainOrthogonalFractures:
    """Create a mixed-dimensional grid for a square domain with up to two
    orthogonal fractures.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    params: dict
    """Parameters for the model geometry. Entries relevant for this mixin are:
        - domain_size: The side length of the square domain.
        - fracture_indices: List of fracture indices to be included in the grid.

    """

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the square domain."""
        # Scale by length unit.
        return self.params.get("domain_size", 1) / self.units.m

    def set_fracture_network(self) -> None:
        """Set the fracture network.

        The fractures are assumed to be orthogonal to the domain boundaries.

        """
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = fracture_sets.orthogonal_fractures_2d(self.domain_size)
        fractures = [all_fractures[i] for i in fracture_indices]
        domain = domains.nd_cube_domain(2, self.domain_size)
        self.fracture_network = pp.FractureNetwork2d(fractures, domain)
