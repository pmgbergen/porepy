"""
This module contains utility functions for verification setups.
"""
from __future__ import annotations

import numpy as np

import porepy as pp


class VerificationUtils:
    """This class collects utility methods commonly used in verification setups.

    Note:
        The class is intended to be used as a Mixin, such as the capabilities of
        another class (typically setup or solution storage classes) can be extended.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    stress_keyword: str
    """Keyword used for accessing the parameters of the mechanical subproblem."""

    bc_values_mechanics_key: str
    """Keyword used for accessing the mechanical boundary values."""

    def displacement_trace(
        self, displacement: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """Project the displacement vector onto the faces.

        Parameters:
            displacement: displacement solution of shape (sd.dim * sd.num_cells, ).
            pressure: pressure solution of shape (sd.num_cells, ).

        Returns:
            Trace of the displacement with shape (sd.dim * sd.num_faces, ).

        Raises:
             Exception if the mixed-dimensional grid contains more that one subdomain.

        """
        # Sanity check
        assert len(self.mdg.subdomains()) == 1

        # Rename arguments
        u = displacement
        p = pressure

        # Discretization matrices
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        discr = data[pp.DISCRETIZATION_MATRICES][self.stress_keyword]
        bound_u_cell = discr["bound_displacement_cell"]
        bound_u_face = discr["bound_displacement_face"]
        bound_u_pressure = discr["bound_displacement_pressure"]

        # Mechanical boundary values
        bc_vals = data[pp.STATE][self.bc_values_mechanics_key].copy()

        # Compute trace of the displacement
        trace_u = bound_u_cell * u + bound_u_face * bc_vals + bound_u_pressure * p

        return trace_u
