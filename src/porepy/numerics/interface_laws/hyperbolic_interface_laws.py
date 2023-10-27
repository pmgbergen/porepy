"""
Module of coupling laws for hyperbolic equations.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp


class UpwindCoupling:
    def __init__(self, keyword: str) -> None:
        # Keywords for accessing discretization matrices

        # Trace operator for the primary grid
        self.trace_primary_matrix_key = "trace"
        # Inverse trace operator (face -> cell)
        self.inv_trace_primary_matrix_key = "inv_trace"
        # Matrix for filtering upwind values from the primary grid
        self.upwind_primary_matrix_key = "upwind_primary"
        # Matrix for filtering upwind values from the secondary grid
        self.upwind_secondary_matrix_key = "upwind_secondary"
        # Matrix that carries the fluxes
        self.flux_matrix_key = "flux"
        # Discretization of the mortar variable
        self.mortar_discr_matrix_key = "mortar_discr"

        self._flux_array_key = "darcy_flux"

    def key(self) -> str:
        return self.keyword + "_"

    def discretization_key(self):
        return self.key() + pp.DISCRETIZATION

    def ndof(self, intf: pp.MortarGrid) -> int:
        return intf.num_cells

    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: dict,
        data_secondary: dict,
        data_intf: dict,
    ) -> None:
        # First check if the grid dimensions are compatible with the implementation.
        # It is not difficult to cover the case of equal dimensions, it will require
        # trace operators for both grids, but it has not yet been done.
        if sd_primary.dim - sd_secondary.dim not in [1, 2]:
            raise ValueError(
                "Implementation is only valid for grids one dimension apart."
            )

        matrix_dictionary = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]

        # Normal component of the velocity from the higher dimensional grid
        lam_flux: np.ndarray = np.sign(
            data_intf[pp.PARAMETERS][self.keyword][self._flux_array_key]
        )

        # mapping from upper dim cells to faces
        # The mortars always points from upper to lower, so we don't flip any
        # signs.
        # The mapping will be non-zero also for faces not adjacent to
        # the mortar grid, however, we wil hit it with mortar projections, thus kill
        # those elements
        inv_trace_h = np.abs(pp.fvutils.scalar_divergence(sd_primary))
        # We also need a trace-like projection from cells to faces
        trace_h = inv_trace_h.T

        matrix_dictionary[self.inv_trace_primary_matrix_key] = inv_trace_h
        matrix_dictionary[self.trace_primary_matrix_key] = trace_h

        # Find upwind weighting. if flag is True we use the upper weights
        # if flag is False we use the lower weighs
        flag = (lam_flux > 0).astype(float)
        not_flag = 1 - flag

        # Discretizations are the flux, but masked so that only the upstream direction
        # is hit.
        upwind_from_primary = sps.diags(flag)
        upwind_from_secondary = sps.diags(not_flag)

        flux = sps.diags(lam_flux)

        matrix_dictionary[self.upwind_primary_matrix_key] = upwind_from_primary
        matrix_dictionary[self.upwind_secondary_matrix_key] = upwind_from_secondary
        matrix_dictionary[self.flux_matrix_key] = flux

        # Identity matrix, to represent the mortar variable itself
        matrix_dictionary[self.mortar_discr_matrix_key] = sps.eye(intf.num_cells)
