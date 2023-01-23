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

    def relative_l2_error(
        self,
        grid: pp.GridLike,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        is_scalar: bool,
        is_cc: bool,
    ) -> pp.number:
        """Compute discrete relative L2-error.

        Parameters:
            grid: Either a subdomain grid or a mortar grid.
            true_array: Array containing the true values of a given variable.
            approx_array: Array containing the approximate values of a given variable.
            is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
                vector quantities. For example, ``is_scalar=True`` for pressure, whereas
                ``is_scalar=False`` for displacement.
            is_cc: Whether the variable is associated to cell centers. Use ``False``
                for variables associated to face centers. For example, ``is_cc=True``
                for pressures, whereas ``is_scalar=False`` for subdomain fluxes.

        Returns:
            Discrete relative L2-error between the true and approximated arrays.

        Raises:
            ValueError if a mortar grid is given and ``is_cc=False``.

        """
        # Sanity check
        if isinstance(grid, pp.MortarGrid) and not is_cc:
            raise ValueError("Mortar variables can only be cell-centered.")

        # Obtain proper measure
        if is_cc:
            if is_scalar:
                meas = grid.cell_volumes
            else:
                meas = grid.cell_volumes.repeat(grid.dim)
        else:
            assert isinstance(grid, pp.Grid)
            if is_scalar:
                meas = grid.face_areas
            else:
                meas = grid.face_areas.repeat(grid.dim)

        # Compute error
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))

        return numerator / denominator
