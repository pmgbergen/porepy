"""
This module contains utility functions for verification setups.
"""
from __future__ import annotations

from typing import Callable

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


class VerificationDataSaving(pp.DataSavingMixin):
    """Class to store relevant data for a generic verification setup."""

    _nonlinear_iteration: int
    """Number of non-linear iterations needed to solve the system. Used only as an
    indicator to avoid saving the initial conditions.

    """

    _is_time_dependent: Callable
    """Wheter the problem is time-dependent."""

    results: list
    """List of objects containing the results of the verification."""

    relative_l2_error: Callable
    """Method that computes the discrete relative L2-error between an exact solution
    and an approximate solution on a given grid. The method is provided by the mixin
    class :class:`porepy.applications.verification_setups.VerificationUtils`.

    """

    def save_data_time_step(self) -> None:
        """Save data to the `results` list."""
        if not self._is_time_dependent():  # stationary problem
            if self._nonlinear_iteration > 0:  # avoid saving initial condition
                collected_data = self.collect_data()
                self.results.append(collected_data)
        else:  # time-dependent problem
            t = self.time_manager.time  # current time
            scheduled = self.time_manager.schedule[1:]  # scheduled times except t_init
            if any(np.isclose(t, scheduled)):
                collected_data = self.collect_data()
                self.results.append(collected_data)

    def collect_data(self):
        """Collect relevant data for the verification setup."""
        raise NotImplementedError()
