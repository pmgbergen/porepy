"""Collection of metrics.

From plain Euclidean norms to model-specific L2 norms of states and residuals.

"""

import numpy as np
import porepy as pp
from functools import partial
from typing import Callable


class EuclideanMetric:
    """Plain Euclidean norm for variables and residuals."""

    def _euclidean_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of an array.

        Parameters:
            values: array to compute the norm of.

        Returns:
            float: measure of values

        """
        return np.linalg.norm(values) / np.sqrt(values.size)

    def variable_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of a variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            float: measure of values

        """
        return self._euclidean_norm(values)

    def residual_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of a residual.

        Parameters:
            values: algebraic representation of a mixed-dimensional residual

        Returns:
            float: measure of values

        """
        return self._euclidean_norm(values)


class MultiphysicsEuclideanMetric:
    """Plain Euclidean norm for variables and residuals, computed per variable and
    equation block.

    """

    equation_system: pp.EquationSystem

    def _euclidean_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of an array.

        Parameters:
            values: array to compute the norm of.

        Returns:
            float: measure of values

        """
        return np.linalg.norm(values) / np.sqrt(values.size)

    def variable_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Euclidean norm of each separate variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            dict[str, float]: measure of values for each variable block

        """
        norms = {}
        variable_blocks = {
            variable.name: self.equation_system.dofs_of([variable])
            for variable in self.equation_system.variables
        }
        for name, indices in variable_blocks.items():
            norms[name] = self._euclidean_norm(values[indices])

        return norms

    def residual_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Euclidean norm of each separate residual.

        Parameters:
            values: algebraic representation of a mixed-dimensional residual

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        norms = {}
        equation_blocks = self.equation_system.assembled_equation_indices
        for name, indices in equation_blocks.items():
            norms[name] = self._euclidean_norm(values[indices])

        return norms


class MultiphysicsLebesgueMetric:
    """Lebesgue L2 norm for variables and residuals, computed per variable and
    equation block.

    """

    equation_system: pp.EquationSystem
    volume_integral: Callable[
        [pp.ad.Operator, list[pp.Grid] | list[pp.MortarGrid], int], pp.ad.Operator
    ]

    def _lebesgue2_norm(
        self,
        values: pp.ad.DenseArray,
        l2_norm: pp.ad.Operator,
        subdomains: list[pp.Grid] | list[pp.MortarGrid],
    ) -> float:
        """Compute the Lebesgue L2 norm of a variable or residual.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable or residual
            l2_norm: operator for computing the L2 norm
            subdomains: list of grids or mortar grids over which to integrate

        Returns:
            float: measure of values

        """
        return np.sqrt(
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(l2_norm(values) ** 2, subdomains, 1)
                )
            )
        )

    def variable_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            dict[str, float]: measure of values for each variable block

        """
        norms = {}
        variable_blocks = {
            variable.name: (
                self.equation_system.dofs_of([variable]),
                variable.domain,
                variable._cells + variable._faces + variable._nodes,
            )
            for variable in self.equation_system.variables
        }
        for name, (indices, sd, variable_dim) in variable_blocks.items():
            variable_values = pp.ad.DenseArray(values[indices])
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, variable_dim), "l2_norm")
            norms[name] = self._lebesgue2_norm(variable_values, l2_norm, [sd])

        return norms

    # TODO: Need to decide which formula to use for residual norms.

    def _residual_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate residual.

        Parameters:
            values: algebraic representation of a mixed-dimensional residual

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        # NOTE: Mathematically, this does not make much sense. The equations are already
        # integrated over cells. Thus a combination of np.linalg.norm(..., ord=1)
        # and np.linalg.norm(..., ord=2) over the values would suffice.
        norms = {}
        equation_blocks = {
            name: (
                self.equation_system.assembled_equation_indices[name],
                list(
                    self.equation_system._equation_image_space_composition[name].keys()
                ),
                self.equation_system._equation_image_size_info[name]["cells"],
            )
            for name in self.equation_system._equations
        }
        for name, (indices, sd, eq_dim) in equation_blocks.items():
            residual_values = pp.ad.DenseArray(values[indices])
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, eq_dim), "l2_norm")
            norms[name] = self._lebesgue2_norm(residual_values, l2_norm, sd)

        return norms

    def residual_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate residual.

        Parameters:
            values: algebraic representation of a mixed-dimensional residual

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        norms = {}
        equation_blocks = {
            name: (
                self.equation_system.assembled_equation_indices[name],
                list(
                    self.equation_system._equation_image_space_composition[name].keys()
                ),
                self.equation_system._equation_image_size_info[name]["cells"],
            )
            for name in self.equation_system._equations
        }
        for name, (indices, sd, eq_dim) in equation_blocks.items():
            if len(sd) == 0:
                continue
            residual_values = values[indices].reshape((eq_dim, -1), order="F")
            cell_weights = np.hstack([_sd.cell_volumes for _sd in sd])
            intensive_residual_values = pp.ad.DenseArray(
                np.linalg.norm(residual_values, ord=2, axis=0) / cell_weights
            )
            l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, 1), "l2_norm")
            norms[name] = self._lebesgue2_norm(intensive_residual_values, l2_norm, sd)

        return norms
