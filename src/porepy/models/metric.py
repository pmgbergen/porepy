"""Collection of metrics.

From plain Euclidean norms to model-specific L2 norms of states and equations.

"""

from functools import partial
from typing import Callable

import numpy as np

import porepy as pp


class EuclideanMetric:
    """Plain Euclidean norm for variables and equations."""

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

    def equation_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of an equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

        Returns:
            float: measure of values

        """
        return self._euclidean_norm(values)


class MultiphysicsEuclideanMetric:
    """Plain Euclidean norm for variables and equations, computed per variable and
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

    def equation_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Euclidean norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        norms = {}
        equation_blocks = self.equation_system.assembled_equation_indices
        for name, indices in equation_blocks.items():
            norms[name] = self._euclidean_norm(values[indices])

        return norms


class MultiphysicsLebesgueMetric:
    """Lebesgue L2 norm for variables and equations, computed per variable and
    equation block.

    """

    equation_system: pp.EquationSystem
    volume_integral: Callable[
        [pp.ad.Operator, list[pp.Grid] | list[pp.MortarGrid], int], pp.ad.Operator
    ]

    def _lebesgue2_norm(
        self,
        values: pp.ad.DenseArray,
        dim: int,
        subdomains: list[pp.Grid | pp.MortarGrid | pp.BoundaryGrid],
    ) -> float:
        """Compute the Lebesgue L2 norm of a variable or equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable or equation
            dim: int, dimension of the variable or equation
            subdomains: list of grids or mortar grids over which to integrate

        Returns:
            float: measure of values

        """
        # Complicated way of making sure subdomains is actually of type
        # list[pp.Grid] | list[pp.MortarGrid]:
        grids: list[pp.Grid] = [sd for sd in subdomains if isinstance(sd, pp.Grid)]
        mortar_grids: list[pp.MortarGrid] = [
            sd for sd in subdomains if isinstance(sd, pp.MortarGrid)
        ]
        boundary_grids: list[pp.BoundaryGrid] = [
            sd for sd in subdomains if isinstance(sd, pp.BoundaryGrid)
        ]
        assert len(boundary_grids) == 0, "Boundary grids not supported yet."
        assert len(grids) == 0 or len(mortar_grids) == 0, (
            "Mixing grids and mortar grids not supported yet."
        )
        _subdomains = grids if len(grids) > 0 else mortar_grids

        l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, dim), "l2_norm")
        return np.sqrt(
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(
                        l2_norm(values) * l2_norm(values),
                        _subdomains,
                        1,
                    )
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
            norms[name] = self._lebesgue2_norm(variable_values, variable_dim, [sd])

        return norms

    # TODO: Need to decide which formula to use for equation norms.

    def _equation_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

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
            equation_values = pp.ad.DenseArray(values[indices])
            norms[name] = self._lebesgue2_norm(equation_values, eq_dim, sd)

        return norms

    def equation_norm(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

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
            equation_values = values[indices].reshape((eq_dim, -1), order="F")
            cell_weights = np.hstack([_sd.cell_volumes for _sd in sd])
            intensive_equation_values = pp.ad.DenseArray(
                np.linalg.norm(equation_values, ord=2, axis=0) / cell_weights
            )
            norms[name] = self._lebesgue2_norm(intensive_equation_values, 1, sd)

        return norms
