"""Collection of metrics.

From plain Euclidean norms to model-specific L2 norms of states and equations.

"""

from functools import partial

import numpy as np

import porepy as pp
from porepy.numerics.ad.operators import DenseArray


class EuclideanMetric:
    """Plain Euclidean norm for variables and equations."""

    def _euclidean_norm(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of an array.

        Parameters:
            values: array to compute the norm of.

        Returns:
            float: measure of values

        """
        return np.linalg.norm(values) / np.sqrt(values.size) if values.size > 0 else 0.0

    def __call__(self, values: np.ndarray) -> float:
        """Compute the Euclidean norm of an array.

        Parameters:
            values: array to compute the norm of.

        Returns:
            float: measure of values

        """
        return self._euclidean_norm(values)


class VariableBasedEuclideanMetric(EuclideanMetric):
    """Plain Euclidean norm for variables and equations, computed per variable and
    equation block.

    """

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            dict[str, float]: measure of values for each variable block

        """
        # Collect variable blocks based on variable names
        variable_names = [
            variable.name for variable in self.model.equation_system.variables
        ]
        variable_blocks = {
            (variable.name, variable.domain): (
                self.model.equation_system.dofs_of([variable])
            )
            for variable in self.model.equation_system.variables
        }
        concatenated_variable_blocks: dict[str, list[int]] = {
            name: [] for name in set(variable_names)
        }
        for (name, _), indices in variable_blocks.items():
            concatenated_variable_blocks[name].extend(indices)

        # Compute norms for each variable block
        norms = {name: 0.0 for name in set(variable_names)}
        for name, indices in concatenated_variable_blocks.items():
            norms[name] = self._euclidean_norm(values[indices])

        return norms


class EquationBasedEuclideanMetric(EuclideanMetric):
    """Plain Euclidean norm for variables and equations, computed per variable and
    equation block.

    """

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate equation.

        Parameters:
            values: Algebraic representation of a mixed-dimensional equation.

        Returns:
            dict[str, float]: measure of values for each equation block.

        """
        norms = {}
        equation_blocks = self.model.equation_system.assembled_equation_indices
        for name, indices in equation_blocks.items():
            norms[name] = self._euclidean_norm(values[indices])
        return norms


class LebesgueMetric:
    def __init__(self, model) -> None:
        self.model = model

    def _lebesgue2_norm(
        self,
        values: DenseArray,
        dim: int,
        grids: pp.GridLikeSequence,
    ) -> float:
        """Compute the Lebesgue L2 norm of a variable or equation.

        Parameters:
            values: Algebraic representation of a mixed-dimensional variable or
                equation.
            dim: Dimension of the variable or equation.
            grids: list of grids over which to integrate

        Returns:
            float: measure of values

        """
        l2_norm = pp.ad.Function(partial(pp.ad.l2_norm, dim), "l2_norm")
        return np.sqrt(
            np.sum(
                self.model.equation_system.evaluate(
                    self.model.volume_integral(
                        l2_norm(values) * l2_norm(values),
                        grids,
                        1,
                    )
                )
            )
        )


class VariableBasedLebesgueMetric(LebesgueMetric):
    """Lebesgue L2 norm for variables and equations, computed per variable block."""

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate variable.

        Parameters:
            values: Algebraic representation of a mixed-dimensional variable.

        Returns:
            dict[str, float]: measure of values for each variable block.

        """
        # Sanity check: Ensure that variables are defined on cells.
        for variable in self.model.equation_system.variables:
            if not variable._faces == 0 and variable._nodes == 0:
                raise NotImplementedError(
                    """VariableBasedLebesgueMetric currently only supports """
                    """variables defined on cells."""
                )

        norms = {v.name: 0.0 for v in self.model.equation_system.variables}
        variable_blocks = {
            (variable.name, variable.domain): (
                self.model.equation_system.dofs_of([variable]),
                variable._cells,  # + variable._faces + variable._nodes,
            )
            for variable in self.model.equation_system.variables
        }
        for (name, sd), (indices, variable_dim) in variable_blocks.items():
            variable_values = pp.ad.DenseArray(values[indices])
            norms[name] += (
                self._lebesgue2_norm(variable_values, variable_dim, [sd]) ** 2
            )
        for name in norms:
            norms[name] = np.sqrt(norms[name])

        return norms


class EquationBasedLebesgueMetric(LebesgueMetric):
    """Lebesgue L2 norm for equations, computed per equation block.

    NOTE: Assumes equations are intensive quantities and defined only on cells.

    """

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        norms = {name: 0.0 for name in self.model.equation_system.equations}

        equation_blocks = {}
        for name in self.model.equation_system.equations:
            if name not in self.model.equation_system.assembled_equation_indices:
                continue
            indices = self.model.equation_system.assembled_equation_indices[name]
            if len(indices) == 0:
                continue
            equation_blocks[name] = (
                indices,
                list(
                    self.model.equation_system.equation_image_space_composition[
                        name
                    ].keys()
                ),
                self.model.equation_system.equation_image_size_info[name]["cells"],
            )
        for name, (indices, sd, eq_dim) in equation_blocks.items():
            if len(sd) == 0:
                norms[name] = 0.0
            else:
                equation_values = values[indices].reshape((eq_dim, -1), order="F")
                cell_weights = np.hstack([_sd.cell_volumes for _sd in sd])
                intensive_equation_values = pp.ad.DenseArray(
                    np.linalg.norm(equation_values, ord=2, axis=0) / cell_weights
                )
                norms[name] = self._lebesgue2_norm(intensive_equation_values, 1, sd)

        return norms
