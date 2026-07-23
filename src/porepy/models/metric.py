"""Collection of metrics.

From plain Euclidean norms to model-specific L2 norms of states and equations.

"""

from functools import partial
from typing import Optional, cast

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
    """Plain Euclidean norm for variables, computed per variable block.

    Parameters:
        model: The model used to compute the metric.
        variable_indexer: Define a subset of variables to evaluate the metric on. If
            None (default), uses all variables.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        variable_indexer: Optional[pp.ad.VariableIndexer] = None,
    ) -> None:
        self.model = model
        if variable_indexer is None:
            variable_indexer = model.equation_system.variable_indexer
        self.variable_indexer = variable_indexer

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            dict[str, float]: measure of values for each variable block

        """
        # Compute norms for each variable block
        norms: dict[str, float] = {}
        for name, domains_dofs in self.variable_indexer.group_by_name().items():
            indices = np.concatenate(list(domains_dofs.values()))
            norms[name] = self._euclidean_norm(values[indices])

        return norms


class EquationBasedEuclideanMetric(EuclideanMetric):
    """Plain Euclidean norm for equations, computed per equation block.

    Parameters:
        model: The model used to compute the metric.
        equation_indexer: Define a subset of equations to evaluate the metric on. If
            None (default), uses all equations.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        equation_indexer: Optional[pp.ad.EquationIndexer] = None,
    ) -> None:
        self.model = model
        if equation_indexer is None:
            equation_indexer = model.equation_system.equation_indexer
        self.equation_indexer = equation_indexer

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate equation.

        Parameters:
            values: Algebraic representation of a mixed-dimensional equation.

        Returns:
            dict[str, float]: measure of values for each equation block.

        """
        norms = {}
        for name, domains_dofs in self.equation_indexer.group_by_name().items():
            indices = np.concatenate(list(domains_dofs.values()))
            norms[name] = self._euclidean_norm(values[indices])
        return norms


class LebesgueMetric:
    def __init__(self, model: pp.PorePyModel) -> None:
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
    """Lebesgue L2 norm for variables, computed per variable block.

    Parameters:
        model: The model used to compute the metric.
        variable_indexer: Define a subset of variables to evaluate the metric on. If
            None (default), uses all variables.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        variable_indexer: Optional[pp.ad.VariableIndexer] = None,
    ) -> None:
        super().__init__(model)
        if variable_indexer is None:
            variable_indexer = model.equation_system.variable_indexer
        self.variable_indexer = variable_indexer

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate variable.

        Parameters:
            values: Algebraic representation of a mixed-dimensional variable.

        Returns:
            dict[str, float]: measure of values for each variable block.

        """
        equation_system = self.model.equation_system
        variable_indexer = equation_system.variable_indexer

        # Sanity check: Ensure that variables are defined on cells.
        for variable in variable_indexer.variable_dofs:
            if not variable._faces == 0 and variable._nodes == 0:
                raise NotImplementedError(
                    """VariableBasedLebesgueMetric currently only supports """
                    """variables defined on cells."""
                )

        norms = {v.name: 0.0 for v in self.model.equation_system.variables}

        for variable, indices in variable_indexer.variable_dofs.items():
            variable_values = pp.ad.DenseArray(values[indices])
            dim = variable.dof_info["cells"]  # + variable._faces + variable._nodes,
            domains: pp.GridLikeSequence = [variable.domain]  # type: ignore[assignment]
            norms[variable.name] += (
                self._lebesgue2_norm(variable_values, dim, domains) ** 2
            )
        for name in norms:
            norms[name] = np.sqrt(norms[name])

        return norms


class EquationBasedLebesgueMetric(LebesgueMetric):
    """Lebesgue L2 norm for equations, computed per equation block.

    NOTE: Assumes equations are intensive quantities and defined only on cells.

    Parameters:
        model: The model used to compute the metric.
        equation_indexer: Define a subset of equations to evaluate the metric on. If
            None (default), uses all equations.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        equation_indexer: Optional[pp.ad.EquationIndexer] = None,
    ) -> None:
        super().__init__(model)
        if equation_indexer is None:
            equation_indexer = model.equation_system.equation_indexer
        self.equation_indexer = equation_indexer

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        equation_system = self.model.equation_system
        norms: dict[str, float] = {}
        for name, grids_dofs in self.equation_indexer.group_by_name().items():
            indices = np.concatenate(list(grids_dofs.values()))
            domains = cast(pp.GridLikeSequence, list(grids_dofs.keys()))
            equation_dim = equation_system.equation_image_size_info[name]["cells"]
            equation_values = values[indices].reshape((equation_dim, -1), order="F")
            cell_weights = np.hstack([domain.cell_volumes for domain in domains])
            intensive_equation_values = pp.ad.DenseArray(
                np.linalg.norm(equation_values, ord=2, axis=0) / cell_weights
            )
            norms[name] = self._lebesgue2_norm(intensive_equation_values, 1, domains)

        return norms
