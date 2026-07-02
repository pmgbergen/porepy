"""Collection of metrics.

From plain Euclidean norms to model-specific L2 norms of states and equations.

"""

from __future__ import annotations

from functools import partial
from typing import Optional, cast

import numpy as np

import porepy as pp
from porepy.numerics.ad._grid_entity import GridEntity
from porepy.numerics.ad.operators import DenseArray
from porepy.numerics.ad.operator_space import OperatorSpace


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
        variable_tags: Define a subset of variables to evaluate the metric on. If
            None (default), uses all variables.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        variable_tags: Optional[list[pp.solvers.VariableTag]] = None,
    ) -> None:
        self.model = model
        self.variable_tags = variable_tags
        """Define a subset of variables to evaluate the metric on. If None (default),
        uses all variables.

        """
        self.variable_indexer: pp.ad.VariableIndexer | None = None
        """Indexer for a subset of variables provided by :attr:`variable_tags`.
        Initialized the first time the metric is called. 

        """

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate variable.

        Parameters:
            values: algebraic representation of a mixed-dimensional variable

        Returns:
            dict[str, float]: measure of values for each variable block

        """
        # Lazy initialization of variable indexer.
        if self.variable_indexer is None:
            variable_indexer = self.model.equation_system.variable_indexer
            if self.variable_tags is not None:
                # Restrict to a subset of variables.
                variable_indexer = (
                    variable_indexer.construct_restricted_indexer_from_tags(
                        tags=self.variable_tags, model=self.model
                    )
                )
            self.variable_indexer = variable_indexer
        variable_indexer = self.variable_indexer

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
        equation_tags: Define a subset of equations to evaluate the metric on. If
            None (default), uses all equations.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        equation_tags: Optional[list[pp.solvers.EquationTag]] = None,
    ) -> None:
        self.model = model
        self.equation_tags = equation_tags
        """Define a subset of equations to evaluate the metric on. If None (default),
        uses all equations.

        """
        self.equation_indexer: pp.ad.EquationIndexer | None = None
        """Indexer for a subset of equations provided by :attr:`equation_tags`.
        Initialized the first time the metric is called.

        """

    def __call__(self, values: np.ndarray) -> dict[str, float]:  # type: ignore[override]
        """Compute the Euclidean norm of each separate equation.

        Parameters:
            values: Algebraic representation of a mixed-dimensional equation.

        Returns:
            dict[str, float]: measure of values for each equation block.

        """
        # Lazy initialization of equation indexer.
        if self.equation_indexer is None:
            equation_indexer: pp.ad.EquationIndexer = (
                self.model.equation_system.equation_indexer
            )
            if self.equation_tags is not None:
                # Restrict to a subset of equations.
                equation_indexer = (
                    equation_indexer.construct_restricted_indexer_from_tags(
                        tags=self.equation_tags, model=self.model
                    )
                )
            self.equation_indexer = equation_indexer

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
        domain_and_range = OperatorSpace.from_domains(grids, {GridEntity.cells: 1})
        l2_norm = pp.ad.Function(
            partial(pp.ad.l2_norm, dim), "l2_norm", domain_and_range, domain_and_range
        )
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
        variable_tags: Define a subset of variables to evaluate the metric on. If
            None (default), uses all variables.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        variable_tags: Optional[list[pp.solvers.VariableTag]] = None,
    ) -> None:
        super().__init__(model)
        self.variable_tags = variable_tags
        """Define a subset of variables to evaluate the metric on. If None (default),
        uses all variables.

        """
        self.variable_indexer: pp.ad.VariableIndexer | None = None
        """Indexer for a subset of variables provided by :attr:`variable_tags`.
        Initialized the first time the metric is called. 

        """

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate variable.

        Parameters:
            values: Algebraic representation of a mixed-dimensional variable.

        Returns:
            dict[str, float]: measure of values for each variable block.

        """
        # Lazy initialization of variable indexer.
        if self.variable_indexer is None:
            variable_indexer = self.model.equation_system.variable_indexer
            if self.variable_tags is not None:
                # Restrict to a subset of variables.
                variable_indexer = (
                    variable_indexer.construct_restricted_indexer_from_tags(
                        tags=self.variable_tags, model=self.model
                    )
                )
            self.variable_indexer = variable_indexer
        variable_indexer = self.variable_indexer

        # Sanity check: Ensure that variables are defined on cells.
        for variable in variable_indexer.indices:
            dof_info = variable.operator_range.dof_info

            if (
                not dof_info.get(GridEntity.faces, 0) == 0
                and dof_info.get(GridEntity.nodes, 0) == 0
            ):
                raise NotImplementedError(
                    """VariableBasedLebesgueMetric currently only supports """
                    """variables defined on cells."""
                )

        norms = {v.name: 0.0 for v in variable_indexer.indices}

        for variable, indices in variable_indexer.indices.items():
            variable_values = pp.ad.DenseArray(values[indices])
            dim = variable.dof_info["cells"]
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
        equation_tags: Define a subset of equations to evaluate the metric on. If
            None (default), uses all equations.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        equation_tags: Optional[list[pp.solvers.EquationTag]] = None,
    ) -> None:
        super().__init__(model)
        self.equation_tags = equation_tags
        """Define a subset of equations to evaluate the metric on. If None (default),
        uses all equations.

        """
        self.equation_indexer: pp.ad.EquationIndexer | None = None
        """Indexer for a subset of equations provided by :attr:`equation_tags`.
        Initialized the first time the metric is called.

        """

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Compute the Lebesgue L2 norm of each separate equation.

        Parameters:
            values: algebraic representation of a mixed-dimensional equation

        Returns:
            dict[str, float]: measure of values for each equation block

        """
        # Lazy initialization of equation indexer.
        if self.equation_indexer is None:
            equation_indexer: pp.ad.EquationIndexer = (
                self.model.equation_system.equation_indexer
            )
            if self.equation_tags is not None:
                # Restrict to a subset of equations.
                equation_indexer = (
                    equation_indexer.construct_restricted_indexer_from_tags(
                        tags=self.equation_tags, model=self.model
                    )
                )
            self.equation_indexer = equation_indexer

        equation_system = self.model.equation_system
        norms: dict[str, float] = {}
        for name, grids_dofs in self.equation_indexer.group_by_name().items():
            indices = np.concatenate(list(grids_dofs.values()))
            domains = cast(pp.GridLikeSequence, list(grids_dofs.keys()))
            equation_dim = equation_system.equation_image_size_info[name][
                GridEntity.cells
            ]
            equation_values = values[indices].reshape((equation_dim, -1), order="F")
            cell_weights = np.hstack([domain.cell_volumes for domain in domains])
            space = OperatorSpace.from_domains(
                domains, {GridEntity.cells: equation_dim}
            )
            intensive_equation_values = pp.ad.DenseArray(
                np.linalg.norm(equation_values, ord=2, axis=0) / cell_weights,
                space,
                space,
            )
            norms[name] = self._lebesgue2_norm(intensive_equation_values, 1, domains)

        return norms
