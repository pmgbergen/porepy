"""Indexers for discretized systems of equations and variables, the values of
which are arranged in a contiguous array.

Used by `EquationSystem` and nonlinear solvers.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

import porepy as pp

__all__ = [
    "EquationOnDomain",
    "VariableIndexer",
    "EquationIndexer",
    "EquationSystemIndexer",
]


@dataclass(frozen=True)
class EquationOnDomain:
    """An identifier of a single equation defined on a single domain.

    Mirrors the `pp.ad.Variable` design, which is used as an identifier for a single
    variable defined on a single domain.

    """

    name: str
    domain: pp.GridLike


class Indexer[EquationOrVariableType: (EquationOnDomain, pp.ad.Variable)]:
    """A variable indexer determines the arrangement of DoFs corresponding to multiple
    variables on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    variables), a new indexer needs to be constructed.

    """

    def __init__(
        self, operators_to_dofs: dict[EquationOrVariableType, np.ndarray]
    ) -> None:
        self.operators_to_dofs: dict[EquationOrVariableType, np.ndarray] = (
            operators_to_dofs
        )
        """An ordered mapping of atomic variables to their DoF indices. The keys are
        ordered, in a sense that if key A goes before key B, DoFs of key A are located
        before DoFs of key B.

        """
        self.num_dofs: int = sum(x.size for x in self.operators_to_dofs.values())

    def projection_indices(self, operators: list[EquationOrVariableType]) -> np.ndarray:
        """Create a projection index array from the system vector represented
        by this indexer to the requested subspace.

        The order of the variables in the projection is defined by the input.

        Parameters:
            variables: Input for which the subspace is requested.

        Returns:
            an index array of `shape=(M,)`, where `0 <= M <= num_dofs`.

        """
        projections = []
        for operator in operators:
            dofs = self.operators_to_dofs.get(operator, None)
            if dofs is None:
                raise ValueError(
                    f"Requested operator is not known to this indexer: {operator}."
                )
            projections.append(dofs)
        return (
            np.concatenate(projections)
            if len(projections) > 0
            else np.empty(0, dtype=int)
        )

    def construct_restricted_indexer(
        self, operators: list[EquationOrVariableType]
    ) -> "Indexer[EquationOrVariableType]":
        """Constructs a new indexer based on requested subset of variables.

        The order of the new indexer is defined by the input.

        Parameters:
            variables: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested variable is not known to this indexer.

        Returns:
            A new instance of VariableIndexer.

        """
        new_operators_to_dofs: dict[EquationOrVariableType, np.ndarray] = {}
        offset = 0
        for operator in operators:
            dofs = self.operators_to_dofs.get(operator, None)
            if dofs is None:
                raise ValueError(
                    f"Requested operator is not known to this indexer: {operator}."
                )
            new_operators_to_dofs[operator] = np.arange(dofs.size) + offset
            offset += dofs.size

        if len(new_operators_to_dofs) != len(operators):
            raise ValueError(f"Requested operators are duplicated: {operators}.")

        return Indexer(operators_to_dofs=new_operators_to_dofs)

    def group_by_name(self) -> dict[str, dict[pp.GridLike, np.ndarray]]:
        """Group :attr:`variable_dofs` by variable names.

        Domains with no dofs are ignored.

        Offset between variables is assumed.

        Return:
            A nested mapping "variable_name" -> "domain" -> "dofs".

        """
        result: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        for operator, dofs in self.operators_to_dofs.items():
            if len(dofs) == 0:
                continue
            # Get by key variable.name, if not found, initialize it with an empty dict.
            result.setdefault(operator.name, {})[operator.domain] = dofs
        return result

    def filter_by_tags(
        self, tags: Sequence[pp.solvers.EquationTag | pp.solvers.VariableTag]
    ) -> tuple[list[EquationOrVariableType], list[EquationOrVariableType]]:
        tags_by_name: dict[
            str, list[pp.solvers.EquationTag | pp.solvers.VariableTag]
        ] = defaultdict(list)
        for tag in tags:
            tags_by_name[tag.name].append(tag)

        filtered_operators: list[EquationOrVariableType] = []
        not_filtered_operators: list[EquationOrVariableType] = []

        for operator in self.operators_to_dofs:
            for tag in tags_by_name[operator.name]:
                if tag.defined_on.filter(operator.domain):
                    filtered_operators.append(operator)
                else:
                    not_filtered_operators.append(operator)

        return filtered_operators, not_filtered_operators


class EquationSystemIndexer(Indexer[EquationOnDomain]):
    """Equation indexer for the block arrangement used by :class:`EquationSystem`.

    The AD framework evaluates each equation separately. Before these per-equation
    results are concatenated into a global matrix and residual vector,
    :class:`EquationSystem` may need to select rows belonging to requested domains.
    :attr:`equation_image_space_composition` provides the per-equation indices needed
    for this selection.

    In the global algebraic system, equations form consecutive blocks. Within each
    equation, the image-space composition maps every requested domain to indices in
    that equation's unreduced result. These local indices need not be consecutive: A
    restricted system can select only part of an equation's result.

    """

    def __init__(
        self,
        equation_image_space_composition: dict[str, dict[pp.GridLike, np.ndarray]],
    ) -> None:
        equation_dofs: dict[EquationOnDomain, np.ndarray] = {}
        global_offset = 0
        for eq_name, dofs_on_domains in equation_image_space_composition.items():
            offset_within_equation = 0
            for domain, dofs in dofs_on_domains.items():
                equation_dofs[EquationOnDomain(name=eq_name, domain=domain)] = (
                    np.arange(dofs.size) + (global_offset + offset_within_equation)
                )
                offset_within_equation += dofs.size
            global_offset += offset_within_equation

        super().__init__(operators_to_dofs=equation_dofs)
        self.equation_image_space_composition: Final[
            dict[str, dict[pp.GridLike, np.ndarray]]
        ] = equation_image_space_composition
        """A mapping `equation_name` -> `domains` -> `dofs`.

        The DoFs stored here refer to rows in each equation's separate AD result. The
        consecutive indices of the selected rows after global concatenation can be
        found in :attr:`equation_dofs`. The equation-local indices allow
        :class:`EquationSystem` to select rows before concatenating the per-equation
        results into the global matrix and residual vector.

        Callers must not mutate the dictionary or its arrays.

        Note: It does not include equations with empty domains.

        """


EquationIndexer = Indexer[EquationOnDomain]
VariableIndexer = Indexer["pp.ad.Variable"]
