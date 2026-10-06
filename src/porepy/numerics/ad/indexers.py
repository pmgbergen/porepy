"""Indexers for discretized systems of equations and variables, the values of which are
arranged in a contiguous array.

Used by `EquationSystem` and nonlinear solvers.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

import porepy as pp
from porepy.numerics.ad.operators import Variable

__all__ = [
    "EquationOnDomain",
    "VariableIndexer",
    "Indexer",
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

    def __str__(self) -> str:
        if isinstance(self.domain, pp.Domain):
            return (
                "Equation {self.name} on subdomain(id={self.domain.id}, "
                f"dim={self.domain.dim})"
            )
        elif isinstance(self.domain, pp.MortarGrid):
            return (
                f"Equation {self.name} on interface(id={self.domain.id}, "
                f"dim={self.domain.dim}, codim={self.domain.codim})"
            )
        else:
            return f"Equation {self.name} on unknown domain"


class Indexer[EquationOrVariableType: (EquationOnDomain, Variable)]:
    """An indexer determines the arrangement of indices corresponding to multiple
    operators (variables or equations) on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    operators), a new indexer needs to be constructed.

    Indexer is defined as a generic class with a parameter type `EquationOrVariableType`
    that can be either `EquationOnDomain` or `pp.ad.Variable`. It helps with typing a
    lot: the indexer that stores variables will always operate on and return variables,
    and not equations, and mypy is aware of it.

    Internally, it relies on `EquationOrVariableType` having the fields `name` and
    `domain`.

    Parameters:
        indices: A mapping of atomic operators to their indices. It should be ordered
            in a sense that if key A goes before key B, indices of key A are located
            before indices of key B. Ordering is not validated, so passing incorrect
            ordering may lead to errors.

    """

    def __init__(self, indices: dict[EquationOrVariableType, np.ndarray]) -> None:
        self.indices: dict[EquationOrVariableType, np.ndarray] = indices
        """An ordered mapping of atomic operators to their indices. The keys are
        ordered, in a sense that if key A goes before key B, indices of key A are
        located before indices of key B.

        """
        self.size: int = sum(x.size for x in self.indices.values())
        """Number of indices in a vector of multiple operators that corresponds to this
        indexer.

        """

    def projection_indices(self, operators: list[EquationOrVariableType]) -> np.ndarray:
        """Create a projection index array from the system vector represented
        by this indexer to the requested subspace.

        The order of the operators in the projection is defined by the input.

        Parameters:
            operators: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested operator is not known to this indexer.

        Returns:
            an index array of `shape=(M,)`, where `0 <= M <= size`.

        """
        projections = []
        for operator in operators:
            indices = self.indices.get(operator, None)
            if indices is None:
                raise ValueError(
                    f"Requested operator is not known to this indexer: {operator}."
                )
            projections.append(indices)
        return (
            np.concatenate(projections, dtype=int)
            if len(projections) > 0
            else np.empty(0, dtype=int)
        )

    def construct_restricted_indexer(
        self, operators: list[EquationOrVariableType]
    ) -> Indexer[EquationOrVariableType]:
        """Constructs a new indexer based on requested subset of operators.

        The order of the new indexer is defined by the input.

        Parameters:
            operators: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested operator is not known to this indexer.

        Returns:
            A new instance of Indexer.

        """
        new_indices: dict[EquationOrVariableType, np.ndarray] = {}
        offset = 0
        for operator in operators:
            indices = self.indices.get(operator, None)
            if indices is None:
                raise ValueError(
                    f"Requested operator is not known to this indexer: {operator}."
                )
            new_indices[operator] = np.arange(indices.size) + offset
            offset += indices.size

        if len(new_indices) != len(operators):
            raise ValueError(f"Requested operators are duplicated: {operators}.")

        return Indexer(indices=new_indices)

    def construct_restricted_indexer_from_tags(
        self,
        tags: Sequence[pp.solvers.OperatorTag[EquationOrVariableType]],
        model: pp.PorePyModel,
    ) -> Indexer[EquationOrVariableType]:
        """Constructs a new indexer based on requested subset of operators, defined by
        tags.

        Parameters:
            tags: Define the requested subspace.
            model: Used to apply tag filters.

        Raises:
            ValueError: If a tag requests an operator not known to this indexer.

        Returns:
            A new instance of Indexer.

        """
        restricted, _ = self.filter_by_tags(tags=tags, model=model)
        return self.construct_restricted_indexer(restricted)

    def group_by_name(self) -> dict[str, dict[pp.GridLike, np.ndarray]]:
        """Group :attr:`indices` by operator names.

        Domains with no indices are ignored.

        Return:
            A nested mapping "operator_name" -> "domain" -> "indices".

        """
        result: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        for operator, indices in self.indices.items():
            if len(indices) == 0:
                continue
            # Get by key variable.name, if not found, initialize it with an empty dict.
            # Then populate the dict with the domain and indices.
            result.setdefault(operator.name, {})[operator.domain] = indices
        return result

    def filter_by_tags(
        self,
        tags: Sequence[pp.solvers.OperatorTag[EquationOrVariableType]],
        model: pp.PorePyModel,
    ) -> tuple[list[EquationOrVariableType], list[EquationOrVariableType]]:
        """Split operators into those selected by ``tags`` and their complement.

        Multiple tags may have the same name when their domain filters are disjoint.
        A ``ValueError`` is raised if multiple tags select the same operator.

        Parameters:
            tags: A list of tags to use for filtering.
            model: The PorePy model. Needed for some domain filters.

        Returns:
            A tuple of two elements:
            - Equations / variables that are within the list of requested tags.
            - Equations / variables that are NOT within the list of requested tags
                (its complement).

        """
        tags_by_name: dict[
            str, list[pp.solvers.OperatorTag[EquationOrVariableType]]
        ] = defaultdict(list)
        for tag in tags:
            tags_by_name[tag.name].append(tag)

        selected: list[EquationOrVariableType] = []
        not_selected: list[EquationOrVariableType] = []
        for operator in self.indices:
            matching_tags = [
                tag
                for tag in tags_by_name[operator.name]
                if tag.defined_on.filter(domain=operator.domain, model=model)
            ]
            if len(matching_tags) > 1:
                raise ValueError(f"Duplicated operators: [{operator}]")
            if len(matching_tags) == 1:
                selected.append(operator)
            else:
                not_selected.append(operator)

        return selected, not_selected

    def identify_dof(self, index: int) -> EquationOrVariableType:
        """Identifies the variable/equation to which a specific index belongs.

        The intended use is to help identify entries in the row/column of the Jacobian.

        This operation is O(n) for n elements in the vector in the worst case. This
        method should not be used in a hot loop.

        Parameters:
            index: a single index in the vector corresponding to this indexer.

        Returns:
            The identified `Variable` or `EquationOnDomain` object.

        Raises:
            KeyError: if the dof is out of range.

        """
        for operator, indices in self.indices.items():
            if index in indices:
                return operator

        raise KeyError("Dof index out of range.")


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
        equation_indices: dict[EquationOnDomain, np.ndarray] = {}
        global_offset = 0
        for eq_name, dofs_on_domains in equation_image_space_composition.items():
            offset_within_equation = 0
            for domain, dofs in dofs_on_domains.items():
                equation_indices[EquationOnDomain(name=eq_name, domain=domain)] = (
                    np.arange(dofs.size) + (global_offset + offset_within_equation)
                )
                offset_within_equation += dofs.size
            global_offset += offset_within_equation

        super().__init__(indices=equation_indices)
        self.equation_image_space_composition: Final[
            dict[str, dict[pp.GridLike, np.ndarray]]
        ] = equation_image_space_composition
        """A mapping `equation_name` -> `domains` -> `dofs`.

        The DoFs stored here refer to rows in each equation's separate AD result. The
        consecutive indices of the selected rows after global concatenation can be
        found in :attr:`indices`. The equation-local indices allow
        :class:`EquationSystem` to select rows before concatenating the per-equation
        results into the global matrix and residual vector.

        Callers must not mutate the dictionary or its arrays.

        Note: It does not include equations with empty domains.

        """


# Specified parametrization.
EquationIndexer = Indexer[EquationOnDomain]
VariableIndexer = Indexer[Variable]
