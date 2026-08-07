"""Indexers for discretized systems of equations and variables, the values of which are
arranged in a contiguous array.

Used by `EquationSystem` and nonlinear solvers.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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


class VariableIndexer:
    """A variable indexer determines the arrangement of indices corresponding to
    multiple variables on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    variables), a new indexer needs to be constructed.

    Parameters:
        indices: A mapping of atomic variables to their indices. It should be ordered
            in a sense that if key A goes before key B, indices of key A are located
            before indices of key B. Ordering is not validated, so passing incorrect
            ordering may lead to errors.

    """

    def __init__(self, indices: dict[pp.ad.Variable, np.ndarray]) -> None:
        self.indices: dict[pp.ad.Variable, np.ndarray] = indices
        """An ordered mapping of atomic variables to their indices. The keys are
        ordered, in the sense that if key A goes before key B, indices of key A are
        located before indices of key B. 

        """
        # TODO YZ: Is it meaningful to enable iterations over this attribute directly
        # through the class? (addressed in the downstream PR).
        self.size: int = sum(x.size for x in self.indices.values())

    def projection_indices(self, variables: list[pp.ad.Variable]) -> np.ndarray:
        """Create a projection index array from the system vector represented by this
        indexer to the requested subspace.

        The order of the variables in the projection is defined by the input.

        Parameters:
            variables: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested variable is not known to this indexer.

        Returns:
            an index array of `shape=(M,)`, where `0 <= M <= size`.

        """
        projections = []
        for variable in variables:
            indices = self.indices.get(variable, None)
            if indices is None:
                raise ValueError(
                    f"Requested variable is not known to this indexer: {variable}."
                )
            projections.append(indices)
        return (
            np.concatenate(projections)
            if len(projections) > 0
            else np.empty(0, dtype=int)
        )

    def construct_restricted_indexer(
        self, variables: list[pp.ad.Variable]
    ) -> VariableIndexer:
        """Constructs a new indexer based on requested subset of variables.

        The order of the new indexer is defined by the input.

        Parameters:
            variables: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested variable is not known to this indexer.

        Returns:
            A new instance of VariableIndexer.

        """
        new_variable_indices: dict[pp.ad.Variable, np.ndarray] = {}
        offset = 0
        for variable in variables:
            indices = self.indices.get(variable, None)
            if indices is None:
                raise ValueError(
                    f"Requested variable is not known to this indexer: {variable}."
                )
            new_variable_indices[variable] = np.arange(indices.size) + offset
            offset += indices.size

        if len(new_variable_indices) != len(variables):
            raise ValueError(f"Requested variables are duplicated: {variables}.")

        return VariableIndexer(indices=new_variable_indices)

    def group_by_name(self) -> dict[str, dict[pp.GridLike, np.ndarray]]:
        """Group :attr:`indices` by variable names.

        Domains with no indices are ignored.

        Return:
            A nested mapping "variable_name" -> "domain" -> "indices".

        """
        variables: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        for variable, indices in self.indices.items():
            if len(indices) == 0:
                continue
            # Get by key variable.name, if not found, initialize it with an empty dict.
            # Then populate the dict with the domain and indices.
            variables.setdefault(variable.name, {})[variable.domain] = indices
        return variables

    def identify_dof(self, dof: int) -> pp.ad.Variable:
        """Identifies the variable to which a specific DOF index belongs.

        The intended use is to help identify entries in the global vector or the column
        of the Jacobian.

        This operation is O(n) for n elements in the vector in the worst case. This
        method should not be used in a hot loop.

        Parameters:
            dof: a single index in the global vector.

        Returns:
            The identified Variable object.

        Raises:
            KeyError: if the dof is out of range.

        """
        for variable, indices in self.indices.items():
            if dof in indices:
                return variable

        raise KeyError("Dof index out of range.")


class EquationIndexer:
    """Map atomic equations to their DoF indices in an algebraic system.

    The DoFs may have an arbitrary arrangement. In particular, the indices belonging to
    an equation need not be contiguous, and equations need not occur in registration
    order. A new indexer must be constructed for a data array with a different
    arrangement.

    """

    def __init__(self, indices: dict[EquationOnDomain, np.ndarray]) -> None:
        self.indices: Final[dict[EquationOnDomain, np.ndarray]] = indices
        """Mapping of atomic equations to their DoF indices."""
        self.size: int = sum(x.size for x in self.indices.values())

    def group_by_name(self) -> dict[str, dict[pp.GridLike, np.ndarray]]:
        """Group :attr:`indices` by equation names.

        Domains with no indices are ignored.

        Offset between equations is assumed.

        Note: This is not equivalent to :attr:`equation_image_space_composition`,
            because the latter does not include offset between equation.

        Return:
            A nested mapping "equation_name" -> "domain" -> "indices".

        """
        equations: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        for equation, indices in self.indices.items():
            if len(indices) == 0:
                continue
            # Get by key equation.name, if not found, initialize it with an empty dict.
            equations.setdefault(equation.name, {})[equation.domain] = indices
        return equations

    def construct_restricted_indexer(
        self, equations: list[EquationOnDomain]
    ) -> EquationIndexer:
        """Constructs a new indexer based on requested subset of equations.

        The order of the new indexer is defined by the input.

        Parameters:
            equations: Input for which the subspace is requested.

        Raises:
            ValueError: If the requested equation is not known to this indexer.

        Returns:
            A new instance of EquationIndexer.

        """
        new_equation_indices: dict[pp.ad.EquationOnDomain, np.ndarray] = {}
        offset = 0
        for equation in equations:
            indices = self.indices.get(equation, None)
            if indices is None:
                raise ValueError(
                    f"Requested equation is not known to this indexer: {equation}."
                )
            new_equation_indices[equation] = np.arange(indices.size) + offset
            offset += indices.size

        if len(new_equation_indices) != len(equations):
            raise ValueError(f"Requested equations are duplicated: {equations}.")

        return EquationIndexer(indices=new_equation_indices)

    def projection_indices(self, equations: list[EquationOnDomain]) -> np.ndarray:
        """Create a projection index array from the system vector represented
        by this indexer to the requested subspace.

        The order of the equations in the projection is defined by the input.

        Parameters:
            equation: Input for which the subspace is requested.

        Returns:
            an index array of `shape=(M,)`, where `0 <= M <= num_dofs`.

        """
        projections = []
        for equation in equations:
            dofs = self.indices.get(equation, None)
            if dofs is None:
                raise ValueError(
                    f"Requested equation is not known to this indexer: {equation}."
                )
            projections.append(dofs)
        return (
            np.concatenate(projections)
            if len(projections) > 0
            else np.empty(0, dtype=int)
        )


class EquationSystemIndexer(EquationIndexer):
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
