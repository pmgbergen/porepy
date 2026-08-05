"""Indexers for discretized systems of equations and variables, the values of which are
arranged in a contiguous array.

Used by `EquationSystem` and nonlinear solvers.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import porepy as pp

__all__ = ["EquationOnDomain", "VariableIndexer", "EquationIndexer"]


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
            raise ValueError(f"Requested variables are duplicated: {variable}.")

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
    """An equation indexer determines the arrangement of indices corresponding to multiple
    equations on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    equations), a new indexer needs to be constructed.

    Implementation note: There is an unfortunate asymmetry between this and
    :class:`VariableIndexer` in the way these classes are initialized and what fields
    they expose. This is done to support convenient operations in `EquationSystem`. The
    preferred way to interact with this class outside `EquationSystem` is by using
    :class:`EquationOnDomain` to query indices and not
    `dict[str, dict[pp.GridLike, np.ndarray]]`. See also the docstring of
    :attr:`equation_image_space_composition`.

    """

    def __init__(
        self, equation_image_composition: dict[str, dict[pp.GridLike, np.ndarray]]
    ) -> None:
        equation_indices: dict[EquationOnDomain, np.ndarray] = {}
        global_offset = 0
        for eq_name, indices_on_domains in equation_image_composition.items():
            offset_per_equation = 0
            for domain, indices in indices_on_domains.items():
                equation_indices[EquationOnDomain(name=eq_name, domain=domain)] = (
                    indices + global_offset
                )
                offset_per_equation += indices.size
            global_offset += offset_per_equation

        self.indices: dict[EquationOnDomain, np.ndarray] = equation_indices
        """An ordered mapping of atomic equations to their DoF indices. The keys are
        ordered, in the sense that if key A goes before key B, indices of key A are located
        before indices of key B.

        """
        self.equation_image_space_composition: dict[
            str, dict[pp.GridLike, np.ndarray]
        ] = equation_image_composition
        """A mapping `equation_name` -> `domains` -> `indices`.

        The indices stored here start from 0 for each equation, i.e. the offset is only
        relative to domains. The indices with the "global" offset can be found in
        :attr:`indices`. This is done to respect the implementation of
        `EquationSystem`, where both types of offsets are needed.

        Note: It does not include equations with empty domains.

        """

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

        # YZ: This is a little hack to be removed in the next PR. Now for simplicity,
        # EquationIndexer needs to be initialized with (equation-local)
        # equation_image_composition, so we reconstruct it here.
        equation_image_composition: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        offsets_per_equation: dict[str, int] = {}
        for equation, indices in new_equation_indices.items():
            equation_offset = offsets_per_equation.get(equation.name, 0)
            equation_image_composition.setdefault(equation.name, {})[
                equation.domain
            ] = np.arange(indices.size) + equation_offset
            offsets_per_equation[equation.name] = equation_offset + indices.size

        return EquationIndexer(equation_image_composition=equation_image_composition)
