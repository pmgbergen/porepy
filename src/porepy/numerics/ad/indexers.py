"""Indexers for systems of discretized systems of equations and variables, the values of
which are arranged in a contiguous array.

Used by `EquationSystem` and nonlinear solvers.

"""

from __future__ import annotations
from dataclasses import dataclass

import porepy as pp

import numpy as np

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
    """Variable indexer determines the arrangement of DoFs corresponding to multiple
    variables on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    variables), a new indexer needs to be constructed.

    """

    def __init__(self, variable_dofs: dict[pp.ad.Variable, np.ndarray]) -> None:
        self.variable_dofs: dict[pp.ad.Variable, np.ndarray] = variable_dofs
        """An ordered mapping of atomic variables to their DoF indices. The keys are
        ordered, in a sense that if key A goes before key B, DoFs of key A are located
        before DoFs of key B.

        """
        self.num_dofs: int = sum(x.size for x in self.variable_dofs.values())

    def projection_indices(self, variables: list[pp.ad.Variable]) -> np.ndarray:
        """Create a projection index array from the vector, system vector, represented
        by this indexer, to the requested subspace.

        The projection preserves the order defined by the this indexer and neglects the
        order of the input data. In other words, it includes no permutation.

        Parameters:
            variables (optional): Input for which the subspace is requested.

        Returns:
            an index array of `shape=(M,)`, where `0 <= M <= num_dofs`.

        """
        variables_lookup = set(variables)
        projections = []
        for variable in self.variable_dofs:
            if variable not in variables_lookup:
                continue
            projections.append(self.variable_dofs[variable])
        return (
            np.concatenate(projections)
            if len(projections) > 0
            else np.empty(0, dtype=int)
        )

    def construct_restricted_indexer(self, variables: list[pp.ad.Variable]):
        """Constructs a new indexer based on requested subset of variables.

        The order of the new indexer is defined by the input.

        Raises:
            ValueError: If the requested variable is not known to this indexer.

        """
        new_variable_dofs: dict[pp.ad.Variable, np.ndarray] = {}
        offset = 0
        for variable in variables:
            dofs = self.variable_dofs.get(variable, None)
            if dofs is None:
                raise ValueError(
                    f"Requested variable is not known to this indexer: {variable}"
                )
            new_variable_dofs[variable] = np.arange(dofs.size) + offset
            offset += dofs.size
        return VariableIndexer(variable_dofs=new_variable_dofs)


class EquationIndexer:
    """Equation indexer determines the arrangement of DoFs corresponding to multiple
    equations on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    equations), a new indexer needs to be constructed.

    Implementation note: There is an unfortunate assymetry between this and
    :class:`VariableIndexer` in the way these classes are initialized and what fields
    they expose. It is done so for convenient operations in `EquationSystem`. The
    preferred way to interact with this class outside `EquationSystem` is by using
    :class:`EquationOnDomain` to query DoFs and not
    `dict[str, dict[pp.GridLike, np.ndarray]]` See also the docstring of
    :attr:`equation_image_space_composition`.

    """

    def __init__(
        self, equation_image_composition: dict[str, dict[pp.GridLike, np.ndarray]]
    ) -> None:
        equation_dofs: dict[EquationOnDomain, np.ndarray] = {}
        global_offset = 0
        for eq_name, dofs_on_domains in equation_image_composition.items():
            offset_per_equation = 0
            for domain, dofs in dofs_on_domains.items():
                equation_dofs[EquationOnDomain(name=eq_name, domain=domain)] = (
                    dofs + global_offset
                )
                offset_per_equation += dofs.size
            global_offset += offset_per_equation

        self.equation_dofs: dict[EquationOnDomain, np.ndarray] = equation_dofs
        """An ordered mapping of atomic equations to their DoF indices. The keys are
        ordered, in a sense that if key A goes before key B, DoFs of key A are located
        before DoFs of key B.

        """
        self.equation_image_space_composition: dict[
            str, dict[pp.GridLike, np.ndarray]
        ] = equation_image_composition
        """A mapping `equation_name` -> `domains` -> `dofs`.
        
        The dofs stored here start from 0 for each equation, i.e. the offset is only
        relative to domains. The DoFs with the "global" offset can be found in
        :attr:`equation_dofs`. This is done to respect the implementation of
        `EquationSystem`, where both types of offsets are needed. 

        """
