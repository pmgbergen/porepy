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

    def projection_to(self, variables: VariableList) -> np.ndarray:
        """TODO YZ
        Create a projection matrix from the global vector of unknowns to a specified
        subspace.

        The transpose of the returned matrix can be used to slice respective columns out
        of the global Jacobian.

        The projection preserves the global order defined by the system, i.e. it
        includes no permutation.

        Parameters:
            variables (optional): VariableType input for which the subspace is
                requested. If no subspace is specified using ``variables``,
                a null-space projection is returned.

        Returns:
            a sparse projection matrix of shape ``(M, num_dofs)``, where
            ``0 <= M <= num_dofs``.

        """
        # TODO YZ: Revisit with indexer
        # current number of total dofs
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


class EquationIndexer:
    """Equation indexer determines the arrangement of DoFs corresponding to multiple
    equations on multiple grids in a contiguous array.

    For a data array with a different arrangement (e.g., produced by taking a subset of
    equations), a new indexer needs to be constructed.

    """

    def __init__(
        self, equation_dofs_per_equation: dict[str, dict[pp.GridLike, np.ndarray]]
    ) -> None:
        equation_dofs: dict[EquationOnDomain, np.ndarray] = {}
        for eq_name, dofs_on_domains in equation_dofs_per_equation.items():
            for domain, dofs in dofs_on_domains.items():
                equation_dofs[EquationOnDomain(name=eq_name, domain=domain)] = dofs

        self.equation_dofs: dict[EquationOnDomain, np.ndarray] = equation_dofs
        """An ordered mapping of atomic equations to their DoF indices. The keys are
        ordered, in a sense that if key A goes before key B, DoFs of key A are located
        before DoFs of key B.

        """
        self.equation_dofs_per_equation: dict[str, dict[pp.GridLike, np.ndarray]] = (
            equation_dofs_per_equation
        )
