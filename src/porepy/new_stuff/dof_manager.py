from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import porepy as pp

__all__ = ["DofManager"]


@dataclass
class EquationVariableTag:
    pass


class DofManager:
    def __init__(self, tags: list[EquationVariableTag]):
        self.tags: list[EquationVariableTag] = tags

    def eq_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th equation group."""
        return []

    def var_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th variable group."""
        return []
