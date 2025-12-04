"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class InitialConditionMixin:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any

    def initial_condition(self) -> None: ...
    def set_initial_values_primary_variables(self) -> None: ...
