"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class EquationMixin:
    # Attributes expected from PorePyModel (via duck typing)
    ad_time_step: Any
    equation_system: Any
    mdg: Any
    reference_variable_values: Any

    def set_equations(self) -> None: ...

class VariableMixin:
    # Attributes expected from PorePyModel (via duck typing)
    ad_time_step: Any
    equation_system: Any
    mdg: Any
    reference_variable_values: Any

    def create_variables(self) -> None: ...
    def perturbation_from_reference(self, name: ..., grids: ...) -> None: ...
