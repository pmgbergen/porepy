"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class SolutionStrategyPhaseProperties:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    fluid: Any
    mdg: Any
    params: Any

    def update_material_properties(self) -> None: ...
    def update_thermodynamic_properties_of_phases(self, state: ...) -> None: ...
    def after_nonlinear_convergence(self) -> None: ...
    def initialize_previous_iterate_and_time_step_values(self) -> None: ...
