"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class DamageHistoryVariable:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any
    nd: Any
    time_manager: Any

    def damage_history(self, subdomains: ...) -> None: ...
    def create_variables(self) -> None: ...
    def update_solution(self, solution: ...) -> None: ...
    def variables_stored_all_time_steps(self) -> None: ...

class DamageHistoryEquation:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any
    nd: Any
    time_manager: Any

    def set_equations(self) -> None: ...
    def before_nonlinear_loop(self) -> None: ...
    def damage_history_equation(self, subdomains: ...) -> None: ...

class IsotropicHistoryEquation:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any
    nd: Any
    time_manager: Any

    def damage_history_equation(self, subdomains: ...) -> None: ...
