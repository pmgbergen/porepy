"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class BoundaryConditionMixin:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any

    def update_all_boundary_conditions(self) -> None: ...
    def update_boundary_values_primary_variables(self) -> None: ...
    def update_boundary_condition(self, name: ..., function: ...) -> None: ...
    def create_boundary_operator(self, name: ..., domains: ...) -> None: ...
    def _combine_boundary_operators(self, subdomains: ..., dirichlet_operator: ..., neumann_operator: ..., robin_operator: ..., bc_type: ..., name: ..., dim: ...) -> None: ...
    def _update_bc_type_filter(self, name: ..., bc_type_callable: ...) -> None: ...
    def __bc_type_storage(self) -> None: ...
