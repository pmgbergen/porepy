"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class InterfaceDisplacementArray:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any
    nd: Any
    numerical: Any

    def interface_displacement(self, interfaces: ...) -> None: ...
    def interface_displacement_parameter_values(self, interface: ...) -> None: ...
    def update_time_dependent_ad_arrays(self) -> None: ...
    def update_interface_displacement_parameter(self) -> None: ...
