"""Sub-package containing flash functionality for fluid phase equilibria."""

__all__ = []

from . import (
    abstract_flash,
    flash_initializer,
    persistent_variable_flash,
    solvers,
    uniflash_equations,
)
from .abstract_flash import *
from .flash_initializer import *

# TODO Not clear what is going wrong here. Mypy throws error
# Incompatible import of "<subclass of "PhysicalState" and "int">7"
# (imported name has type "type[porepy.compositional.compiled_flash.uniflash.<subclass
# of "PhysicalState" and "int">7]", local name has type "type[porepy.compositional.
# compiled_flash.flash_initializer.<subclass of "PhysicalState" and "int">7]")
# [assignment]
from .persistent_variable_flash import *  # type:ignore[assignment]
from .solvers import *
from .uniflash_equations import *

__all__.extend(solvers.__all__)
__all__.extend(abstract_flash.__all__)
__all__.extend(flash_initializer.__all__)
__all__.extend(uniflash_equations.__all__)
__all__.extend(persistent_variable_flash.__all__)
