"""Sub-package containing flash functionality for fluid phase equilibria."""

__all__ = []

from . import eos_compiler, flash_initializer, solvers, uniflash, uniflash_equations
from .eos_compiler import *
from .flash_initializer import *
from .solvers import *

# TODO Not clear what is going wrong here. Mypy throws error
# Incompatible import of "<subclass of "PhysicalState" and "int">7"
# (imported name has type "type[porepy.compositional.compiled_flash.uniflash.<subclass
# of "PhysicalState" and "int">7]", local name has type "type[porepy.compositional.
# compiled_flash.flash_initializer.<subclass of "PhysicalState" and "int">7]")
# [assignment]
from .uniflash import *  # type:ignore[assignment]
from .uniflash_equations import *

__all__.extend(solvers.__all__)
__all__.extend(eos_compiler.__all__)
__all__.extend(flash_initializer.__all__)
__all__.extend(uniflash_equations.__all__)
__all__.extend(uniflash.__all__)
