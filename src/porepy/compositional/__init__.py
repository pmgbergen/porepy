"""The compositional subpackage provides utilities to model multi-phase multi-component
fluid mixtures, and fluid phase equilibrium problems.

The entry point to compositional modelling is the module
:mod:`porepy.compositional.base`, wich provides means to model a component-context, a
phase-context and a fluid mixture. Classes for storing the state of a fluid (values of
primary and secondary variables and their derivatives) can be found in
:mod:`porepy.compositional.states`, while functionality for coupling with the models is
provided in :mod:`porepy.compositional.compositional_mixins`.

While the package is in principal self-contained, it provides two interfaces to PorePy's
modelling framework in the form of model mixins:

1. :mod:`porepy.compositional.compositional_mixins`
2. :mod:`porepy.compositional.equilibrium_mixins`

.. rubric:: Some additional information.

    1. The package is built to support the unified formulation of the equilibrium
       problem [1,2].
    2. While thermodynamically consistent, it does not provide full support for any kind
       of thermodynamic computations. It focues on properties required for
       flow & transport.
    3. For the case of more sophisticated thermodynamics, the groundwork is layed by
       defining a thermodynamic reference state (:mod:`~porepy.compositional._core`)
       most commonly used in other packages and literature [3].
    4. Units are standard SI units, and there is in principal no distinguishing between
       massic or molar quantities. Once the modeller decides what the model represents,
       massic or molar values must be consistently enforced.
       The only exception is :data:`~porepy.compositional._core.R_IDEAL_MOL`, which is
       given as a molar quantity.

References:
    [1]: `Lauser et al. (2011) <https://doi.org/10.1016/j.advwatres.2011.04.021>`_
    [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [3]: `IAPWS <http://iapws.org/relguide/IF97-Rev.html>`_

"""

__all__ = []

from . import (  # peng_robinson,
    _core,
    base,
    chem_species,
    compositional_mixins,
    eos_compiler,
    flash,
    states,
    unified_equilibrium_mixins,
    uniflash_c,
    uniflash_utils_c,
    utils,
)
from ._core import *
from .base import *
from .chem_species import *
from .compositional_mixins import *
from .eos_compiler import *
from .flash import *
from .states import *
from .unified_equilibrium_mixins import *
from .uniflash_c import *
from .uniflash_utils_c import *
from .utils import *

__all__.extend(_core.__all__)
__all__.extend(chem_species.__all__)
__all__.extend(base.__all__)
__all__.extend(utils.__all__)
__all__.extend(compositional_mixins.__all__)
__all__.extend(unified_equilibrium_mixins.__all__)
__all__.extend(flash.__all__)
__all__.extend(uniflash_c.__all__)
__all__.extend(states.__all__)
__all__.extend(eos_compiler.__all__)
