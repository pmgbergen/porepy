"""The ``composite`` sub-package in PorePy contains classes representing
mixtures and mixture components, as well as an implementations of the unified flash
procedure.

The unified flash is largely based on the work listed below.

Compositions/mixtures are intended to be part of a flow model, i.e. they use PorePy's AD
framework to represent variables and equations.
The subsystems can naturally be extended by the respective flow model.

The composite module works (for now) with the following units as base units:

- Pressure:     [Pa] (Pascal)
- Temperature:  [K] (Kelvin)
- Mass:         [mol] (mol)
- Energy:       [J] (Joule)
- Volume:       [m^3] (Cubic Meter)

For the reference state, an ideal tri-atomic gas (like water),
with internal energy at the triple point of water set to zero, was chosen (as per
IAPWS standard).

All phases, components and thermodynamic properties are to be modelled with
respect to this reference state.

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
    composite_mixins,
    composite_utils,
    eos_compiler,
    equilibrium_mixins,
    flash,
    flash_c,
    states,
    utils_c,
)
from ._core import *
from .base import *
from .chem_species import *
from .composite_mixins import *
from .composite_utils import *
from .eos_compiler import *
from .equilibrium_mixins import *
from .flash import *
from .flash_c import *
from .states import *
from .utils_c import *

__all__.extend(_core.__all__)
__all__.extend(chem_species.__all__)
__all__.extend(base.__all__)
__all__.extend(composite_utils.__all__)
__all__.extend(composite_mixins.__all__)
__all__.extend(equilibrium_mixins.__all__)
__all__.extend(flash.__all__)
__all__.extend(flash_c.__all__)
__all__.extend(states.__all__)
__all__.extend(eos_compiler.__all__)
__all__.extend(utils_c.__all__)
