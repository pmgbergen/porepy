"""The ``composite`` sub=package in PorePy contains classes representing various
mixtures and implementations of the unified flash procedure in p-T and p-h.

The unified flash is largely based on the work listed below.

Compositions/mixtures are intended to be part of a flow model, i.e. they use PorePy's AD
framework to represent variables and equations.
The p-h and p-T subsystems can naturally be extended by the respective flow model.

The composite module works (for now) with the following units as base units:

- Pressure:     [MPa] (Mega Pascal)
- Temperature:  [K] (Kelvin)
- Mass:         [mol] (mol)
- Energy:       [kJ] (Kilo Joule)
- Volume:       [m^3] (Cubic Meter)

For the reference state, an ideal tri-atomic gas (like water),
with internal energy at the triple point of water set to zero, was chosen.

All modelled phases, components and thermodynamic properties are to be modelled with
respect to this reference state.

References:
    [1]: `Lauser et al. (2011) <https://doi.org/10.1016/j.advwatres.2011.04.021>`_
    [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [3]: `IAPWS <http://iapws.org/relguide/IF97-Rev.html>`_

"""

__all__ = []

from . import _composite_utils, composition, flash, peng_robinson, simple_composition
from ._composite_utils import *
from .composition import *
from .flash import *
from .peng_robinson import *
from .simple_composition import *

__all__.extend(_composite_utils.__all__)
__all__.extend(composition.__all__)
__all__.extend(flash.__all__)
__all__.extend(peng_robinson.__all__)
__all__.extend(simple_composition.__all__)
