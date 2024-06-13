""">>> import porepy.compositional as ppc

The compositional subpackage provides utilities to model multi-phase multi-component
fluid mixtures, and fluid phase equilibrium problems.

The entry point to compositional modelling is the module
:mod:`porepy.compositional.base`, wich provides means to model a component-context, a
phase-context and a fluid mixture.

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
    states,
    utils,
)

# TODO flake8 complains about from . import * imports
# Even though __all__ is defined in all packages
# But it does not complain so for the AD subpackage??
from ._core import P_REF, R_IDEAL_MOL, T_REF
from .base import AbstractEoS, Component, Compound, FluidMixture, Phase
from .chem_species import ChemicalSpecies, load_species
from .compositional_mixins import CompositionalVariables, FluidMixtureMixin
from .states import (
    ExtensiveState,
    FluidState,
    IntensiveState,
    PhaseState,
    initialize_fluid_state,
)
from .utils import (
    CompositionalModellingError,
    compute_saturations,
    extend_fractional_derivatives,
    normalize_rows,
    safe_sum,
)

__all__.extend(_core.__all__)
__all__.extend(chem_species.__all__)
__all__.extend(base.__all__)
__all__.extend(utils.__all__)
__all__.extend(compositional_mixins.__all__)
__all__.extend(states.__all__)
