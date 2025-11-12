"""Sub-package of ``porepy.compositional`` with implementations of the Peng-Robinson
EoS.

The work here is largely based on below references.

.. rubric:: Module guide

The base functionality revolves around obtaining the real solutions or real cubic
polynomials, and their derivatives with respect to dependencies such as polynomial
coefficients. The main aspect is an efficient compilation to be used in other modules.
This can be found in :mod:`~porepy.compositional.peng_robinson.cubic_polynomial`.

Based on that, the computation of the compressibility factor is implemented with
dependencies on dimensionless cohesion and covolume. This package supports a
persistent-variable formulation of the phase equilibrium problem, which means it
provides functionalities to compute surrogates for compressibility factors, where a
phase is not physically present.
This is implemented in
:mod:`~porepy.compositional.peng_robinson.compressibility_factor`.

The equation of state itself is implemented using :mod:`numba`-compilation.
Ahead-of-time compilation is mimicked using signatures, i.e. static types, which mich
result in a slow import when importing this subpackage for the first time.
Thermodynamic properties of a fluid are implemented using :mod:`sympy` to provide
lambdified expressions for the properties, tailored to individual fluids.
They then need to be compiled before computations begin.
For more see :mod:`~porepy.compositional.peng_robinson.eos`, and the two classes
therein.

Finally, some extensions are available, including the Soereide extension for brine
mixtures (:mod:`~porepy.compositional.peng_robinson.soereide`), and Lphrenz-Bray-Clark
correlations for viscosity (:mod:`~porepy.compositional.peng_robinson.lbc_viscosity`).
Note however, that the latter is more general and can be used with any other EoS.

References:
    [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_
    [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [3]: `Soereide (1992) <https://doi.org/10.1016/0378-3812(92)85105-H>`_

"""

__all__ = []

from . import (
    compressibility_factor,
    cubic_polynomial,
    eos,
    lbc_viscosity,
    soereide,
    utils,
)
from .compressibility_factor import *
from .cubic_polynomial import *
from .eos import *
from .lbc_viscosity import *
from .soereide import *
from .utils import *

__all__.extend(eos.__all__)
__all__.extend(utils.__all__)
__all__.extend(soereide.__all__)
__all__.extend(lbc_viscosity.__all__)
__all__.extend(cubic_polynomial.__all__)
__all__.extend(compressibility_factor.__all__)
