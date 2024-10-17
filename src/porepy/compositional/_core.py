"""This private module contains central assumptions and data for the entire
compositional subpackage.

Changes here should be done with much care.

"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "R_IDEAL_MOL",
    "P_REF",
    "T_REF",
    "COMPOSITIONAL_VARIABLE_SYMBOLS",
    "PhysicalState",
]


NUMBA_CACHE: bool = True
"""Flag to instruct the numba compiler to cache (!and use cached!) functions.

This might cause some confusion in the developing process due to some lack in numba's
caching functionality.
(Does not recognize changes in nested functions and hence does not trigger
re-compilation).

Use with care.

Note:
    Functions which do not use other numba-compiled functions are cached by default.
    This flag is for those who do use other functions.

See Also:
    https://numba.readthedocs.io/en/stable/user/jit.html#cache

"""

NUMBA_FAST_MATH: bool = False
"""Flag to instruct the numba compiler to use it's ``fastmath`` functions.

To be used with care, due to loss in precision.

See Also:
    https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit

"""

NUMBA_PARALLEL: bool = True
"""Flag to instruct numba to compile functions in parallel mode, where applicable.

By default, the parallel backend will be used.

Flag is introduced for developing processes when involving other packages supporing
parallelism such as numpy and PETSc.

Affected numba functionality includes:

1. `JIT parallelism
   <https://numba.readthedocs.io/en/stable/user/jit.html#parallel>`_
2. `Numpy universal functions
   <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_

"""

R_IDEAL_MOL: float = 8.31446261815324
"""Universal gas constant in ``[J / K mol]``."""

P_REF: float = 611.657
"""The reference pressure for the composite module is set to the triple point pressure
of pure water in ``[Pa]``.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

"""

T_REF: float = 273.16
"""The reference temperature for the composite module is set to the triple point
temperature of pure water in ``[K]``.

This value must be used to calculate the reference state when dealing with thermodynamic
properties.

"""

V_REF: float = 1.0
"""The reference volume is set to 1 ``[m^3]``.

Computations in porous media, where densities are usually
expressed as per Reference Element Volume, have to be adapted respectively.

"""

RHO_REF: float = P_REF / (R_IDEAL_MOL * T_REF) / V_REF
"""The reference density is computed using the ideal gas law and :data:`P_REF`,
:data:`T_REF`, :data:`V_REF` and :data:`R_IDEAL`. Its physical dimension is
``[mol / m^3]``"""

U_REF: float = 0.0
"""The reference value for the specific internal energy ``[J / mol]``, at :data:`T_REF`
and :data:`P_REF`. It is set to zero."""

H_REF: float = U_REF + P_REF / RHO_REF
"""The reference value for the specific enthalpy ``[J / mol]``.

It holds :math:`h_r = u_r + \\frac{p_r}[\\rho_r]`

"""

_heat_capacity_ratio: float = 8.0 / 6.0
"""Heat capacity ratio for ideal, triatomic gases like water.
Set to :math:`\\frac{8}{6}`"""

CP_REF: float = _heat_capacity_ratio / (_heat_capacity_ratio - 1) * R_IDEAL_MOL
"""The specific heat capacity at constant pressure for ideal water vapor in
``[J / K mol]``.

It holds :math:`c_p = \\frac{\\gamma}{\\gamma - 1} R_{ideal}`, with
:math:`\\gamma = \\frac{8}{6}`.

See Also:

    https://en.wikipedia.org/wiki/Heat_capacity_ratio

"""

CV_REF: float = 1.0 / (_heat_capacity_ratio - 1) * R_IDEAL_MOL
"""The specific heat capacity at constant volume for ideal water vapor in
``[J / K mol]``.

It holds :math:`c_v = \\frac{1}{\\gamma - 1} R_{ideal}`, with
:math:`\\gamma = \\frac{8}{6}`.

See Also:

    https://en.wikipedia.org/wiki/Heat_capacity_ratio

"""

COMPOSITIONAL_VARIABLE_SYMBOLS = {
    "pressure": "p",
    "enthalpy": "h",
    "temperature": "T",
    "volume": "v",
    "overall_fraction": "z",
    "phase_fraction": "y",
    "phase_saturation": "s",
    "phase_composition": "x",
    "tracer_fraction": "c",
}
"""A dictionary mapping names of variables (key) to their symbol (value), which is used
in the compositional framework.

Important:
    When using the composite framework, it is important to **not** name any other
    variable using the symbols here.

"""


class PhysicalState(Enum):
    """Enum object for characterizing the physical states of a phase.

    - :attr:`liquid`: liquid-like state (value 0)
    - ``gas: int = 1``: gas-like state (value 1)
    - values above 1 are reserved for further development

    """

    liquid: int = 0
    gas: int = 1
