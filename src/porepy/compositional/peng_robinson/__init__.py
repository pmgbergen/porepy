"""Sub-package of ``porepy.compositional`` with implementations using the Peng-Robinson
EoS.

The work here is largely based on below references.

This subpackage implements the standard Peng-Robinson equation of state,
including some mixing rules and model components.

The core of the module are its EoS classes. One is a symbolic representation
:class:`~porepy.compositional.peng_robinson.eos_s.PengRobinsonSymbolic`
and the other is a numba-compiled representation, based on the functions obtained
by the symbolic one
:class:`~porepy.compositional.peng_robinson.eos_c.PengRobinsonCompiler`.

It provides furthermore an interface to load binary interaction parameters from the
package ``thermo``, as well as some mixing rules to obtain a mixture's cohesion and
covolume, such as
:class:`~porepy.compositional.peng_robinson.mixing.VanDerWaals`.

The mixing rules are independent of the EoS and can in theory be used for any other
cubic EoS.

References:
    [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_
    [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [3]: `Driesner (2007) Part I <https://doi.org/10.1016/j.gca.2006.01.033>`_
    [4]: `Driesner (2007) Part II <https://doi.org/10.1016/j.gca.2007.05.026>`_
    [5]: `Soereide (1992) <https://doi.org/10.1016/0378-3812(92)85105-H>`_

"""

__all__ = []

from . import eos_c, eos_s, pr_utils
from .eos_c import *
from .eos_s import *
from .pr_utils import *

__all__.extend(eos_c.__all__)
__all__.extend(eos_s.__all__)
