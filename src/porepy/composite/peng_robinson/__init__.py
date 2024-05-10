"""Sub-package of ``porepy.composite`` with implementations using the Peng-Robinson EoS.

The work here is largely based on below references.

This subpackage implements the standard Peng-Robinson equation of state,
including some mixing rules and model components.

It provides an intermediate abstraction for model components
:class:`~porepy.composite.peng_robinson.pr_components.Component_PR`,
such that custom implementations for some physical and chemical quantities can be done.

The core of the module is its EoS class
:class:`~porepy.composite.peng_robinson.eos.PengRobinsonEoS`,
which implements the calculation of physical properties.

It provides furthermore an interface to load binary interaction parameters from the
package ``thermo``, as well as some mixing rules to obtain a mixture's cohesion and
covolume, such as
:class:`~porepy.composite.peng_robinson.mixing.VanDerWaals`.

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

from . import eos_c, eos_s, pr_bip, pr_components
from .eos_c import *
from .eos_s import *
from .pr_bip import *
from .pr_components import *

__all__.extend(pr_bip.__all__)
__all__.extend(pr_components.__all__)
__all__.extend(eos_c.__all__)
__all__.extend(eos_s.__all__)
