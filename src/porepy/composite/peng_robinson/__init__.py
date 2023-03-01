"""Sub-package of ``porepy.composite`` with implementations using the Peng-Robinson EoS.

The work here is largely based on below references.

References:
    [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_
    [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [3]: `Driesner (2007) Part I <https://doi.org/10.1016/j.gca.2006.01.033>`_
    [4]: `Driesner (2007) Part II <https://doi.org/10.1016/j.gca.2007.05.026>`_
    [5]: `Soereide (1992) <https://doi.org/10.1016/0378-3812(92)85105-H>`_

"""

__all__ = []

from . import pr_bip, pr_eos, pr_mixture, pr_model_components, pr_utils
from .pr_bip import *
from .pr_eos import *
from .pr_mixture import *
from .pr_model_components import *
from .pr_phase import PR_Phase
from .pr_utils import *

__all__.extend(pr_mixture.__all__)
__all__.extend(pr_bip.__all__)
__all__.extend(pr_eos.__all__)
__all__.extend(pr_model_components.__all__)
__all__.extend(["PR_Phase"])
__all__.extend(pr_utils.__all__)
