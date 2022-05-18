""" Contains concrete substances for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""

from typing import List

import iapws

from ..substance import FluidSubstance

__all__: List[str] = ["H20_iapws"]


class H20_iapws(FluidSubstance):
    pass
