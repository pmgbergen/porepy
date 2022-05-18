""" Contains concrete substances for the solid skeleton of the porous medium.
In the current setting, we expect these substances to only appear in the solid, immobile phase,
"""

from typing import List

from ..substance import SolidSubstance

__all__: List[str] = ["NaCl_simple"]


class NaCl_simple(SolidSubstance):
    pass
