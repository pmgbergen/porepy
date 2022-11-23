"""This modules contains Binary Interaction Parameters for components modelled for the
Peng-Robinson EoS.

BIPs are intended for genuine components (and compounds), but not for pseudo-components.
The effect of pseudo-components is integrated in respective interaction law involving
compounds.

Respective references for these largely heuristic laws can be found in respective
implementations.

BIPs are implemented as callable objects. This module provides a map ``BIP_MAP`` which maps
between two components and their respective BIP.

The BIP between a component/compound and itself is assumed to be 1, and hence not given here.

"""
from __future__ import annotations

from typing import Callable

__all__ = [
    "BIP_MAP"
]

BIP_MAP: dict[tuple[str, str], Callable] = {
    ("H2O", "CO2"): 1.0,
    ("H2O", "H2S"): 1.0,
    ("CO2", "H2S"): 1.0,
    ("NaClBrine", "H2S"): 1.0,
    ("NaClBrine", "CO2"): 1.0,
}
"""Contains for a pair of component/compound names (key) the respective
binary interaction parameter in form of a callable.

This map serves the Peng-Robinson composition to assemble the attraction parameter of the
mixture and its intended use is only there.

"""