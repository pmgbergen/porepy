"""This module provides functionalities to obtain binary interaction parameters
between two components for the Peng-Robinson EoS.

This includes interfaces to third-party packages like ``thermo``.

BIPs are required for genuine components (and compounds), which switch phases and appear
in the mixture cohesion term calculated by
:class:`~porepy.compositional.peng_robinson.eos.PengRobinsonEoS`.

This module serves standard, constant BIPs.
For custom implementation of BIPs, see
:attr:`~porepy.compositional.peng_robinson.pr_components.Component_PR.bip_map`.

"""

from __future__ import annotations

from thermo.interaction_parameters import IPDB

__all__ = ["load_bip"]


def load_bip(cas_n1: str, cas_n2: str, package: str = "thermo") -> float:
    """Loads the Peng-Robinson binary interaction parameter for two components from a
    third-party database.

    This function is an interface to third-party databases.

    CAS registry numbers can be accessed using
    :class:`~porepy.compositional.chem_species.ChemicalSpecies.CASr_number`.

    Parameters:
        cas_n1: CAS registry number of first component.
        cas_n2: CAS registry number of second component.
        package: ``default='thermo'``

            Third-party package containing databases from which the BIP are loaded.
            Currently supported packages include:

            - ``'thermo'``

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.

    Returns:
        The constant binary interaction parameter for the Peng-Robinson between two
        components.

    """
    if package == "thermo":
        bip = IPDB.get_ip_automatic(CASs=[cas_n1, cas_n2], ip_type="PR kij", ip="kij")
    else:
        raise NotImplementedError(f"Unsupported package `{package}`.")

    return float(bip)
