"""This module contains functionality to set up mixtures using the advanced EoS.

Intended use is for utilities providing interfaces to third-party data or packages.

Example:
    Obtaining BIPs requires usually third-party software which is not necessarily
    included in PorePy's requirements (e.g. :mod:`thermo`).

"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np

from ..base import Component

__all__ = [
    "get_bip_matrix",
]


def get_bip_matrix(
    components: Sequence[Component], package: str = "thermo"
) -> np.ndarray:
    """Loads the Peng-Robinson binary interaction parameters from a
    third-party database.

    Note:
        Requires the package ``chemicals`` to load and identify components by
        CAS registry numbers.

    Parameters:
        components: A list of components with valid CASs registry numbers.
        package: ``default='thermo'``

            Third-party package containing databases from which the BIP are loaded.
            Currently supported packages include:

            - ``'thermo'``

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.

    Returns:
        A symmtric 2D array ``bip_matrix`` containing BIP values.

        The row/column order for BIPs corresponds to the order of ``components``.
        I.e., the BIP between ``components[i]`` and ``components[j]`` is given by
        ``bip_matrix[i, j]``.

        Note that ``bip_matrix[i, i]`` is always zero. Zeros in the upper and lower
        triangle of the matrix are most likely a result of missing data in the used
        package. A warnings is issued if that is the case.

    """
    nc = len(components)
    bip_mat = np.zeros((nc, nc))

    # type-hinting how a package-specific BIP fetching function should look like
    # to obtain the BIP for two components identified with their CASs registry number
    # in string format
    fetcher: Callable[[str, str], float]

    try:
        import chemicals  # type:ignore

        cas_numbers = [chemicals.CAS_from_any(comp.name) for comp in components]
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Require chemicals package to load CAS registry numbers."
        ) from err

    if package == "thermo":
        from thermo.interaction_parameters import IPDB  # type:ignore[import-untyped]

        def fetcher(cas_1: str, cas_2: str) -> float:
            bip = IPDB.get_ip_automatic(CASs=[cas_1, cas_2], ip_type="PR kij", ip="kij")
            return float(bip)

    else:
        raise NotImplementedError(f"Unsupported package `{package}`.")

    for i in range(nc):
        comp_i = components[i]
        cas_i = cas_numbers[i]
        for j in range(i + 1, nc):
            comp_j = components[j]
            cas_j = cas_numbers[j]

            bip_ij = fetcher(cas_i, cas_j)

            if bip_ij == 0.0:
                warnings.warn(
                    f"Fetched BIP ({package}) for components ({comp_i.name}, "
                    + f"{comp_j.name}) is zero. Most likely due to missing data in"
                    + " third-party package."
                )

            bip_mat[i, j] = bip_ij

    return bip_mat + bip_mat.T
