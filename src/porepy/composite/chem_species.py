"""This module contains a dataclass structure for chemical species.

The data, defined here as attributes of a chemical species, are the minimal necessary
amount of data for a species to be compatible with the composite submodule.

Use the dataclass contained here for various interfaces with chemical databases
or other, third-party software.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import chemicals

__all__ = ["load_species", "ChemicalSpecies"]


def load_species(names: list[str], package: str = "chemicals") -> list[ChemicalSpecies]:
    """Creates a species, if identifiable by ``name`` in ``package``

    Important:
        The ``name`` is passed directly to the package. There is no guarantee if the
        returned values are correct, of if the third-party package will work without
        throwing errors.

    Parameters:
        names: A list of names or chemical formulae to look up the chemical species.
        package: ``default='chemicals'``

            Name of one of the supported packages containing chemical databases.

            Currently supported:

            - chemicals

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.

    Returns:
        If the look-up was successful, extracts the relevant data and returns respective
        data structure.

    """

    species: list[ChemicalSpecies] = []

    cas: str

    cas_loader: Callable
    mw_loader: Callable
    pc_loader: Callable
    Tc_loader: Callable
    vc_loader: Callable
    omega_loader: Callable

    if package == "chemicals":
        cas_loader = chemicals.CAS_from_any
        mw_loader = lambda x: chemicals.MW(x) * 1e-3  # molas mass in kg / mol
        pc_loader = chemicals.Pc  # critical pressure in Pa
        Tc_loader = chemicals.Tc  # critical temperature in K
        vc_loader = chemicals.Vc  # critical volume in m^3 / mol
        omega_loader = chemicals.acentric.omega  # acentric factor

    else:
        raise NotImplementedError(f"Unsupported package `{package}`.")

    for name in names:
        cas = str(cas_loader(name))
        species.append(
            ChemicalSpecies(
                name=name,
                CASr_number=cas,
                molar_mass=float(mw_loader(cas)),
                p_crit=float(pc_loader(cas)),
                T_crit=float(Tc_loader(cas)),
                V_crit=float(vc_loader(cas)),
                omega=float(omega_loader(cas)),
            )
        )

    return species


@dataclass(frozen=True)
class ChemicalSpecies:
    """A basic data class for species, containing identification and basic properties."""

    name: str
    """Name (or chemical formula) of the chemical species."""

    CASr_number: str
    """CAS registry number."""

    molar_mass: float
    """Molar mass in ``[kg / mol]``."""

    p_crit: float
    """Critical pressure in ``[Pa]``."""

    T_crit: float
    """Critical temperature in ``[K]``."""

    V_crit: float
    """Critical volume in ``[m^3 / mol]``"""

    omega: float
    """Acentric factor."""
