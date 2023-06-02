"""This is a module containing interface functions for loading chemical and physical
data for chemical species.

Currently supported (Python-) packages include

- chemicals
- thermo

"""
from __future__ import annotations

import chemicals

from .chem_species import FluidSpeciesData

__all__ = ["load_fluid_species"]


def load_fluid_species(
    names: list[str], package: str = "chemicals"
) -> list[FluidSpeciesData]:
    """Creates a fluid species, if identifiable by ``name`` in ``package``

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

    species: list[FluidSpeciesData] = []

    cas: str
    mw: float
    pc: float
    Tc: float
    vc: float
    omega: float

    for name in names:
        if package == "chemicals":

            cas = str(chemicals.CAS_from_any(name))

            # extracting data
            mw = float(chemicals.MW(cas)) * 1e-3  # molas mass in kg / mol
            pc = float(chemicals.Pc(cas))  # critical pressure in Pa
            Tc = float(chemicals.Tc(cas))  # critical temperature in K
            vc = float(chemicals.Vc(cas))  # critical volume in m^3 / mol
            omega = float(chemicals.acentric.omega(cas))  # acentric factor
        else:
            raise NotImplementedError(f"Unsupported package `{package}`.")

        species.append(
            FluidSpeciesData(
                name=name,
                CASr_number=cas,
                molar_mass=mw,
                p_crit=pc,
                T_crit=Tc,
                V_crit=vc,
                omega=omega,
            )
        )

    return species
