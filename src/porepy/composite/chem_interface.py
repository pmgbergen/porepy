"""This is a module containing interface functions for loading chemical and physical
data for chemical species.

Currently supported (Python-) packages include

- chemicals
- thermo

"""
from __future__ import annotations

from typing import Callable, Literal, overload

import chemicals

from .chem_species import ChemicalSpecies, FluidSpecies

__all__ = ["load_species"]


@overload
def load_species(
    names: list[str],
    package: str = "chemicals",
    species_type: Literal["fluid"] = "fluid",
) -> list[FluidSpecies]:
    # overload for default species type fluid
    ...


@overload
def load_species(
    names: list[str],
    package: str = "chemicals",
    species_type: Literal["basic"] = "basic",
) -> list[ChemicalSpecies]:
    # overload for basic species type
    ...


def load_species(
    names: list[str], package: str = "chemicals", species_type: str = "fluid"
) -> list[ChemicalSpecies] | list[FluidSpecies]:
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
        species_type: ``default='fluid'``

            Species type to be loaded. By default, fluid species are created.

            This argument defines which parameters are attempted to be loaded.
            Only parameters relevant for the species type are loaded.

            - ``'fluid'``: Returns :class:`~~porepy.composite.chem_species.FluidSpecies`
            - ``'basic'``: :class:`~~porepy.composite.chem_species.ChemicalSpecies`

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.

    Returns:
        If the look-up was successful, extracts the relevant data and returns respective
        data structure.

    """

    species: list[FluidSpecies] = []

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
        # default species
        if species_type == "fluid":
            species.append(
                FluidSpecies(
                    name=name,
                    CASr_number=cas,
                    molar_mass=float(mw_loader(cas)),
                    p_crit=float(pc_loader(cas)),
                    T_crit=float(Tc_loader(cas)),
                    V_crit=float(vc_loader(cas)),
                    omega=float(omega_loader(cas)),
                )
            )
        # Basic species
        elif species_type == "basic":
            species.append(
                ChemicalSpecies(
                    name=name,
                    CASr_number=cas,
                    molar_mass=float(mw_loader(cas)),
                )
            )
        else:
            raise NotImplementedError(f"Unsupported species type `{species_type}`.")

    return species
