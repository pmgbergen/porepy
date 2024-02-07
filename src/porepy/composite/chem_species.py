"""This module contains a dataclass structure for chemical species.

The data, defined here as attributes of a chemical species, are the minimal necessary
amount of data for a species to be compatible with the composite submodule.

Use the dataclass contained here for various interfaces with chemical databases
or other, third-party software.

"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChemicalSpecies", "FluidSpecies"]


@dataclass(frozen=True)
class ChemicalSpecies:
    """A basic data class for species, containing identification and basic properties."""

    name: str
    """Name (or chemical formula) of the chemical species."""

    CASr_number: str
    """CAS registry number."""

    molar_mass: float
    """Molar mass in ``[kg / mol]``."""


@dataclass(frozen=True)
class FluidSpecies(ChemicalSpecies):
    """A data class containing required physical properties for fluid species
    such that they can be converted into a
    :class:`~porepy.composite.component.Component`.
    """

    p_crit: float
    """Critical pressure in ``[Pa]``."""

    T_crit: float
    """Critical temperature in ``[K]``."""

    V_crit: float
    """Critical volume in ``[m^3 / mol]``"""

    omega: float
    """Acentric factor."""
