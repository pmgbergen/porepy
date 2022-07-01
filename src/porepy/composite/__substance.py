"""
Containts the abstract base class for all components (species/ pure substances)
used in this framework.
"""
from __future__ import annotations

import abc
from typing import Dict, List

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_variable

__all__ = ["Substance", "FluidSubstance", "SolidSubstance"]


class Substance(abc.ABC):
    """
    Abstract base class for pure substances, providing abstract physical properties
    which need to be implemented for concrete child classes to work in PorePy.

    Provides and manages substance-related AD-variables.

    Turns every Substance child class into a singleton, such that there is only one object
    containing all variables and properties, which can be instantiated in multiple
    model phases.
    This assures that all phases access "the same" substance.

    Instantiated AD variables are provided as properties, as well as the names under which
    they are stored in the grid data dictionary.

    Current relevant variables (per substance instance):
        - overall molar fraction
        - molar fraction in phase for given phase name

    Physical attributes include constants and scalar functions.
    The latter one is dependent on the thermodynamic state (e.g. pressure, enthalpy).

    Note:   The doc strings of the abstract properties and methods contain information
            about intended physical dimensions.
            Keep it consistent when deriving child classes!

    1. Constants
        - Molar Mass

    2. Scalars (THD dependent)
        - molar density
        - Fick diffusivity coefficient
        - thermal conductivity coefficient

    """

    """ For a grid bucket (keys), contains a list of present substances (values). """
    __substances_per_gb: Dict[pp.GridBucket, List[str]] = dict()

    """ For a name (keys), contains the singleton (subtance instance). """
    __susbtance_instances: Dict[str, Substance] = dict()

    """ Flags if a singleton has been re-instantiated in order to skip the initialization. """
    __singleton_accessed: bool = False

    def __new__(cls, gb: pp.GridBucket) -> Substance:
        """
        Declarator assures the substance name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables and usage
        of the name as a key.
        """
        name = str(cls.__name__)
        if gb in Substance.__substances_per_gb.keys():
            if name in Substance.__substances_per_gb[gb]:
                # flag that the singleton has been accessed and return it.
                Substance.__singleton_accessed = True
                return Substance.__susbtance_instances[name]
        else:
            Substance.__substances_per_gb.update({gb: list()})

        Substance.__substances_per_gb[gb].append(name)
        new_instance = super().__new__(cls)
        Substance.__susbtance_instances.update({name: new_instance})
        return new_instance

    def __init__(self, gb: pp.GridBucket) -> None:
        """Abstract base class constructor. Initiates component-related AD-variables.
        Contains symbolic names of associated model variables.

        :param gb: geometry in which the substance is modelled
        :type gb:
            :class:`~porepy.grids.grid_bucket.GridBucket`
        """
        # skipping re-instantiation if class if __new__ returned the previous reference
        if Substance.__singleton_accessed:
            Substance.__singleton_accessed = False
            return

        super().__init__()

        ## PUBLIC
        self.gb: pp.GridBucket = gb

        # creating the overall molar fraction variable
        self._omf = create_merged_variable(gb, {"cells": 1}, self.overall_fraction_var)
        # for a phase name (key),
        # provide the MergedVariable for the molar fraction in that phase (value)
        self._mfip: Dict[str, pp.ad.MergedVariable] = dict()

    @property
    def name(self) -> str:
        """
        :return: name of the substance class. The name is used to construct names for
        AD variables and keys to store them.
        :rtype: str
        """
        return str(self.__class__.__name__)

    @property
    def overall_fraction_var(self) -> str:
        """
        :return: name of the overall molar fraction variable under which it is stored
        in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["component_overall_fraction"] + "_" + self.name

    def fraction_in_phase_var(self, phase_name: str) -> str:
        """
        :param phase_name: name of the  :class:`~porepy.composite.phase.PhaseField` for which
        the fraction variable's name is requested
        :type phase_name: str

        :return: name of the molar fraction in phase variable
        :rtype: str
        """
        return (
            COMPUTATIONAL_VARIABLES["component_fraction_in_phase"]
            + "_"
            + self.name
            + "_"
            + str(phase_name)
        )

    @property
    def overall_fraction(self) -> pp.ad.MergedVariable:
        """As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`~porepy.ad.MergedVariable` representing
        the overall molar fraction of this component
        :rtype: :class:`~porepy.ad.MergedVariable`
        """
        return self._omf

    def fraction_in_phase(self, phase_name: str) -> pp.ad.MergedVariable:
        """As a fractional quantity, all values are between 0 and 1.

        :param phase_name: Name of the  :class:`~porepy.composit.PhaseField` for which
        the fractions are requested
        :type phase_name: str

        :return: reference to domain-wide :class:`~porepy.ad.MergedVariable`
        representing the molar fraction of this component in phase `phase_name`.
        :rtype: :class:`~porepy.ad.MergedVariable`
        """
        phase_name = str(phase_name)
        # if MergedVariable for this phase already exists, return it
        if phase_name in self._mfip.keys():
            return self._mfip[phase_name]
        # else create new one
        else:
            mfip = create_merged_variable(
                self.gb, {"cells": 1}, self.fraction_in_phase_var(phase_name)
            )
            self._mfip.update({phase_name: mfip})
            return mfip

    # ------------------------------------------------------------------------------
    ### CONSTANT SCALAR ATTRIBUTES
    # ------------------------------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [kg / mol]

        :return: molar mass of component (constant)
        :rtype: float
        """
        pass

    # ------------------------------------------------------------------------------
    ### SCALAR ATTRIBUTES (dependent on thermodynamic state)
    # ------------------------------------------------------------------------------

    def mass_density(self, pressure: float, temperature: float) -> float:
        """
        Uses the molar mass and molar density to compute the mass density.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m^3]

        :return: mass density of the component
        :rtype: float
        """
        return self.molar_mass() * self.molar_density(pressure, temperature)

    @abc.abstractmethod
    def molar_density(self, pressure: float, temperature: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :return: molar density of the component
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def Fick_diffusivity(self, pressure: float, temperature: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        m^2 / s

        :return: Fick's diffusivity coefficient (or tensor in the case of heterogeneity)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, pressure: float, temperature: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [W / m / K]

        :return: thermal conductivity of the substance
        :rtype: float
        """
        pass


class FluidSubstance(Substance):
    """
    A class extending the list of abstract physical properties with new ones,
    associated with fluid components.

    The extensive list includes:
        - dynamic_viscosity

    """

    @abc.abstractmethod
    def dynamic_viscosity(self, pressure: float, temperature: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [Pa s] = [kg / m / s]

        :return: dynamic viscosity of the fluid
        :rtype: float
        """
        pass


class SolidSubstance(Substance):
    """
    A class extending the list of abstract physical properties with new ones,
    associated with material for the skeleton of various porous media.

    The extensive list includes:
        - base porosity (constant)
        - base permeability (constant)
    """

    @staticmethod
    @abc.abstractmethod
    def base_porosity() -> float:
        """
        Constant value.

        Math. Dimension:        scalar
        Phys. Dimension:        dimensionsless, fractional

        :return: base porosity of the material
        :rtype: float
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def base_permeability() -> float:
        """
        Constant value.

        Math. Dimension:        scalar
        Phys. Dimension:        [m^2] ( [Darcy] not official SI unit)

        :return: base permeability of the material
        :rtype: float
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def poro_reference_pressure() -> float:
        """
        Constant value.

        Math. Dimension:        scalar
        Phys. Dimension:        [Pa]

        :return: reference pressure for i.e. linear pressure law for porosity
        :rtype: float
        """
