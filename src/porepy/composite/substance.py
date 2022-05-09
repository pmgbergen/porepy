"""
Containts the abstract base class for all components (species/ pure substances)
used in this framework.
"""
from __future__ import annotations

import abc

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES

__all__ = [
    "Substance",
    "FluidSubstance",
    "SolidSubstance"
]


class Substance(abc.ABC):
    """
    Abstract base class for pure substances, providing abstract physical propertis
    which need to be implemented for concrete child classes to work in PorePy.

    NOTE proposition:
    Turns every child in a conditional singleton class: It can only be instantiated once
    per computational domain.
    Any further instantiation will return a reference to the same, first class.
    Assures uniqueness and integrity of variable values (like concentrations)
    in grid-specific dictionaries.

    Provides and manages component-related AD-variables.

    Instantiated AD variables are provided as properties, as well as the names under which
    they are stored in the grid data dictionary.

    Current relevant variables (per substance instance):
        - overall molar fraction
        - molar fraction in phase for given phase name

    Physical attributes include constants and scalar functions.
    The latter one is dependent on the thermodynamic state (e.g. pressure, enthalpy).

    IMPORTANT:  The first argument of every physical attribute is 'state_of_matter',
        a string indicating for which state of matter the respective attribute is requested.
        The rest of the thermodynamic arguments is free to choose

    IMPORTANT:  The doc strings of the abstract properties and methods contain information
                about intended physical dimensions.
                Keep it consistent when deriving child classes!

    1. Constants
        - Molar Mass

    2. Scalars (THD dependent)
        - molar density
        - Fick diffusivity coefficient
        - thermal conductivity coefficient

    """

    def __init__(self, computational_domain: "pp.composite.CompositionalDomain") -> None:
        """Abstract base class constructor. Initiates component-related AD-variables.
        Contains symbolic names of associated model variables.

        :param computational_domain: domain of computations containing the component
        :type computational_domain:
            :class:`~porepy.composite.computational_domain.ComputationalDomain`
        """
        super().__init__()

        ## PUBLIC
        self.cd = computational_domain

        ## PRIVATE
        self._registered_phases: set = set()

        # adding the overall molar fraction to the primary variables
        self.cd(self.omf_var, {"cells": 1})

    @property
    def name(self) -> str:
        """
        :return: name of the substance class. The name is used to construct names for
        AD variables and keys to store them.
        :rtype: str
        """
        return str(self.__class__.__name__)

    @property
    def omf_var(self) -> str:
        """
        :return: name of the overall molar fraction variable under which it is stored
        in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["component_overall_fraction"] + "_" + self.name

    def mfip_var(self, phase_name: str) -> str:
        """
        :param phase_name: name of the  :class:`~porepy.composite.phase.Phase` for which
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
    def overall_molar_fraction(self) -> pp.ad.MergedVariable:
        """As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing
        the overall molar fraction of this component
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.omf_var)

    def molar_fraction_in_phase(self, phase_name: str) -> pp.ad.MergedVariable:
        """As a fractional quantity, all values are between 0 and 1.

        :param phase_name: Name of the  :class:`porepy.composig.Phase` for which
        the fractions are requested
        :type phase_name: str

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable`
        representing the molar fraction of this component in phase `phase_name`.
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        # in order to avoid uncontrolled or arbitrary creation of variables,
        # we can perform a check if the phase is present at all
        # NOTE this limits the possible order of instantiations and method calls.
        # phase_name = str(phase_name)
        # if phase_name in [phase.name for phase in self.cd.Phases]:
        self._registered_phases.add(str(phase_name))
        return self.cd(self.mfip_var(phase_name))
        # else:
        #     raise ValueError("Phase '%s' not present in computational domain."%(phase_name))

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
    ### SCALAR ATTRIBUTES
    # ------------------------------------------------------------------------------

    def mass_density(self, state_of_matter: str, *args, **kwargs) -> float:
        """
        Non-abstract. Uses the molar mass and mass density to compute it.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m^3]

        :return: mass density of the component (dependent on thermodynamic state)
        :rtype: float
        """
        return self.molar_mass * self.molar_density(state_of_matter, *args, **kwargs)

    @abc.abstractmethod
    def molar_density(self, state_of_matter: str, *args, **kwargs) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :return: molar density of the component (dependent on thermodynamic state)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def Fick_diffusivity(self, state_of_matter: str, *args, **kwargs) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        m^2 / s

        :return: Fick's diffusivity coefficient (or tensor in the case of heterogeneity)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, state_of_matter: str, *args, **kwargs) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [W / m / s]

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
    def dynamic_viscosity(self, state_of_matter: str, *args, **kwargs) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m / s]

        :return: dynamic viscosity of the fluid (dependent on thermodynamic state)
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
