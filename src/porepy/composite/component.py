"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc
from typing import Dict

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_subdomain_variable

__all__ = ["Component", "FluidComponent", "SolidComponent"]


class Component(abc.ABC):
    """
    Abstract base class for chemical components, providing abstract physical properties
    which need to be implemented for concrete child classes to work in PorePy.

    Provides and manages component-related AD-variables.

    Turns every Substance child class into a ``gb``-singleton,
    such that there is only one instance per grid bucket which is is returned if the substance
    is instantiated in e.g., multiple phases
    This assures that all phases access "the same" substance.

    Attributes:
        gb (:class:`~porepy.GridBucket`): domain of computation.
            A component has values in each cell.
        name (str): class name of the component, used as a unique identifier
        fraction (:class:`porepy.ad.MergedVariable`): feed fraction of this component per cell
        fraction_var (str): name of the saturation variable.
            Currently a combination of the general symbol and the name.

    """

    # contains per GB the singleton, using the class name as a unique identifier
    __mdg_singletons: Dict[pp.MixedDimensionalGrid, Dict[str, Component]] = dict()
    # flag if a singleton has recently been accessed, to skip re-instantiation
    __singleton_accessed: bool = False

    def __new__(cls, mdg: pp.MixedDimensionalGrid) -> Component:
        """Declarator assures the substance name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables and usage
        of the name as a key.

        """

        name = str(cls.__name__)
        if mdg in Component.__mdg_singletons.keys():
            if name in Component.__mdg_singletons[mdg].keys():
                # flag that the singleton has been accessed and return it.
                Component.__singleton_accessed = True
                return Component.__mdg_singletons[mdg][name]
        else:
            Component.__mdg_singletons.update({mdg: dict()})

        # create a new instance and store it
        new_instance = super().__new__(cls)
        Component.__mdg_singletons[mdg].update({name: new_instance})
        return new_instance

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Initiates component-related AD-variables.

        Args:
            gb (:class:`~porepy.GridBucket`): computational domain.
                Component-related quantities will be assumed to be present in each cell.

        """

        # skipping re-instantiation if class if __new__ returned the previous reference
        if Component.__singleton_accessed:
            Component.__singleton_accessed = False
            return

        super().__init__()

        ### PUBLIC
        self.mdg: pp.MixedDimensionalGrid = mdg

        #### PRIVATE
        # creating the overall molar fraction variable
        self._feed_fraction: pp.ad.MergedVariable = create_merged_subdomain_variable(
            mdg, {"cells": 1}, self.fraction_var
        )
        self._fractions_in_phases: Dict[str, pp.ad.MergedVariable] = dict()

        # for a phase name (key),
        # provide the MergedVariable for the molar fraction in that phase (value)
        self._mfip: Dict[str, pp.ad.MergedVariable] = dict()

    @property
    def name(self) -> str:
        """
        Returns:
            str: name of the class, used as a unique identifier.

        """
        return str(self.__class__.__name__)

    @property
    def fraction_var(self) -> str:
        """
        Returns:
            str: name of the feed fraction variable, given by the general symbol and the
                component class name.

        """
        return COMPUTATIONAL_VARIABLES["component_fraction"] + "_" + self.name

    @property
    def fraction(self) -> pp.ad.MergedVariable:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [-] fractional

        Returns:
            :class:`~porepy.ad.MergedVariable`: feed fraction, a primary variable on the
                whole domain. Says how much of the present mass per cell belongs to this
                component.

        """
        return self._feed_fraction

    def fraction_in_phase_var(self, phase_name: str) -> str:
        """
        Args:
            phase_name (str): :meth:`~porepy.composite.Phase.name` of the phase in which this
                component is present.

        Returns:
            str: name of the respective variable, consisting of the general symbol, the
                component name and the phase name

        """

        return (
            COMPUTATIONAL_VARIABLES["component_fraction_in_phase"]
            + "_"
            + self.name
            + "_"
            + str(phase_name)
        )

    def fraction_in_phase(self, phase_name: str) -> pp.ad.MergedVariable:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [-] fractional

        Args:
            phase_name (str): :meth:`~porepy.composite.Phase.name` of the phase in which this
                component is present.

        Returns:
            :class:`~porepy.ad.MergedVariable`: fraction in phase, a secondary variable on the
                whole domain. Says how much of the mass in given phase per cell belongs to this
                component.

        """

        phase_name = str(phase_name)
        # if MergedVariable for this phase already exists, return it
        if phase_name in self._mfip.keys():
            return self._mfip[phase_name]
        # else create new one
        else:
            mfip = create_merged_subdomain_variable(
                self.mdg, {"cells": 1}, self.fraction_in_phase_var(phase_name)
            )
            self._mfip.update({phase_name: mfip})
            return mfip

    def mass_density(self, p: float, T: float) -> float:
        """
        Uses the molar mass and molar density to compute the mass density.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / REV]

        Args:
            p (float): pressure
            T (float): temperature

        Returns:
            float: mass density

        """
        return self.molar_mass() * self.density(p, T)

    # -----------------------------------------------------------------------------------------
    ### ABSTRACT PHYSICAL PROPERTIES
    # -----------------------------------------------------------------------------------------

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / mol]

        Returns:
            float: molar mass

        """
        pass

    # -----------------------------------------------------------------------------------------
    ### NON-CONSTANT ABSTRACT PHYSICAL PROPERTIES
    # -----------------------------------------------------------------------------------------

    @abc.abstractmethod
    def density(self, p: float, T: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [mol / REV]

        Args:
            p (float): pressure
            T (float): temperature

        Returns:
            float: molar density

        """
        pass

    @abc.abstractmethod
    def Fick_diffusivity(self, p: float, T: float) -> float:
        """TODO: This can also be a tensor

        Math. Dimension:        scalar
        Phys. Dimension:        m^2 / s

        Args:
            p (float): pressure
            T (float): temperature

        Returns:
            float: Fick diffusivity coefficient

        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, p: float, T: float) -> float:
        """TODO: THis can also be a tensor

        Math. Dimension:        scalar
        Phys. Dimension:        [W / m / K]

        Args:
            p (float): pressure
            T (float): temperature

        Returns:
            float: thermal conductivity for Fourier's law

        """
        pass


class FluidComponent(Component):
    """Extends the list of necessary physical attributes for components by those which are
    usually used for fluids in flow problems.

    """

    @staticmethod
    @abc.abstractmethod
    def critical_pressure() -> float:
        """This is a constant value, hence to be a static function.

        Math. Dimension:        scalar
        Phys. Dimension:        [kPa]

        Returns:
            float: critical pressure for this component (critical point in p-T diagram)

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_temperature() -> float:
        """This is a constant value, hence to be a static function.

        Math. Dimension:        scalar
        Phys. Dimension:        [K]

        Returns:
            float: critical temperature for this component (critical point in p-T diagram)

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_pressure() -> float:
        """This is a constant value, hence to be a static function.

        Math. Dimension:        scalar
        Phys. Dimension:        [kPa]

        Returns:
            float: triple point pressure for this component
                (intersection of vapor and melting curve in p-T diagram)

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_temperature() -> float:
        """This is a constant value, hence to be a static function.

        Math. Dimension:        scalar
        Phys. Dimension:        [K]

        Returns:
            float: triple point temperature for this component
                (intersection of vapor and melting curve in p-T diagram)

        """
        pass


    @abc.abstractmethod
    def dynamic_viscosity(self, p: float, T: float) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [Pa s] = [kg / m / s]

        Args:
            p (float): pressure
            T (float): temperature

        Returns:
            float: dynamic viscosity

        """
        pass


class SolidComponent(Component):
    """Extends the list of necessary physical attributes for components by those which are
    usually used for solids in elasticity and plasticity problems
    """

    @staticmethod
    @abc.abstractmethod
    def base_porosity() -> float:
        """
        Constant value, hence to be a static function

        Math. Dimension:        scalar
        Phys. Dimension:        [-] fractional

        Returns:
            float base porosity of the material

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def base_permeability() -> float:
        """
        Constant value.

        Math. Dimension:        scalar
        Phys. Dimension:        [m^2] ( [Darcy] not official SI unit)

        Returns:
            float: base permeability of the material

        """
        pass
