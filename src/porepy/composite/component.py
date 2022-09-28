"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc
from typing import Dict, Optional, Union

import porepy as pp

__all__ = ["Component", "FluidComponent", "SolidComponent"]

VarLike = Union[pp.ad.MergedVariable, pp.ad.Variable, pp.ad.Operator, float, int]


class Component(abc.ABC):
    """Abstract base class for chemical components, providing abstract physical properties
    which need to be implemented for concrete child classes to work in PorePy.

    Provides and manages component-related physical quantities and properties.

    If a :class:`~porepy.ad.ADSystemManager` is passed at instantiation, the AD framework is
    is used to represent the fractional variables and the component class becomes a singleton
    with respect to the mixed-dimensional domain contained in the AD system.
    Unique instantiation over a given domain is assured by using this class's name as an unique
    identifier.
    Ambiguities and uniqueness must be assured due to central storage of the fractional values
    in the grid data dictionaries.

    If the AD system is not passed, this class can be used in standalone mode.
    Respective fractional variables are not existent in this case and the
    :class:`~porepy.composite.Composition` takes over the computation and storage of values.

    Parameters:
        ad_system (optional): If given, this class will use the AD framework and the respective
            mixed-dimensional domain to represent fractions cell-wise in each subdomain.

    """

    # contains per mdg the singleton, using the class name as a unique identifier
    __ad_singletons: Dict[pp.MixedDimensionalGrid, Dict[str, Component]] = dict()
    # flag if a singleton has recently been accessed, to skip re-instantiation
    __singleton_accessed: bool = False

    def __new__(cls, ad_system: Optional[pp.ad.ADSystemManager] = None) -> Component:
        # class name is used as unique identifier
        name = str(cls.__name__)

        # check for AD singletons per grid
        if ad_system:
            mdg = ad_system.dof_manager.mdg
            if mdg in Component.__ad_singletons:
                if name in Component.__ad_singletons[mdg]:
                    # flag that the singleton has been accessed and return it.
                    Component.__singleton_accessed = True
                    return Component.__ad_singletons[mdg][name]
            else:
                Component.__ad_singletons.update({mdg: dict()})

        # create a new instance and store it, if no previous instantiations were found
        new_instance = super().__new__(cls)
        if ad_system:
            Component.__ad_singletons[mdg].update({name: new_instance})

        return new_instance

    def __init__(self, ad_system: Optional[pp.ad.ADSystemManager] = None) -> None:

        # skipping re-instantiation if class if __new__ returned the previous reference
        if Component.__singleton_accessed:
            Component.__singleton_accessed = False
            return

        super().__init__()

        ### PUBLIC

        self.ad_system: Optional[pp.ad.ADSystemManager] = ad_system
        """The AD system optionally passed at instantiation."""

        #### PRIVATE

        # creating the overall molar fraction variable
        self._fraction: Optional[pp.ad.MergedVariable] = None
        if ad_system:
            self._fraction = ad_system.create_variable(
                self.fraction_var_name, True
            )
        else:
            self._fraction = 0.

        # for a phase name (key),
        # provide the MergedVariable for the molar fraction in that phase (value)
        self._fractions_in_phases: Dict[str, Optional[pp.ad.MergedVariable]] = dict()

    @property
    def name(self) -> str:
        """
        Returns: name of the class, used as a unique identifier.

        """
        return str(self.__class__.__name__)

    @property
    def fraction_var_name(self) -> str:
        """Name of the feed fraction variable, given by the general symbol and :meth:`name`."""
        return "z" + "_" + self.name

    # def fraction_in_phase_var(self, phase_name: str) -> str:
    #     """
    #     Parameters:
    #         phase_name: :meth:`~porepy.composite.Phase.name` of the phase in which this
    #             component is present.

    #     Returns:
    #         name of the respective variable, given by the general symbol, the
    #         component name and the phase name

    #     """
    #     return f"x_{self.name}_{phase_name}"

    @property
    def fraction(self) -> Optional[pp.ad.MergedVariable]:
        """Initialized with 0.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        If no AD system is present, the feed fraction value can be set directly.
        If the AD framework is used, respective functionalities must be used to set values for
        the feed fraction (merged) variable, or alternatively
        :meth:`~porepy.composite.Composition.set_feed_composition`

        Parameters:
            value: a value to set the feed fraction of this component for standalone
                applications.

        Returns:
            feed fraction, a primary variable on the whole domain (cell-wise).
            Indicates how many of the total moles belong to this component.

        """
        return self._fraction
    
    @fraction.setter
    def fraction(self, value: float) -> None:
        if self.ad_system:
            raise RuntimeError(
                f"Cannot set the component fraction when AD is used. "
                "Use respective functionalities of the AD system or the Composition class."
            )
        else:
            self._fraction = value

    # def fraction_in_phase(self, phase_name: str) -> Optional[pp.ad.MergedVariable]:
    #     """
    #     | Math. Dimension:        scalar
    #     | Phys. Dimension:        [-] fractional

    #     Parameters:
    #         phase_name: :meth:`~porepy.composite.Phase.name` of the phase in which this
    #             component is present.

    #     Returns:
    #         fraction in phase, a secondary variable on the whole domain (cell-wise).
    #         Indicates how many moles in a present phase belongs to this component.
    #         It is supposed to represent the value at thermodynamic equilibrium.

    #     """
    #     # if variable for this phase already exists, return it
    #     if phase_name in self._fractions_in_phases.keys():
    #         return self._fractions_in_phases[phase_name]
    #     # else create new one otherwise
    #     else:
    #         # for AD systems create a merged variable
    #         if self.ad_system:
    #             mfip = self.ad_system.create_variable(
    #                 self.fraction_in_phase_var(phase_name), False)
    #         # for standalone applications this remains None
    #         else:
    #             mfip = None
    #         # store and return the variable
    #         self._fractions_in_phases.update({phase_name: mfip})
    #         return mfip

    def mass_density(self, p: VarLike, T: VarLike) -> VarLike:
        """Uses the molar mass and molar density to compute the mass density.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: mass density.

        """
        return self.molar_mass() * self.density(p, T)

    # -----------------------------------------------------------------------------------------
    ### ABSTRACT PHYSICAL PROPERTIES
    # -----------------------------------------------------------------------------------------

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / mol]

        Returns: molar mass.

        """
        pass

    # -----------------------------------------------------------------------------------------
    ### NON-CONSTANT ABSTRACT PHYSICAL PROPERTIES
    # -----------------------------------------------------------------------------------------

    @abc.abstractmethod
    def density(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: molar density.

        """
        pass

    @abc.abstractmethod
    def Fick_diffusivity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        Notes:
            This can also be a tensor and will be submitted respective changes in the future.

        | Math. Dimension:        scalar
        | Phys. Dimension:        m^2 / s

        Parameters:
            p: pressure
            T: temperature

        Returns: Fick diffusivity coefficient.

        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        Notes:
            This can also be a tensor and will be submitted respective changes in the future.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [W / m / K]

        Parameters:
            p: pressure
            T: temperature

        Returns: thermal conductivity for Fourier's law.

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

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa]

        Returns: critical pressure for this component (critical point in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns: critical temperature for this component (critical point in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_pressure() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa]

        Returns:
            triple point pressure for this component
            (intersection of vapor and melting curve in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns:
            triple point temperature for this component
            (intersection of vapor and melting curve in p-T diagram).

        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [Pa s] = [kg / m / s]

        Args:
            p: pressure
            T: temperature

        Returns: dynamic viscosity.

        """
        pass


class SolidComponent(Component):
    """Extends the list of necessary physical attributes for components by those which are
    usually used for solids in elasticity and plasticity problems, or which are used to
    for the porous medium.

    """

    @staticmethod
    @abc.abstractmethod
    def base_porosity() -> float:
        """Constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns: base porosity of the material.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def base_permeability() -> float:
        """Constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [m^2] ( [Darcy] not official SI unit)

        Returns: base permeability of the material.

        """
        pass
