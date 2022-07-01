""" Contains the abstract base class for all phases."""
from __future__ import annotations

import abc
import warnings
from typing import Dict, Generator, List, Union

import numpy as np

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES, create_merged_variable

__all__ = ["Phase"]


class Phase(abc.ABC):
    """A class representing phases in a multiphase multicomponent mixture.

    The term 'phase' includes both, states of matter and general fields
    A phase is identified by the (time-dependent) region it occupies and a
    respective velocity field (or flux) in that region.

    The phase class is a singleton for each pair of given ``name`` and ``GridBucket``.
    If the same name is instantiated twice on the same grid bucket,
    the previous instance is returned.

    Attributes:
        gb (:class:`~porepy.GridBucket`): domain of computation.
            A phase has values in each cell.
        name (str): name and unique identifier of the phase
        saturation (:class:`porepy.ad.MergedVariable`): saturation of this phase.
        saturation_var (str): name of the saturation variable.
            Currently a combination of the general symbol and the name
    """

    """ For a grid bucket (keys), contains a list of present phases (values). """
    __gb_singletons: Dict[pp.GridBucket, Dict[str, Phase]] = dict()
    __singleton_accessed: bool = False

    def __new__(cls, gb: pp.GridBucket, name: str) -> Phase:
        """Assures the class is a name-gb-based Singleton."""
        if gb in Phase.__gb_singletons.keys():
            if name in Phase.__gb_singletons[gb].keys():
                # flag that the singleton has been accessed and return it.
                Phase.__singleton_accessed = True
                return Phase.__gb_singletons[gb][name]
        else:
            Phase.__gb_singletons.update({gb: dict()})
        
        # create a new instance and store it
        new_instance = super().__new__(cls)
        Phase.__gb_singletons[gb].update({name: new_instance})
        return new_instance

    def __init__(self, gb: pp.GridBucket, name: str) -> None:
        """Initiated phase-related AD variables.

        If the same combination of ``name`` and ``GridBucket`` was already used once,
        the reference to the previous instance will be returned,
        i.e. the Phase class is a gb-and-name-based singleton.

        Args:
            gb (:class:`~porepy.GridBucket`): computational domain.
                Phase-related quantities will be assumed to be present in each cell.
            name (str): a name for the phase which will be used as a unique identifier.
        """
        # skipping re-instantiation if class if __new__ returned the previous reference
        if Phase.__singleton_accessed:
            Phase.__singleton_accessed = False
            return

        super().__init__()

        # public attributes
        self.gb: pp.GridBucket = gb

        # private attributes
        self._s: pp.ad.MergedVariable
        self._fraction: pp.ad.MergedVariable
        self._name = str(name)
        self._present_components: List[pp.composite.Component] = list()
        # Instantiate saturation variable
        self._s = create_merged_variable(gb, {"cells": 1}, self.saturation_var)
        # Instantiate phase molar fraction variable
        self._molar_fraction = create_merged_variable(
            self.gb, {"cells": 1}, self.molar_fraction_var
        )

    def __iter__(self) -> Generator[pp.composite.Component, None, None]:
        """
        Generator over substances present in this phase.

        IMPORTANT:
        The order from this iterator is used for choosing e.g. the values in a
        list of 'numpy.array' when setting initial values.
        Use the order returned here every time you deal with substance-related values
        for substances in this phase.

        :return: yields present substance
        :rtype: :class:`~porepy.composite.substance.Substance`
        """
        for substance in self._present_components:
            yield substance

    @property
    def name(self) -> str:
        """
        The name is used to construct names for AD variables and keys to store them.

        :return: name of the phase field.
        :rtype: str
        """
        return self._name

    @property
    def saturation(self) -> pp.ad.MergedVariable:
        """Has the values resulting from the last flash.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] fractional

        Returns:
            :class:`~porepy.ad.MergedVariable`:
                saturation (of this phase), a secondary variable on the whole domain,
                representing values at equilibrium. Also known as volumetric phase fraction.
        """
        return self._s

    @property
    def saturation_var(self) -> str:
        """The name is used as name for the MergedVariable and
        as key for data in grid dictionaries.

        Returns:
            str: name of the saturation variable. 
                Currently a combination of the general symbol and the phase name.
        """
        return COMPUTATIONAL_VARIABLES["saturation"] + "_" + self.name

    @property
    def fraction(self) -> pp.ad.MergedVariable:
        """Has the values resulting from the last flash.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] fractional

        Returns:
            :class:`~porepy.ad.MergedVariable`:
                molar phase fraction (of this phase), a secondary variable on the whole domain,
                representing values at equilibrium.
        """
        return self._fraction

    @property
    def fraction_var(self) -> str:
        """The name is used as name for the MergedVariable and
        as key for data in grid dictionaries.

        Returns:
            str: name of the molar fraction variable variable. 
                Currently a combination of the general symbol and the phase name.
        """
        return COMPUTATIONAL_VARIABLES["molar_phase_fraction"] + "_" + self.name

    def add_component(
        self,
        component: Union[List[pp.composite.Component], pp.composite.Component],
    ) -> None:
        """Adds components which are expected by the model or calculations in this phase.

        Args:
            component (:class:`porepy.composite.Component`): a component,
                or list of components, which appear in this phase.
        """

        if isinstance(component, pp.composite.Component):
            component = [component]

        present_components = [ps.name for ps in self._present_components]

        for comp in component:
            #TODO check if this can be omitted with a set
            # (if Python compares Component instances correctly)
            # skip already present components:
            if comp.name in present_components:
                continue
            # store reference to present substance
            self._present_components.append(comp)

    def mass_density(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """
        Uses the  molar mass values in combination with the molar masses and fractions 
        of components in this phase, to compute the mass density of the phase.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / REV]

        Args:
            p (:class:`~porepy.ad.MergedVariable`): pressure
            T (:class:`~porepy.ad.MergedVariable`): temperature

        Returns:
            :class:`~porepy.ad.Operator`: mass density of this phase
        """

        weight = pp.ad.Scalar(0.0, "mass-density-%s" % (self.name))
        # if there are no substances in this phase, return a zero
        if not self._present_components:
            return weight

        # add the mass-weighted fraction for each present substance.
        for component in self._present_components:
            weight += component.molar_mass() * component.fraction_in_phase(self.name)

        # Multiply the mass weight with the molar density and return the operator
        return weight * self.molar_density(p, T)

    # ------------------------------------------------------------------------------
    ### Abstract, phase-related physical properties
    # ------------------------------------------------------------------------------

    @abc.abstractmethod
    def density(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """Abstract physical property, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present substances) can be accessed
        by internal reference.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / REV]

        Args:
            p (:class:`~porepy.ad.MergedVariable`): pressure
            T (:class:`~porepy.ad.MergedVariable`): temperature

        Returns:
            :class:`~porepy.ad.Operator`: mass density of this phase
        """
        pass

    @abc.abstractmethod
    def specific_enthalpy(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """Abstract physical quantity, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present components) can be accessed
        by internal reference.

        Math. Dimension:        scalar
        Phys.Dimension:         [kJ / mol / K]

        Args:
            p (:class:`~porepy.ad.MergedVariable`): pressure
            T (:class:`~porepy.ad.MergedVariable`): temperature

        Returns:
            :class:`~porepy.ad.Operator`: specific molar enthalpy of this phase
        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(
        self, p: pp.ad.MergedVariable, T: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """Abstract physical property, dependent on thermodynamic state and the composition.
        The composition variables (molar fractions of present substances) can be accessed
        by internal reference.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m / s]

        Args:
            p (:class:`~porepy.ad.MergedVariable`): pressure
            T (:class:`~porepy.ad.MergedVariable`): temperature

        Returns:
            :class:`~porepy.ad.Operator`: dynamic viscosity
        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(
        self, pressure: pp.ad.MergedVariable, temperature: pp.ad.MergedVariable
    ) -> pp.ad.Operator:
        """Abstract physical property, dependent on thermodynamic state and composition.
        The composition variables (molar fractions of present substances) can be accessed
        by internal reference.

        Math. Dimension:    2nd-order tensor
        Phys. Dimension:    [W / m / K]

        Args:
            p (:class:`~porepy.ad.MergedVariable`): pressure
            T (:class:`~porepy.ad.MergedVariable`): temperature

        Returns:
            :class:`~porepy.ad.Operator`: thermal conductivity
        """
        pass
