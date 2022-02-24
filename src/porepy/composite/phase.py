""" Contains the abstract base class for all phases."""

import abc

import porepy as pp

from ._composite_utils import create_merged_variable_on_gb

class Phase(abc.ABC):
    """
    Abstract base class for all phases. Provides functionalities to handle and manage anticipated components.
    
    Provides and manages phase-related AD-variables.

    Instantiated variables are provided as properties, as well as the names under which they are stored in the grid data dictionary.
    """

    def __init__(self, name: str, grid_bucket: pp.GridBucket) -> None:
        """Base class constructor. Initiates phase-related AD-variables.
        Contains symbolic names of associated model variables.
        
        :param name: name of the phase
        :type name: str

        :param grid_bucked: domain of computations containing the phase
        :type grid_bucket: :class:`porepy.GridBucket`
        """
        super().__init__()

        # private properties
        self._mortar_prefix: str = "mortar"
        self._gb = grid_bucket
        self._present_components = list()
        self._name = str(name)

        self._saturation: pp.ad.MergedVariable
        self._mortar_saturation: pp.ad.MergedVariable
        self._saturation_var: str = "S"

        self._molar_fraction: pp.ad.MergedVariable
        self._mortar_mf : pp.ad.MergedVariable
        self._mf_var: str = "xi"

        self._instantiate_phase_variables()

    @property
    def name(self) -> str:
        """ This attribute is protected by the property decorator since it is essential for the framework to function.
        It must not be manipulated after instantiation.

        :return: name of the phase, given at instantiation
        :rtype: str
        """
        return self._name

    @property
    def saturation(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the saturation of this phase
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._saturation

    @property
    def saturation_name(self) -> str:
        """
        :return: name of the saturation variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._saturation_var + "_" + self.name

    @property
    def mortar_saturation(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the saturation of this phase on the mortar grids
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._mortar_saturation
    
    @property
    def mortar_saturation_name(self) -> str:
        """
        :return: name of the mortar saturation variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._mortar_prefix + "_" + self._saturation_var + "_" + self.name

    @property
    def molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this phase
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._molar_fraction

    @property
    def mf_name(self) -> str:
        """
        :return: name of the phase molar fraction variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._mf_var + "_" + self.name

    @property
    def molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this phase on the mortar grids
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._mortar_mf

    @property
    def mortar_mf_name(self) -> str:
        """
        :return: name of the mortar phase molar fraction variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._mortar_prefix + "_" + self._mf_var + "_" + self.name

    def add_component(self, component: pp.composite.Component) -> None:
        """ Adds a component to this phase and registers this phase with the component.
        A phase-related molar fraction variable is created in the component.
        
        :param component: a component/species anticipated in this phase
        :type component: :class:`porepy.composite.Component`
        """
        # associate component with this phase. Might need more elegant solution in the future
        component.create_molar_fraction_in_phase(self.name)
        self._present_components.append(component)

    def _instantiate_phase_variables(self) -> None:
        """ Creates the AD representation of the saturation and phase molar fraction.
        These are primary variables in the composite flow.
        """
        # creating and storing the saturation variable
        sat_var, mortar_sat_var = create_merged_variable_on_gb(
            self._gb, {"cells": 1},  self.saturation_name, self.mortar_saturation_name
        )

        self._saturation = sat_var
        self._mortar_saturation = mortar_sat_var

        # creating and storing the phase molar fraction variable
        mf_var, mortar_mf_var = create_merged_variable_on_gb(
            self._gb, {"cells": 1},  self.mf_name, self.mortar_mf_name
        )

        self._molar_fraction = mf_var
        self._mortar_mf = mortar_mf_var