""" Contains the abstract base class for all phases."""

import abc

import porepy as pp

from typing import Generator

from porepy.composite.substance import Substance

from ._composite_utils import create_merged_variable, COMPUTATIONAL_VARIABLES
from.computational_domain import ComputationalDomain

class Phase(abc.ABC):
    """
    Abstract base class for all phases. Provides functionalities to handle and manage anticipated components.
    
    Provides and manages phase-related AD-variables.

    Instantiated variables are provided as properties, as well as the names under which they are stored in the grid data dictionary.
    """

    """ For a computational domain (key), contains a list of present phases"""
    _present_phases = dict()

    def __new__(cls, name:str, computational_domain: pp.ComputationalDomain):
        """ Declarator. Assures the phase name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables.
        """
        name = str(name)
        if computational_domain in cls._present_phases.keys():
            if name in cls._present_phases[computational_domain]:
                raise RuntimeError("Phase with name '" + name + "' already present in\n" + str(computational_domain))
            else:
                cls._present_phases[computational_domain].append(name)
        else:
            cls._present_phases.update({computational_domain: list()})
        
        return super().__new__(cls)

    def __init__(self, name: str, computational_domain: pp.ComputationalDomain) -> None:
        """Base class constructor. Initiates phase-related AD-variables.
        Contains symbolic names of associated model variables.
        
        :param name: name of the phase
        :type name: str

        :param computational_domain: domain of computations containing the phase
        :type computational_domain: :class:`~porepy.composite.computational_domain.ComputationalDomain` 
        """
        super().__init__()

        # public properties
        self.cd = computational_domain

        # private properties
        self._present_substances = list()
        self._name = str(name)

        # Instantiate saturation variable
        self.cd(self.saturation_name)
        # TODO maths
        self.cd(self.mortar_saturation_name)
        # Instantiate phase molar fraction variable
        self.cd(self.molar_fraction_name)
        #TODO maths
        self.cd(self.mortar_molar_fraction_name)

    def __iter__(self) -> Generator[Substance]:
        """ Iterator over substances present in phase.

        :return: yields present substance
        :rtype: :class:`~porepy.composite.substance.Substance`
        """
        for substance in self._present_substances:
            yield substance

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
        return self.cd(self.saturation_name)

    @property
    def saturation_name(self) -> str:
        """
        :return: name of the saturation variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["saturation"] + "_" + self.name

    @property
    def mortar_saturation(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the saturation of this phase on the mortar grids
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.mortar_saturation_name)
    
    @property
    def mortar_saturation_name(self) -> str:
        """
        :return: name of the mortar saturation variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["mortar_prefix"] + "_" + COMPUTATIONAL_VARIABLES["saturation"] + "_" + self.name

    @property
    def molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this phase
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.molar_fraction_name)

    @property
    def molar_fraction_name(self) -> str:
        """
        :return: name of the phase molar fraction variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["phase_molar_fraction"] + "_" + self.name

    @property
    def mortar_molar_fraction(self) -> pp.ad.MergedVariable:
        """
        As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this phase on the mortar grids
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self.cd(self.mortar_molar_fraction_name)

    @property
    def mortar_molar_fraction_name(self) -> str:
        """
        :return: name of the mortar phase molar fraction variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return COMPUTATIONAL_VARIABLES["mortar_prefix"] + "_" + COMPUTATIONAL_VARIABLES["phase_molar_fraction"] + "_" + self.name

    def add_substance(self, substance: pp.composite.Substance) -> None:
        """ Adds a component to this phase and registers this phase with the component.
        A phase-related molar fraction variable is created in the component.
        
        :param component: a component/species anticipated in this phase
        :type component: :class:`porepy.composite.Component`
        """
        # associate component with this phase.
        self._present_substances.append(substance)
        # instantiate related AD variables
        self.cd(substance.molar_fraction_in_phase(self.name))
        # TODO needs maths
        self.cd(substance.mortar_molar_fraction_in_phase(self.name))
