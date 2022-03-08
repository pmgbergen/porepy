""" Contains the abstract base class for all phases."""
from __future__ import annotations

import abc
import porepy as pp
import numpy as np

from ._composite_utils import COMPUTATIONAL_VARIABLES

from typing import Iterator, Tuple, Iterable, TYPE_CHECKING
# this solution avoids circular imports due to type checking. Needs __future__.annotations
if TYPE_CHECKING:
    from .computational_domain import ComputationalDomain
    from .substance import Substance


class Phase(abc.ABC):
    """
    Abstract base class for all phases. Provides functionalities to handle and manage anticipated components.
    
    Provides and manages phase-related AD-variables.

    Instantiated variables are provided as properties, as well as the names under which they are stored in the grid data dictionary.
    """

    """ For a computational domain (key), contains a list of present phases"""
    _present_phases = dict()

    def __new__(cls,
    name:str, computational_domain: ComputationalDomain):
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

    def __init__(self,
    name: str, computational_domain: ComputationalDomain) -> None:
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

    def __iter__(self) -> Iterator[Substance]:
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

    def add_substances(self,
    phase_composition: Iterable[Tuple[Substance, np.array]]) -> None:
        """ Adds a composition of substances to this phase, including their initial molar fraction.

        Asserts the sum over molar fractions is 1 per cell.
        Ergo, the array arguments have to be of length :method:`~porepy.grids.grid_bucket.GridBucket.num_cells`.

        The Substances have to instantiated using the same
        :class:`~porepy.composite.computational_domain.ComputationalDomain` instance as for this object.

        
        :param phase_composition: iterable object containing 2-tuples of substances and respective concentrations in this phase
        :type component: List[Tuple[:class:`~porepy.composite.substance.Substance`, :class:`~numpy.array`]]
        """
        # Sum of molar fractions in this phase. Has to be 1 in each cell
        sum_over_fractions = np.zeros(self.cd.gb.num_cells())

        for substance, fraction in phase_composition:
            # check if objects are meant for same domain
            if substance.cd != self.cd:
                raise ValueError("Substance '%s' instantiated on unknown ComputationalDomain."%(substance.name))
            # check if initial conditions (fractions) are complete
            if len(fraction) != len(sum_over_fractions):
                raise ValueError("Initial fraction of substance '%s' not given on correct number of cells"%(substance.name))
            # sum the fractions for assertion that it equals 1. on each cell
            sum_over_fractions += fraction

            # store reference to present substance
            self._present_substances.append(substance)
            # instantiate related AD variables
            self.cd(substance.mfip_name(self.name), {"cells": 1})
            # TODO needs maths
            self.cd(substance.mortar_mfip_name(self.name), {"cells": 1})

            ## setting initial values (fractions)
            # create a copy to avoid issues in case there is another manipulated reference to this values
            init_vals = np.copy(fraction)

            for grid_data, vals in zip(self.cd.gb, vals):
                data = grid_data[1]  # get data out (grid, data) tuple
                if pp.STATE not in data:
                    data[pp.STATE] = {}
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = {}

                data[pp.STATE][self.omf_name] = vals
                data[pp.STATE][pp.ITERATE][self.omf_name] = vals

            # TODO set initial values at interfaces, if applicable (the maths is missing here)
            # check if above code works for GridBuckets without fractures
            
        # assert the fractional character (sum equals 1) is given on each cell
        if np.any(sum_over_fractions != 1.): #TODO check sensitivity
            raise ValueError("Initial fractions do not sum up to 1. on each cell.")