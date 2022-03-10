""" Contains the abstract base class for all phases."""
from __future__ import annotations

# import abc
from multiprocessing.sharedctypes import Value
import porepy as pp
import numpy as np
from weakref import WeakValueDictionary, WeakKeyDictionary

from ._composite_utils import COMPUTATIONAL_VARIABLES

from typing import Iterator, Tuple, Iterable, Dict, List, TYPE_CHECKING
# this solution avoids circular imports due to type checking. Needs __future__.annotations
if TYPE_CHECKING:
    from .computational_domain import ComputationalDomain
    from .substance import Substance, FluidSubstance


class Phase:#(abc.ABC):
    """
    Base class for all phases. Provides functionalities to handle and manage anticipated components.

    NOTE proposition:
    Turns every child in a conditional singleton class: It can only be instantiated once per computational domain.
    Any further instantiation will return a reference to the same, first class.
    Assures uniqueness and integrity of variable values (like concentrations) in grid-specific dictionaries.
    
    Provides and manages phase-related AD-variables.

    Instantiated variables are provided as properties, as well as the names under which they are stored in the grid data dictionary.
    
    Current relevant variables (per phase instance):
        - saturation
        - phase molar fraction
    """

    """ For a computational domain (key), contains a list of present phases. """
    __present_phases: Dict[ComputationalDomain, list] = dict()
    # """ For a """
    # __singletons: dict = dict()
    # """Switch for skipping the instantiation of phase instances, which are already present."""
    # __new_instance = True

    def __new__(cls,
    name: str, solvent: FluidSubstance, computational_domain: ComputationalDomain) -> Phase:
        """ Declarator. Assures the phase name is unique for a given computational domain.
        Ambiguities must be avoided due to the central storage of the AD variables and usage of the name as a key
        """
        name = str(name)
        if computational_domain in Phase.__present_phases.keys():
            if name in Phase.__present_phases[computational_domain]:
                # reference = Phase.__present_phases[computational_domain]
                # Phase.__new_instance = False
                raise RuntimeError("Phase with name '" + name + "' already present in\n" + str(computational_domain))
            else:
                Phase.__present_phases[computational_domain].append(name)
        else:
            Phase.__present_phases.update({computational_domain: list()})
        
        return super().__new__(cls)

    def __init__(self,
    name: str, solvent: FluidSubstance, computational_domain: ComputationalDomain) -> None:
        """
        Base class constructor. Initiates phase-related AD-variables.
        
        :param name: name of the phase
        :type name: str
        :param solvent: solvent substance of this phase. Influences the phase bahaviour.
        :type solvent: :class:`~porepy.composite.substance.FluidSubstance` 

        :param computational_domain: domain of computations containing the phase
        :type computational_domain: :class:`~porepy.composite.computational_domain.ComputationalDomain` 
        """
        super().__init__()

        # public properties
        self.cd = computational_domain

        # private properties
        self._name = str(name)
        self._solvent: FluidSubstance = solvent
        self._present_substances: List[Substance] = list()

        # Instantiate saturation variable
        self.cd(self.saturation_name)
        # TODO maths
        # self.cd(self.mortar_saturation_name)
        # Instantiate phase molar fraction variable
        self.cd(self.molar_fraction_name)
        #TODO maths
        # self.cd(self.mortar_molar_fraction_name)

    def __iter__(self) -> Iterator[Substance]:
        """ Iterator over substances present in phase.

        :return: yields present substance
        :rtype: :class:`~porepy.composite.substance.Substance`
        """
        for substance in self._present_substances:
            yield substance

    @property
    def name(self):
        """
        :return: name of the class. The name is used to construct names for AD variables and keys to store them.
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

    def mass_phase_density(self) -> pp.ad.Operator:
        """ 
        Uses the  molar mass values  in combination with the molar fractions in this phase to compute
        the mass density of the phase

        rho_e_mass = sum_s density_e_molar() * chi_s_e * M_s

        This holds since 
            density_e_molar() * chi_s_e
        give the number of poles of the substance present in this phase and
        the molar mass M_s converts the physical dimension.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m^3]

        :return: mass density of the phase (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        # compute the mass-weighted fraction of the solvent
        mass_fraction_sum = self._solvent.molar_mass()* self._solvent.molar_fraction_in_phase(self.name)

        # add the mass-weighted fraction for each present substance.
        for substance in self._present_substances:
            mass_fraction_sum += substance.molar_mass() * substance.molar_fraction_in_phase(self.name)
        
        # Multiply the mass weight with the molar density and return the operator
        return mass_fraction_sum  * self.molar_phase_density() 

#------------------------------------------------------------------------------
### HEURISTIC LAWS NOTE all heuristic laws can be modularized somewhere and referenced here
#------------------------------------------------------------------------------

    def molar_phase_density(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """ 
        Currently supported heuristic laws (values for 'law'):
            - 'stupid':        just stupid (VL)
            - 'solvent':       uses the unmodified solvent density

        Inherit this class and overwrite this method if you want to implement special models for the phase density.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: molar density of the phase (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "stupid":

            def stupid(p, h, *fracs):
                """
                Stupid, incorrect law using substance densities and molar fractions. 
                Thanks VL for making this mistake.
                """
                solv_dens = self._solvent.molar_density(p, h) * self._solvent.molar_fraction_in_phase(self.name)
                for subs, frac in zip(self._present_substances, fracs):
                    solv_dens += subs.molar_density(p, h) * frac

            molar_fractions = (subs.molar_fraction_in_phase(self.name) for subs in self._present_substances)
            pressure = self.cd(COMPUTATIONAL_VARIABLES["pressure"])
            enthalpy = self.cd(COMPUTATIONAL_VARIABLES["enthalpy"])

            return pp.ad.Function(stupid , "molar-density-%s-%s"%(law, self.name))(
                pressure, enthalpy, *molar_fractions)
        if law == "solvent":
            return pp.ad.Function(lambda mu: mu , "molar-density-%s-%s"%(law, self.name))(
                self._solvent.molar_density())
        else:
            raise NotImplementedError("Unknown 'law' keyword for phase viscosity.: %s \n"%(law)
            + "Available: 'solvent,'")

    def phase_viscosity(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """ 
        Currently supported heuristic laws (values for 'law'):
            - 'solvent':        uses the unmodified solvent viscosity

        Inherit this class and overwrite this method if you want to implement special models for the phase viscosity.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m / s]

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: dynamic viscosity of the phase (dependent on thermodynamic state and the composition)
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "solvent":
            return pp.ad.Function(lambda mu: mu , "viscosity-%s-%s"%(law, self.name))(self._solvent.dynamic_viscosity())
        else:
            raise NotImplementedError("Unknown 'law' keyword for phase viscosity.: %s \n"%(law)
            + "Available: 'solvent,'")

    def relative_permeability(self, law: str, *args, **kwargs) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'brooks_corey':   Brook-Corey model TODO finish
            - 'quadratic':      quadratic power law for saturation

        Inherit this class and overwrite this method if you want to implement special models for the relative permeability.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: relative permeability using the respectie law
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "quadratic":
            return pp.ad.Function(lambda S: S ** 2, "rel-perm-%s-%s"%(law, self.name))(self.saturation)
        else:
            raise NotImplementedError("Unknown 'law' keyword for rel.Perm.: %s \n"%(law)
            + "Available: 'quadratic,'")