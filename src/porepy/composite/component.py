""" Containts the abstract base class for all components (species/ pure substances) used in this framework. """

import abc

from typing import Union, Dict, List, Tuple

from cv2 import phase

import porepy as pp
import numpy as np

from ._composite_utils import create_merged_variable_on_gb


class Component(abc.ABC):
    """
    Abstract base class for pure substances, providing abstract physical propertis which need to be implemented
    for concrete child classes to work in PorePy.

    Provides and manages component-related AD-variables.

    Instantiated AD variables are provided as properties, as well as the names under which they are stored in the grid data dictionary.

    Physical attributes include constants and scalar functions.
    The latter one is dependent on the thermodynamic state (e.g. pressure, enthalpy),
    which are provided as :class:`pp.ad.MergedVariables` by this class for a given domain (:class:`pp.GridBucket`)

    IMPORTANT:  The doc strings of the abstract properties and methods contain information about
                intended physical dimensions. Keep it consistent when deriving child classes!

    1. Constants
        - Molar Mass
    
    2. Scalars (THD dependent)
        - molar density
    
    """

    def __init__(self, grid_bucket: pp.GridBucket) -> None:
        """ Abstract base class constructor. Initiates component-related AD-variables.
        Contains symbolic names of associated model variables.
        
        :param grid_bucked: domain of computations containing the component
        :type grid_bucket: :class:`porepy.GridBucket`        
        """
        super().__init__()

        # private properties
        self._gb = grid_bucket
        self._mortar_prefix = "mortar"

        self._overall_molar_fraction: pp.ad.MergedVariable
        self._mortar_omf: pp.ad.MergedVariable
        self._omf_var: str = "zeta"
        self._instantiate_overall_molar_fraction()

        self._molar_fraction_in_phase: Dict[str, "pp.ad.MergedVariable"] = dict()
        self._mortar_molar_fraction_in_phase: Dict[str, "pp.ad.MergedVariable"] = dict()
        self._mfip_var: str = "chi"
        self._mfip_names: dict = dict()
        self._mortar_mfip_names: dict = dict()

    @property
    def name(self):
        """
        :return: name of the class. The name is used to construct names for AD variables and keys to store them.
        :rtype: str 
        """
        return str(self.__class__.__name__)

    @property
    def nc(self) -> int:
        """ 
        :return: number of cells in computational domain of instantiation
        :rtype: int
        """
        return self.cd.num_cells()

    @property
    def ideal_gas_constant(self) -> float:
        """ Protected by the property decorator to avoid manipulation by mistake.

        Math. Dimension:        scalar
        Phys. Dimension:        [kg m^2 / s K mol]

        :return: universal molar gas constant
        :rtype: float
        """
        return 8.31446261815324  # NOTE we might consider adding this to pp.utils.common_constants
    
    @property
    def overall_molar_fraction(self) -> pp.ad.MergedVariable:
        """ As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the overall molar fraction of this component
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._overall_molar_fraction

    @property
    def omf_name(self) -> str:
        """
        :return: name of the overall molar fraction variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._omf_var + "_" + self.name

    @property
    def mortar_overall_molar_fraction(self) -> pp.ad.MergedVariable:
        """ As a fractional quantity, all values are between 0 and 1.

        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the overall molar fraction on mortar grids of this component
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._mortar_omf
    
    @property
    def mortar_omf_name(self) -> str:
        """
        :return: name of the overall molar fraction variable on the mortar grids under which it is stored in the grid data dictionaries
        :rtype: str
        """
        return self._mortar_prefix + "_" + self._omf_var + "_" + self.name

    def molar_fraction_in_phase(self, phase_name: str) -> pp.ad.MergedVariable:
        """ As a fractional quantity, all values are between 0 and 1.

        :param phase_name: Name of the  :class:`porepy.composig.Phase` for which the fractions are requested
        :type phase_name: str
        
        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this component in phase `phase_name`.
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._molar_fraction_in_phase[phase_name]

    def mfip_name(self, phase_name: str) -> str:
        """
        :param phase_name: name of the  :class:`porepy.composig.Phase` for which the fraction variable's name is requested
        :type phase_name: str

        :return: name of the molar fraction in phase variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        if phase_name in self._molar_fraction_in_phase.keys():
            return self._mfip_var + "_" + self.name + "_" + phase_name
        else:
            raise ValueError("Component " + self.name + " not in phase " + str(phase_name))

    def mortar_molar_fraction_in_phase(self, phase_name: str) -> pp.ad.MergedVariable:
        """ As a fractional quantity, all values are between 0 and 1.

        :param phase_name: Name of the  :class:`porepy.composig.Phase` for which the fractions are requested
        :type phase_name: str
        
        :return: reference to domain-wide :class:`porepy.ad.MergedVariable` representing the molar fraction of this component in phase `phase_name` on the mortar grids.
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        return self._mortar_molar_fraction_in_phase[phase_name]
    
    def mfip_name(self, phase_name: str) -> str:
        """
        :param phase_name: name of the  :class:`porepy.composig.Phase` for which the fraction variable's name is requested
        :type phase_name: str

        :return: name of the molar fraction in phase variable under which it is stored in the grid data dictionaries
        :rtype: str
        """
        if phase_name in self._mortar_molar_fraction_in_phase.keys():
            return self._mortar_prefix + "_" + self._mfip_var + "_" + self.name + "_" + phase_name
        else:
            raise ValueError("(Mortar) Component " + self.name + " not in phase " + str(phase_name))

    def create_molar_fraction_in_phase(self, phase_name: str) -> Tuple[pp.ad.MergedVariable, pp.ad.MergedVariable]:
        """ Creates an AD variable representing the molar fraction of the component in phase `phase_name`.
        This is a primary variable in composite flow obtained by flash calculations.
        Given phase name will be key for accessing the variable using.

        NOTE (VL): Use this function carefully.
        The supposed usage is only by the Phase class to register itself with this component.
        This solution is not elegant and meant only for prototyping.
        
        :param phase_name: name of the phase
        :type phase_name: str

        :return: reference to the domain-wide variable 
        :rtype: :class:`porepy.ad.MergedVariable`
        """
        
        phase_name = str(phase_name)

        # set name for this global variable
        fraction_name = self._mfip_var + "_" + self.name + "_" + phase_name
        # set the name for the associated mortar variable
        mortar_fraction_name = self._mortar_prefix + "_" + self._mfip_var + "_" + self.name + "_" + phase_name

        # create domain-wide MergedVariable object
        # TODO maths. check if mortar var is needed
        new_mfip_var, new_mortar_mfip_var = create_merged_variable_on_gb(
            self._gb, {"cells": 1}, fraction_name, mortar_fraction_name
        )

        # stor the variables in a private dictionary
        self._molar_fraction_in_phase.update({phase_name: new_mfip_var})
        self._mortar_molar_fraction_in_phase.update({phase_name: new_mortar_mfip_var})

    def set_initial_overall_molar_fraction(self, initial_values: List["np.ndarray"]) -> None:
        """ Order of grid-related initial values in `initial_values` has to be the same as 
        returned by the iterator of the respective :class:`porepy.GridBucket`.
        Keep in mind that the sum over all components of this quantity has to be 1 at all times.
        
        :param initial_values: initial values for the overall molar fractions of this component
        :type initial_values: List[numpy.ndarray]
        """

        # create a copy to avoid issues in case there is another manipulated reference to the values
        vals = [arr.copy() for arr in initial_values]

        for grid_data, vals in zip(self._gb, vals):
            data = grid_data[1]  # get data out (grid, data) tuple
            if pp.STATE not in data:
                data[pp.STATE] = {}
            if pp.ITERATE not in data[pp.STATE]:
                data[pp.STATE][pp.ITERATE] = {}

            data[pp.STATE][self.omf_name] = vals
            data[pp.STATE][pp.ITERATE][self.omf_name] = vals

        # TODO set initial values at interfaces, if applicable (the maths is missing here)
        # check if above code works for GridBuckets without fractures

    def _instantiate_overall_molar_fraction(self) -> None:
        """ Creates the AD representation of the overall molar fraction.
        This is a primary variable in composit flows.
        """
        # adding the overall molar fraction to the primary variables
        # important to notice that they are ADDED, i.e. update the dictionary of prim.vars., don't overwrite it
        # TODO math. check if mortar var is needed
        omf_var, mortar_omf_var = create_merged_variable_on_gb(
            self._gb, {"cells": 1},  self.omf_name, self.mortar_omf_name
        )

        self._overall_molar_fraction = omf_var
        self._mortar_omf = mortar_omf_var


#------------------------------------------------------------------------------
### CONSTANT SCALAR ATTRIBUTES
#------------------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def molar_mass(self) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        [kg / mol]

        :return: molar mass of component (constant)
        :rtype: float
        """
        pass

#------------------------------------------------------------------------------
### SCALAR ATTRIBUTES
#------------------------------------------------------------------------------

    @abc.abstractmethod
    def molar_density(self) -> pp.ad.Function:
        """ 
        Math. Dimension:        scalar
        Phys. Dimension:        [mol / m^3]

        :return: molar density of the component (dependent on thermodynamic state)
        :rtype: :class:`porepy.ad.Function`
        """
        pass

    # @abc.abstractmethod  # TODO decide whether this is necessary or not (next to molar density and molar mass)
    # def mass_density(self, **kwargs) -> pp.ad.Function:
    #     """ 
    #     Math. Dimension:        scalar
    #     Phys. Dimension:        [kg / m^3]
    #
    #     :return: mass density of the component (dependent on thermodynamic state)
    #     :rtype: :class:`pp.ad.Function`
    #     """
    #     pass


class FluidComponent(Component):
    """
    A class extending the list of abstract physical properties with new ones, associated with fluid components.

    The extensive list includes:
        - dynamic_viscosity
    
    """

    @abc.abstractmethod
    def dynamic_viscosity(self) -> pp.ad.Function:
        """ 
        Math. Dimension:        scalar
        Phys. Dimension:        [kg / m s]

        :return: dynamic viscosity of the fluid (dependent on thermodynamic state)
        :rtype: :class:`porepy.ad.Function`
        """
        pass


class SolidSkeletonComponent(Component):
    """
    A class extending the list of abstract physical properties with new ones,
    associated with material for the skeleton of various porous media.

    The extensive list includes:
        - porosity
            - constant
            - pressure related
    
    """

    @abc.abstractmethod
    def base_porosity(self,) -> float:
        """
        Math. Dimension:        scalar
        Phys. Dimension:        dimensionsless, fractional

        :return: base porosity of the material
        :rtype: float
        """
        pass

    def porosity_p(self, reference_pressure: float) -> pp.ad.Function:
        """
        Implements a simple, linear  pressure-based law for porosity:

        \Phi = \Phi_0 * ( p - p_0 )

        :param reference_pressure: a reference pressure value for the linear relation
        :type reference_pressure: float

        :return: Ad object representing the pressure-dependent porosity
        :rtype: :class:`porepy.ad.Function`
        """
        p = 1 # TODO get reference to pressure variable

        return pp.ad.Function(lambda p: self.base_porosity*(p - reference_pressure),
        "Rel. perm. liquid")(p)