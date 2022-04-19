""" Contains the physical extension for :class:`~porepy.grids.grid_bucket.GridBucket`."""
from __future__ import annotations

import porepy as pp
import numpy as np
import warnings

from porepy.composite.material_subdomain import MaterialSubdomain
from porepy.composite.substance import Substance

from ._composite_utils import (
    COMPUTATIONAL_VARIABLES,
    create_merged_variable,
    create_merged_mortar_variable
)

from typing import (Optional, Dict, Set, Iterator, List, Union, Tuple,
TYPE_CHECKING)
# this solution avoids circular imports due to type checking. Needs __future__.annotations
if TYPE_CHECKING:
    from .phase import PhaseField
    from .substance import SolidSubstance, Substance


class CompositionalDomain:
    """ Physical extension of :class:`~porepy.GridBucket`.
    NOTE: alternative name could be PhysicalDomain

    Constructs computational elements for the simulation, by
        - combining the geometrical functions provided by a :class:`~porepy.grids.grid_bucket.GridBucket`
        - providing calls to global, physical variables (merged variables from the AD submodule)
            - serves as a central point for creating, storing and accessing those variables

    The physical properties of the domain are assembled using a material representations of each grid in the gridbucket,
    namely :class:`~porepy.composite.material_subdomain.MaterialSubDomain`.
    Currently they are accessed grid-wise in the model-classes.

    Public attributes:
        - 'gb': GridBucket for which the CompositionalDomain was initiated
        - 'INITIAL_EQUILIBRIUM': bool. Flagged false if composition changes.
                                 Only flagged True after initial equilibrium is computed, i.e. after initial variables were set.
                                 DO NOT flag this bool yourself as you please.
        - 'pressure': MergedVariable representing the global pressure
        - 'enthalpy': MergedVariable representing the global enthalpy

    """

    def __init__(self, gridbucket: pp.GridBucket) -> None:
        """ Constructor.

        :param gridbucket: geometrical representation of domain
        :type gridbucket: :class:`~porepy.grids.grid_bucket.GridBucket`

        Instantiate default material subdomains using the unit solid class.
        NOTE this approach should be discussed. One could also instantiate None and demand certain steps by the modeler
        The current solution keeps the model 'runable' without the modeler explicitely setting material properties for the grids.
        """
        ## PRIVATE
        # keys: symbolic variable names, values: respective MergedVariable
        self._global_ad: Dict[str, "pp.ad.MergedVariable"] = dict()
        # set containing all present substances
        self._present_substances: Set[Substance] = set()
        # key: phase name, value: tuple of present substance names
        self._phases_per_substance: Dict[Substance, Set[PhaseField]] = dict()
        # instances of added phases 
        self._anticipated_phases: List[PhaseField] = list()
        # key: grid, value: MaterialSubdomain
        self._material_subdomains: Dict["pp.Grid", MaterialSubdomain] = dict()

        # initiate system-wide primary variables
        self._pressure_var: str = COMPUTATIONAL_VARIABLES["pressure"]
        self._enthalpy_var: str = COMPUTATIONAL_VARIABLES["enthalpy"]

        for grid, _ in self.gb:
            self._material_subdomains.update({grid: MaterialSubdomain(grid, pp.composite.UnitSolid(self))}) 

        ## PUBLIC
        self.gb: pp.GridBucket = gridbucket
        self.INITIAL_EQUILIBRIUM: bool = False
        self.pressure: pp.ad.MergedVariable = self(self._pressure_var)
        self.enthalpy: pp.ad.MergedVariable = self(self._enthalpy_var)

    def __str__(self) -> str:
        """ Returns string representation of instance,
        with information about invoked variables and phases.
        Concatenates the string representation of the underlying gridbucket.
        """

        out = "Computational domain with " + str(len(self._global_ad)) + " AD variables:\n"

        for var_name in self._global_ad.keys():
            out += var_name + "\n"

        out += "\nand %s phases:\n"%(str(len(self._anticipated_phases)))

        for phase_name in [phase.name for phase in self._anticipated_phases]:
            out += phase_name + "\n"

        out += "\non gridbucket \n"

        return out + str(self.gb)

    def __call__(self, variable_name: str, dof_info: Optional[Dict[str, int]] = {"cells":1}) -> pp.ad.MergedVariable:
        """ Returns a reference to the unique, domain wide variable with a distinguishable name.
        Accessing the same variable is of importance in inherited models and multi-physics scenarios.

        If the name is unused so far, a new variable will be constructed.
        NOTE: this approach needs discussion, could be ambiguous and/or arbitrary.
        But we want this to be the single point of access (for central storage)

        :param variable_name: symbolic variable name (see :data:`~porepy.params.computational_variables.COMPUTATIONAL_VARIABLES`)
        :type variable_name: str
        :param dof_info: (optional) number of DOFs per grid element (e.g. cells, faces, nodes). Defaults to 1 DOF per cell
        :type dof_info: dict

        :return: A AD representation of the domain-wide variable
        :rtype: :class:`~porepy.ad.operators.MergedVariable`
        """
        if variable_name in self._global_ad.keys():
            #TODO case when variable exists AND new DOF information is given (dereference old var and create new one)
            var = self._global_ad[variable_name]
        else:
            split_name = variable_name.split("_")
            # case: variable on the mortar grids
            if COMPUTATIONAL_VARIABLES["mortar_prefix"] == split_name[0]:
                symbol = split_name[1]
                is_mortar = True
            # case: variable on subdomains
            else:
                symbol = split_name[0]
                is_mortar = False

            # NOTE VL: Think about introducing some kind of validation for the symbol, to avoid arbitrary creation of variables

            if is_mortar:
                var = create_merged_mortar_variable(self.gb, dof_info, variable_name)
            else:
                var = create_merged_variable(self.gb, dof_info, variable_name)
            
            self._global_ad.update({variable_name: var})
            # update DOFs since a new variable has been created

        return var

    def __iter__(self) -> Iterator[Tuple[pp.Grid, dict, MaterialSubdomain]]:
        """
        Returns an Iterator over all grids of this domain.
        Similar to the iterator of :class:`~porepy.grids.grid_bucket.GridBucket`,
        only here the respective MaterialDomain is added as a third component in the yielded tuple.
        """
        for grid, data in self.gb:
            yield (grid, data, self._material_subdomains[grid])
    
    @property
    def nc(self) -> int:
        """ 
        :return: number of cells in grid bucket of instantiation
        :rtype: int
        """
        return self.gb.num_cells()

    @property
    def np(self) -> int:
        """
        :return: number of added phases
        :rtype: int
        """
        return len(self._anticipated_phases)

    @property
    def ns(self) -> int:
        """
        :return: total number of distinct substances in all phases
        :rtype: int
        """
        return len(self._present_substances)
    
    @property
    def Phases(self) -> Tuple[PhaseField]:
        """
        IMPORTANT: The order in this iterator (tuple) is used for choosing e.g. the values in a list of 'numpy.array' when setting initial values.
        Use the order returns here everytime you deal with phase-related values or other.
        
        :return: returns the phases created and added to this domain.
        :rtype: tuple
        """
        return (phase for phase in self._anticipated_phases)

    def assign_material_to_grid(self, grid: pp.Grid, substance: SolidSubstance) -> None:
        """
        Assigns a material to a grid i.e., creates an instance of :class:`~porepy.composite.material_subdomain.MaterialSubdomain`
        Replaces the default material subdomain instantiated in the constructor using the :class:`~porepy.composite.unit_substances.UnitSolid`.

        You can use the iterator of this instances :class:`~porepy.grids.grid_bucket.GridBucket` to assign substances to grids.

        :param grid: a sub grid present in the gridbucket passed at instantiation
        :type grid: :class:`~porepy.grids.grid.Grid`

        :param substance: the substance to be associated with the subdomain
        :type substance: :class:`~porepy.composite.substance.SolidSubstance`
        """
        if grid in self.gb.get_grids():
            self._material_subdomains.update({grid: MaterialSubdomain(grid, substance)}) 
        else:
            raise KeyError("Argument 'grid' not among grids in GridBucket.")

    def is_variable(self, var_name: str) -> bool:
        """
        :param var_name: name of the variable you want to check for existence in this domain
        :type var_name: str

        :return: True, if variable has already been instantiated, False otherwise.
        :rtype: bool
        """
        if var_name in self._global_ad.keys():
            return True
        else:
            return False

    def add_phase(self, phases: Union[List[PhaseField], PhaseField]) -> None:
        """
        Adds the phases to the compositional domain.

        Resolves the composition of the flow (which substance appears in which phase).

        Skips phases which were already added.

        The phases must be instantiated on the this
        :class:`~porepy.composite.computational_domain.ComputationalDomain` instance.

        
        :param phases: a phase instance to be added or multiple phase instances in a list.
        :type phases: :class:`~porepy.composite.phase.Phase`
        """

        if isinstance(phases, PhaseField):
            phases = [phases]

        old_names = {phase.name for phase in self._anticipated_phases}
        # check if phase is instantiated on same domain or if it's name is already among the present phases
        for phase in phases:
            if phase.cd != self: 
                raise ValueError("Phase '%s' instantiated on unknown ComputationalDomain."%(phase.name))
            
            if phase.name in old_names:
                warnings.warn("Phase '%s' has already been added. Skipping..."%(phase.name))
                continue
            else:
                self._anticipated_phases.append(phase)

        self.resolve_composition()

    def set_initial_values(self,
    pressure: Union[List[float], List[np.array]],
    temperature: Union[List[float], List[np.array]],
    saturations: List[Union[List[float], List[np.array]]]
    ) -> None:
        """
        Sets the initial compositional and thermodynamic state of the system.
        Natural variables are used as arguments, since they are more relatable in application.

        Enthalpy is computed using an isenthalpic flash.

        THE FOLLOWING IS ASSUMED FOR THE ARGUMENTS:
            - the top list contains lists/arrays/floats per grid in gridbucket
            - the lists per grid contain arrays/floats per phase
            - the nesting List[List[]] applies only to saturation values
            - the order of the gridbuckets iterator is used for the list-entries per grid
            - the order of this instances iterator :method:`~porepy.composite.computational_domain.Phases` is assumed for the values in the nested lists
            - each variable is either given homogenously (float per variable) or heterogeneously (float per cell, i.e. array per grid)

        Finally, this methods asserts the initial unitarity of the saturation values per cell.

        :param pressure: initial pressure values per grid
        :type pressure: list
        :param temperature: initial temperature per grid
        :type temperature: list
        :param saturations: saturation values per grid per anticipated phase
        :type saturations: list
        """
        
        for idx, grid_data in enumerate(self.gb):
            
            grid, data = grid_data

            ## check if data dictionary has necessary keys
            # If not, create them. TODO check if we should also prepare the 'previous_timestep' values here
            if pp.STATE not in data:
                data[pp.STATE] = {}
            if pp.ITERATE not in data[pp.STATE]:
                data[pp.STATE][pp.ITERATE] = {}

            ## setting initial pressure values for this grid
            vals = pressure[idx]
            # convert homogenous fractions to values per cell
            if isinstance(vals, float):
                vals = np.ones(grid.num_cells) * vals
            data[pp.STATE][self._pressure_var] = np.copy(vals)
            data[pp.STATE][pp.ITERATE][self._pressure_var] = np.copy(vals)  

            ## setting initial enthalpy values for this grid
            vals_t = temperature[idx]
            # convert homogenous fractions to values per cell
            if isinstance(vals, float):
                vals_t = np.ones(grid.num_cells) * vals_t

            # TODO convert TEMPERATURE vals to ENTHALPY vals
            
            data[pp.STATE][self._enthalpy_var] = np.copy(vals_t)
            data[pp.STATE][pp.ITERATE][self._enthalpy_var] = np.copy(vals_t)

            ## setting initial saturation values for this grid
            # assertions of unitarity of saturation per grid (per cell actually)
            sum_saturation_per_grid = np.zeros(grid.num_cells)

            # loop over next level: fractions per phase (per grid)
            # this throws an error if there are values missing for a phase (or if there are too many)
            for phase, values in zip(self.Phases, saturations[idx]):
            
                # convert homogenous fractions to values per cell
                if isinstance(values, float):
                    values = np.ones(grid.num_cells) * values

                # this throws an error if the dimensions should mismatch when giving fractions for a grid in array form
                sum_saturation_per_grid += values


                data[pp.STATE][phase.saturation_var] = np.copy(values)
                data[pp.STATE][pp.ITERATE][phase.saturation_var] = np.copy(values)  

            # assert the fractional character (sum equals 1) in each cell
            if np.any(sum_saturation_per_grid != 1.): #TODO check sensitivity
                raise ValueError("Initial saturations do not sum up to 1. on each cell on grid:\n" + str(grid))

    def phases_of_substance(self, substance: Substance) -> Tuple[PhaseField]:
        """
        :return: for given substance, a tuple of phases is returned which contain this substance.
        :rtype: tuple
        """
        return self._phases_per_substance[substance]

    def resolve_composition(self) -> None:
        """
        Analyzes the composition, i.e. presence of substances in phases.
        Information about substances which are anticipated in multiple phases is stored.

        This method is called internally by phases and domains, everytime any new component is added.
        """
        # for given substance (keys), save set of phases containing the substance (values)
        phases_per_substance: Dict[Substance, Set[PhaseField]] = dict()
        # unique substance names. Independent of number of phases in which a substance is anticipated, the name appears here only once.
        unique_substances: Set[Substance] = set()

        for phase in self.Phases:
            for substance in phase:
                unique_substances.add(substance)

                if substance in phases_per_substance.keys():
                    phases_per_substance[substance].add(phase)
                else:
                    phases_per_substance.update({substance: set()})
                    phases_per_substance[substance].add(phase)
            
        
        self._present_substances = unique_substances
        self._phases_per_substance = phases_per_substance

        self.INITIAL_EQUILIBRIUM = False

    def compute_initial_equilibrium(self) -> None:
        """
        Computes the initial equilibrium using the initial values.
        Throws an error if not all initial values have been set.

        Initial values of molar variables are computed using the natural variables.
        (see :method:`~porepy.composite.substance.Substance.overall_molar_fraction` and
        :method:`~porepy.composite.phase.Phase.molar_fraction`).

        Use :method:`~porepy.compostie.phase.PhaseField.set_initial_fractions_in_phase` to set initial molar fractions per phase
        Use :method:`~porepy.compostie.compositional_domain.CompositionalDomain.set_initial_values` to set the rest.

        If successful, flags this instance by setting :data:`~~porepy.compostie.compositional_domain.CompositionalDomain.INITIAL_EQUILIBRIUM` true.

        NOTE: this approach of setting initial values and computing the equilibrium is a first.
        The reason for splitting it between the PhaseField class and this class is for reasons of easier overview,
        and to give more responsibility to the PhaseField.
        """

        self._calculate_initial_phase_molar_fractions()
        self._calculate_initial_component_overall_fractions()

        # TODO compute equilibrium

        self.INITIAL_EQUILIBRIUM = False

    def _calculate_initial_phase_molar_fractions(self) -> None:
        """ 
        Name is self-explanatory.

        These calculations have to be done everytime everytime new initial values are set.
        """
        pass

    def _calculate_initial_component_overall_fractions(self) -> None:
        """ 
        Name is self-explanatory.
        
        These calculations have to be done everytime everytime new initial values are set.
        """
        pass
    
    def _set_initial_overall_molar_fraction(self, initial_values: List["np.ndarray"]) -> None:
        """ Order of grid-related initial values in `initial_values` has to be the same as 
        returned by the iterator of the respective :class:`porepy.GridBucket`.
        Keep in mind that the sum over all components of this quantity has to be 1 at all times.
        
        :param initial_values: initial values for the overall molar fractions of this component
        :type initial_values: List[numpy.ndarray]
        """

        for grid_data, val in zip(self.gb, initial_values):
            data = grid_data[1]  # get data out (grid, data) tuple
            if pp.STATE not in data:
                data[pp.STATE] = {}
            if pp.ITERATE not in data[pp.STATE]:
                data[pp.STATE][pp.ITERATE] = {}

            # create a copy to avoid issues in case there is another manipulated reference to the values
            data[pp.STATE][self.omf_var] = np.copy(val)
            data[pp.STATE][pp.ITERATE][self.omf_var] = np.copy(val)