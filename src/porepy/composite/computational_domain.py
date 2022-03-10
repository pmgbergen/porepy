""" Contains the physical extension for :class:`~porepy.grids.grid_bucket.GridBucket`."""
from __future__ import annotations

import porepy as pp
import numpy as np
import warnings

from porepy.composite.material_subdomain import MaterialSubdomain

from ._composite_utils import (
    COMPUTATIONAL_VARIABLES,
    create_merged_variable,
    create_merged_mortar_variable
)

from typing import (Optional, Dict, Set, Iterator, List, Union, Tuple,
TYPE_CHECKING)
# this solution avoids circular imports due to type checking. Needs __future__.annotations
if TYPE_CHECKING:
    from .phase import Phase
    from .substance import SolidSubstance


class ComputationalDomain:
    """ Physical extension of :class:`~porepy.GridBucket`.

    Constructs computational elements for the simulation,
    by combining the geometrical functions provided by a :class:`~porepy.grids.grid_bucket.GridBucket`.

    Currently, this class is not concerned with the physical properties of the domain. 
    Those are split on each subdomain (see :class:`~porepy.composite.material_subdomain.MaterialSubDomain`),
    due to how AD variables such as the mass matrix are constructed (they access grid-wise the parameters, which need to be set in te model)
    
    Merged physical properties are constructed based on the subdomain properties.

    Merged AD variables are constructed, stored and accessed, based on the subdomain AD variables.
    """

    def __init__(self, gridbucket: pp.GridBucket) -> None:
        """ Constructor.

        :param gridbucket: geometrical representation of domain
        :type gridbucket: :class:`~porepy.grids.grid_bucket.GridBucket`
        """
        # public
        self.gb: pp.GridBucket = gridbucket

        # keys: symbolic variable names, values: respective MergedVariable
        self._global_ad: dict = dict()
        self._phase_names: set = set()
        self._substance_names: set = set()
        # key: phase name, value: set with present substance names
        self._substance_in_phase: Dict[str, Set[str]] = dict()
        # instances of added phases 
        self._phases: list = list()
        # key: grid, value: MaterialSubdomain
        self._material_subdomains: dict = dict()

        for grid, _ in self.gb:
            #TODO: discuss whether None or UnitSolid should be used to create a MaterialSubdomain
            # the first approach needs further programmatical assertions
            # the latter will keep the model "runable" without the user explicitely stating that it should be like that.
            self._material_subdomains.update({grid: pp.composite.UnitSolid(self)}) 

    def __str__(self) -> str:
        """ Returns string representation of instance,
        with information about invoked variables and concatanated string representation of underlying gridbucket.
        """

        out = "Computational domain with " + str(len(self._global_ad)) + " AD variables:\n"

        for var_name in self._global_ad.keys():
            out += var_name + "\n"

        out += "\non\n"

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
                var, _ = create_merged_variable(self.gb, dof_info, variable_name)
            
            self._global_ad.update({variable_name: var})

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
        return len(self._phase_names)

    @property
    def ns(self) -> int:
        """
        :return: total number of distinct substances in all phases
        :rtype: int
        """
        return len(self._substance_names)
    
    @property
    def Phases(self) -> Tuple[Phase]:
        """
        IMPORTANT: The order in this iterator (tuple) is used for choosing e.g. the values in a list of 'numpy.array' when setting initial values.
        Use the order returns here everytime you deal with phase-related values or other.
        
        :return: returns the phases created and added to this domain.
        :rtype: tuple
        """
        return (phase for phase in self._phases)

    def assign_material_to_grid(self, grid: pp.Grid, substance: SolidSubstance) -> None:
        """
        Assigns a material to a grid i.e., creates an instance of :class:`~porepy.composite.material_subdomain.MaterialSubdomain`
        
        Currently, one has to assign substances to each grid.
        Domain-wide objects like domain porosity will throw an error if this is missing.
        More elegant solutions should be found in the future.

        Use the iterator of :class:`~porepy.grids.grid_bucket.GridBucket` to assign substances to each grid.
        That iterator is used as a base for this class' iterator.

        """
        if grid in self.gb.get_grids():
            pass
        else:
            raise KeyError("Argument 'grid' not among grids of GridBucket.")

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

    def add_phase(self, phases: Union[List[Phase], Phase]) -> None:
        """
        Adds the phases to the compositional domain.
        Asserts uniqueness of present phases via :method:`~porepy.composite.phase.Phase.name` and object comparison.
        Asserts unitarity of initial saturations
        Calls an internal method to resolve the compositions and store relevant information and compute initial overall molar fractions based on the saturation.

        NOTE: Overwrites previously passed phases and removes their
        
        
        :param phases: a phase instance to be added or multiple phase instances in a list.
        :type phases: :class:`~porepy.composite.phase.Phase`
        """

        if isinstance(phases, Phase):
            phases = [phases]

        old_names = {phase.name for phase in self._phases}
        # check if phase is instantiated on same domain or if it's name is already among the present phases
        for phase in phases:
            if phase.cd != self: 
                raise ValueError("Phase '%s' instantiated on unknown ComputationalDomain."%(phase.name))
            
            if phase.name in old_names:
                warnings.warn("Phase '%s' has already been added. Skipping..."%(phase.name))
            else:
                self._phases.append(phase)

        self.resolve_composition()

    def set_initial_values(self,
    pressure: Union[float, np.array],
    temperature: Union[float, np.array],
    saturations: Union[List[float], List[np.array]],
    molar_fractions_in_phase: Dict[str, Union[float, np.array]],
    compute_equilibrium: Optional[bool] = True
    ) -> None:
        """ Sets the initial compositional and thermodynamic state of the system.
        Natural variables are used as arguments, since the approach using them is relatable.

        Enthalpy is computed using an isenthalpic flash.

        Initial values of molar variables are computed using above values.
        (see :method:`~porepy.composite.substance.Substance.overall_molar_fraction` and
        :method:`~porepy.composite.phase.Phase.molar_fraction`).
        
        If 'compute_equilibrium' is True, the equilibrium equations are iterated until initial equilibrium is reached.
        NOTE: This needs some investigations, as omitting this might influence the stability of the solver.
        
        Each variable can either be given homogenously (float per variable)
        or heterogeneously (float per cell).
        Floats will be used for arrays (values per cell) and unitary conditions
        for the fractional variables will be checked either way.
        IMPORTANT: If initial unitary conditions are not met, an error will be thrown.

        :param saturations: saturation values per cell (arrays, if heterogeneous)
        :type saturations: List[float] / List[numpy.array]
        """
        
        self._calculate_initial_phase_molar_fractions()
        self._calculate_initial_component_overall_fractions()

    def resolve_composition(self) -> None:
        """
        Analyzes the composition, i.e. presence of substances in phases.
        Information about substances which are anticipated in multiple phases is stored.

        IMPORTANT: This method is called internally by phases and domains, everytime any new component is added.

        """
        pass

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

    # def _compatibility_check(to_check: List[MaterialSubdomain]) -> Union[None, str]:
    #     """ Checks if all elements in 'to_check' are compatible with this extension.
        
    #     :param to_check: iterable containing all objects to be checked
    #     :type to_check: iterable

    #     :return: concetaneted str representation of each element of 'to_check', if incompatibility detected. None otherwise
    #     :rtype: str/None
    #     """
    #     incompatible = list()

    #     for subdomain in to_check:
    #         if not isinstance(subdomain, MaterialSubdomain):
    #             incompatible.append(subdomain)

    #     if incompatible:
    #         err_msg = ""
    #         for inc in incompatible:
    #             err_msg += str(inc) + "\n"
    #     else:
    #         err_msg = None

    #     return err_msg

    # def add_nodes(self, new_subdomains: Union[MaterialSubdomain, Iterable[MaterialSubdomain]]) -> None:
    #     """ See documentation of parent method :method:`~porepy.GridBucket.add_nodes`.

    #     This child classes assures only that arguments are of type :class:`~porepy.domain.SubDomain` in order for the
    #     extended functionality to work.
    #     """

    #     if not isinstance(new_subdomains, list):
    #         new_subdomains = [new_subdomains]
        
    #     check = self._compatibility_check(new_subdomains)
    #     if check:
    #         raise TypeError("Following subdomains expected to be of type 'SubDomain':\n" + check)

    #     return super().add_nodes(new_subdomains)

    # def add_edge(self, subdomains: List[MaterialSubdomain], primary_secondary_map: sps.spmatrix) -> None:
    #     """ See documentation of parent method :method:`~porepy.GridBucket.add_edge`.

    #     This child classes assures only that arguments are of type :class:`~porepy.domain.SubDomain` in order for the
    #     extended functionality to work.
    #     """

    #     check = self._compatibility_check(subdomains)
    #     if check:
    #         raise TypeError("Following subdomains expected to be of type 'SubDomain':\n" + check)

    #     return super().add_edge(subdomains, primary_secondary_map)