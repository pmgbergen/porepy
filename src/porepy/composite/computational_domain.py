""" Contains the physical extension for :class:`~porepy.grids.grid_bucket.GridBucket`."""
from __future__ import annotations

import porepy as pp
import numpy as np

from ._composite_utils import (
    COMPUTATIONAL_VARIABLES,
    create_merged_variable,
    create_merged_mortar_variable
)

from typing import (Optional, Dict, Set, Iterator, Iterable, Tuple, 
TYPE_CHECKING)
# this solution avoids circular imports due to type checking. Needs __future__.annotations
if TYPE_CHECKING:
    from .phase import Phase


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
        self._unique_phase_names: set = set()
        self._unique_substance_names: set = set()
        # key: phase name, value: set with present substance names
        self._substance_in_phase: Dict[str, Set[str]] = dict()
        self._phases: list = list()

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
            split_name = variable_name.split("-")
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

        return var

    def __iter__(self) -> Iterator[Phase]:
        """ Returns an Iterator over all present phases."""
        for phase in self._phases:
            yield phase
    
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
        return len(self._unique_phase_names)

    @property
    def ns(self) -> int:
        """
        :return: total number of distinct substances in all phases
        :rtype: int
        """
        return len(self._unique_substance_names)
    
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

    def add_phase(self, phases: Iterable[Tuple[Phase, np.ndarray]]) -> None:
        """
        Adds the phases to the compositional domain.
        Asserts uniqueness of present phases via :method:`~porepy.composite.phase.Phase.name` and object comparison.
        Asserts unitarity of initial saturations
        Calls an internal method to resolve the compositions and store relevant information and compute initial overall molar fractions based on the saturation.

        NOTE: Overwrites previously passed phases and removes their
        
        
        :param phases: Iterable of 2-tuples, containing the phase instance and it's respective initial saturation.
        :type phases: Iterable[Tuple[Phase, np.ndarray]]
        """
        self._resolve_composition()


    def _resolve_composition(self) -> None:
        """
        Analyzes the composition, i.e. presence of substances in phases.
        Information about substances which are anticipated in multiple phases is stored.
        Computes initial overall molar fractions per component
        (see :method:`~porepy.composite.substance.Substance.overall_molar_fraction`).
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