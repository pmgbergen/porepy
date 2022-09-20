"""Contains a central access class for domain-wide primary and secondary variables."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import porepy as pp

__all__ = ["VariableStore"]


class VariableStore:
    """The Variable-Store provides symbolic access to (physical) quantities as primary and
    secondary variables for a given computational domain.
    
    Accessed quantities are stored and included in the global order of DOFs. Hence this class
    also provides an own :class:`~porepy.DofManager` to be used in a multi-physics setting.
    Once all relevant quantities where invoked, this class and its DOF manager can be used
    with the AD framework to create an :class:`~porepy.ad.EquationManager` and to start
    modelling the system.
    
    Parameters:
        mdg: mixed-dimensional grid representing the computational domain.
    """

    admissible_dof_types: set[str] = {"cells", "faces", "nodes"}
    """A set denoting admissible types of DOFs for variables.
    
    - nodes: DOFs per node, which constitute the grid
    - cells: DOFs per cell (center), which are defined by nodes
    - faces: DOFS per face, which form the (polygonal) boundary of cells
    """

    symbol_map: dict[str, str] = {
        "mortar": "mortar",  # prefix for symbols for variables which appear on mortar grids
        "pressure": "p",  # [Pa] = [N / m^2] = [ kg / m / s^2]
        "enthalpy": "h",  # (specific, molar) [J / mol] = [kg m^2 / s^2 / mol]
        "temperature": "T",  # [K]
        "displacement": "u",  # [m]
        "component_fraction": "z",  # (fractional, molar) [-]
        "component_fraction_in_phase": "x",  # (fractional, molar) [-]
        "molar_phase_fraction": "y",  # (fractional, molar) [-]
        "saturation": "s",  # (fractional, volumetric) [-]
    }
    """Maps common quantity names to usual symbols e.g., 'pressure' to 'p'.
    
    The intended use for symbols is to use them when storing numerical values in data dicts,
    in an attempt to unify the notation for larger problems.
    """
    
    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        
        ### PUBLIC
        self.mdg: pp.MixedDimensionalGrid = mdg
        """The mixed-dimensional, computational domain passed at instantiation."""

        ### PRIVATE
        self._nc: int = mdg.num_subdomain_cells()
        self._dm: pp.DofManager = pp.DofManager(mdg)

        # internal reference to already created vars. The given name is used as key.
        self._created_vars: dict[str, pp.ad.MergedVariable] = dict()

    @property
    def dof_manager(self) -> pp.DofManager:
        """Reference to the DOF manager for the given domain."""
        return self._dm

    @property
    def variables(self) -> dict[str, pp.ad.MergedVariable]:
        """A dictionary containing all variables created."""
        return self._created_vars

    def create_variable(
        self,
        name: str,
        primary: bool,
        dof_info: dict[str, int] = {"cells": 1},
        subdomains: Union[None, list[pp.Grid]] = list(),
        interfaces: Optional[list[pp.MortarGrid]] = None,
    ) -> pp.ad.MergedVariable:
        """Creates a new variable according to specifications.
        
        Parameters:
            name: used here as an identifier. Can be used to associate the variable with some
                physical quantity.
            primary: indicator if primary variable. Otherwise it is a secondary variable and 
                will not contribute to the global DOFs
            dof_info: dictionary containing information about number of DOFs per admissible
                type (see :data:`admissible_dof_types`). Defaults to ``{'cells':1}``.
            subdomains: List of subdomains on which the variable is defined. If None, then it
                will not be defined on any subdomain. If the list is empty, it will be defined
                on **all** subdomains. Defaults to empty list.
            interfaces: List of interfaces on which the variable is defined. If None, then it
                will not be defined on any interface. Here an empty list is equally treated as 
                None. Defaults to none.

        Returns:
            a merged variable with above specifications.    
        """
        # sanity check for admissible DOF types
        requested_type = set(dof_info.keys())
        if not requested_type.issubset(self.admissible_dof_types):
            non_admissible = requested_type.difference(self.admissible_dof_types)
            raise ValueError(f"Non-admissible DOF types {non_admissible} requested.")
        # sanity check if variable is defined anywhere
        if subdomains is None and interfaces is None:
            raise ValueError("Cannot create variable not defined on any grid or interface.")
        # check if variable was already defined
        if name in self._created_vars.keys():
            raise KeyError(f"Variable with name '{name}' already created.")

        # if an empty list was received, we use ALL subdomains
        if isinstance(subdomains, list) and len(subdomains) == 0:
            subdomains = [sg for sg in self.mdg.subdomains()]

        variables = list()

        if primary:
            variable_category = pp.PRIMARY_VARIABLES
        else:
            variable_category = pp.SECONDARY_VARIABLES

        if isinstance(subdomains, list):
            for sd in subdomains:
                data = self.mdg.subdomain_data(sd)

                # prepare data dictionary if this was not done already
                if variable_category not in data:
                    data[variable_category] = dict()
                if pp.STATE not in data:
                    data[pp.STATE] = dict()
                if pp.ITERATE not in data[pp.STATE]:
                    data[pp.STATE][pp.ITERATE] = dict()

                data[variable_category].update({name: dof_info})

                # create grid-specific variable
                variables.append(pp.ad.Variable(name, dof_info, subdomains=[sd]))
        
        if isinstance(interfaces, list):
            for intf in interfaces:
                data = self.mdg.interface_data(intf)
            
                if intf.codim == 2:  # no variables in points TODO check if this up-to-date
                    continue
                else:
                    # prepare data dictionary if this was not done already
                    if variable_category not in data:
                        data[variable_category] = dict()
                    if pp.STATE not in data:
                        data[pp.STATE] = dict()
                    if pp.ITERATE not in data[pp.STATE]:
                        data[pp.STATE][pp.ITERATE] = dict()
                    
                    # store DOF information about variable
                    data[variable_category].update({name: dof_info})

                    # create mortar grid variable
                    variables.append(
                        pp.ad.Variable(name, dof_info, interfaces=[intf], num_cells=self._nc)
                    )

        # create and store the merged variable
        merged_var = pp.ad.MergedVariable(variables)
        self._created_vars.update({name: merged_var})

        # append the new DOFs to the global system if a primary variable has been created
        if primary:
            self.dof_manager.append_dofs([name])

        return merged_var

    def set_values(
        self, var_name: str, values: np.ndarray, copy_to_state: bool = False
    ) -> None:
        """Sets values for a given variable name in the grid data dictionaries.

        It is assumed the variable (name) is known to this instance.
        This is a shallow wrapper for respective functionalities of the DOF manager.
        The values are set for the ITERATE, additionally to the STATE if flagged.

        Parameters:
            var_name: name of the :class:`~porepy.ad.MergedVariable` for which the STATE should
                be set.
            values: respective values. It is assumed the values are provided such that they fit
                in the global DOF vector with the variable's respective DOF indexes.
            copy_to_state: copies the values additionally to the STATE.
        
        Raises:
            KeyError: if the variable name is not known to this instance.
        """
        if var_name not in self._created_vars.keys():
            raise KeyError(f"Unknown variable '{var_name}'.")
        
        variable = [var_name]

        # inserting the values in a global-sized zero vector.
        X = np.zeros(self.dof_manager.num_dofs())
        dof = self.dof_manager.dof_var(variable)
        X[dof] = values
        # setting ITERATE and optionally STATE
        self.dof_manager.distribute_variable(X, variables=variable, to_iterate=True)
        if copy_to_state:
            self.dof_manager.distribute_variable(X, variables=variable)
