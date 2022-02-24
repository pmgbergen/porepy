""" Utility functions for the composite submodule.

Functions:
    - create_merged_variable_on_gb

"""

from typing import Tuple, Union, Optional, Dict

import porepy as pp


def create_merged_variable_on_gb(
    gb: pp.GridBucket, dof_info: Dict[str, int],
    variable_name: str, mortar_variable_name: Optional[Union[str, None]] = None
    )-> Tuple[pp.ad.MergedVariable, Union[None, pp.ad.MergedVariable]]:
    """ Creates domain-wide merged variables for a given grid bucket.
    Stores information about the variable in the data dictionary associated with each subdomain. 
    The key :data:`porepy.PRIMARY_VARIABLES` and given variable names are used for storage.

    For creating a variable without values on mortar grids, leave the argument `mortar_variable_name` as `None`.
    
    :param gb: grid bucket representing the whole computational domain
    :type gb: :class:`porepy.GridBucket`
    :param dof_info: number of DOFs per grid element (e.g. cell, face)
    :type dof_info: dict
    :param variable_name: name given to variable, used as keyword for storage
    :type variable_name: str
    :param mortar_variable_name: (optional) name given to respective mortar variable, used as keyword for storage. If none, no mortar variable will be assigned.
    :type mortar_variable_name: str

    :return: Returns a 2-tuple containing the new objects. The second object will be None, if no mortar variable name is specified.
    :rtype: tuple(2)
    """

    # creating variables on each subdomain
    variables = list()
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES].update({variable_name: dof_info})
        # TODO test if the order of variables is alright
        variables.append(pp.ad.Variable(
            variable_name, dof_info, 
            grids=[g])) 

    # creating variables on each mortar grid, if requested
    if mortar_variable_name is not None:
        mortar_variables = list()
        for e, d in gb.edges():
            if d["mortar_grid"].codim == 2:  # no variables in points
                continue
            else:
                d[pp.PRIMARY_VARIABLES].update({mortar_variable_name: dof_info})
                # TODO test if the order of variables is alright
                mortar_variables.append(pp.ad.Variable(
                    mortar_variable_name, dof_info,
                    edges=[e], num_cells=d["mortar_grid"].num_cells))

        return (pp.ad.MergedVariable(variables), pp.ad.MergedVariable(mortar_variables))
    else:
        return (pp.ad.MergedVariable(variables), None)