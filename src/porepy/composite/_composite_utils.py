""" Utility functions for the composite submodule.

Data:
    - COMPUTATIONAL_VARIABLES

Functions:
    - create_merged_variable_on_gb

"""

from typing import Tuple, Union, Optional, Dict

import porepy as pp



""" Contains an overview of computational variables.

The dictionary :data:`~porepy.params.computational_variables.COMPUTATIONAL_VARIABLES` contains strings as keys and 2-tuples as values.

The key is a word describing the physical variable.
They are used for accessing the variable using :class:`~pp.composite.material_subdomain.MaterialSubDomain` and
:class: `~porepy.compostie.computational_domain.ComputationalDomain`.

The value is a string, representing the mathematical symbol,

Above information is used to construct AD variables and to store respective information in the grid data structure.
(e.g. if pressure has the symbol 'p', the relevant information will be found in 'data[pp.PRIMARY_VARIABLES]["p"]')

IMPORTANT: For more complex components, the symbol string will be augmented.
E.g. if component molar fractions in phase have the symbol 'chi', the final name for a specific component in a specific 
phase will be 'chi_<component name>_<phase name>'.
EXAMPLE: the molar fraction of water in vapor form can have the name
    xi_H2O_vapor
if the modeler decides to call the substance class 'H20' and the phase 'vapor'

Assumed, default SI units of the respective variables are found below in comments.
"""
COMPUTATIONAL_VARIABLES: Dict[str, str] = {
    "mortar_prefix": "mortar",                                                  # SYMBOL prefix for variables which appear on mortar grids
    
    "pressure" : "p",                                                           # [Pa] = [N / m^2] = [ kg / m / s^2]

    "enthalpy" : "h",                                                           # (specific, molar) [J / mol] = [kg m^2 / s^2 / mol]
    "temperature" : "T",                                                        # [K]

    "displacement" : "u",                                                       # [m]
    
    "component_overall_fraction" : "zeta",                                      # (fractional, molar) [-]
    "component_fraction_in_phase" : "chi",                                      # (fractional, molar) [-]
    "phase_molar_fraction" : "xi",                                              # (fractional, molar) [-]
    "saturation" : "S",                                                         # (fractional, volumetric) [-]
}


def create_merged_variable(
    gb: pp.GridBucket, dof_info: Dict[str, int],
    variable_name: str,
    )-> Tuple[pp.ad.MergedVariable, Union[None, pp.ad.MergedVariable]]:
    """
    Creates domain-wide merged variables for a given grid bucket.
    Stores information about the variable in the data dictionary associated with each subdomain. 
    The key :data:`porepy.PRIMARY_VARIABLES` and given variable names are used for storage.

    For creating a variable without values on mortar grids, leave the argument `mortar_variable_name` as `None`.
    
    :param gb: grid bucket representing the whole computational domain
    :type gb: :class:`porepy.GridBucket`
    :param dof_info: number of DOFs per grid element (e.g. cell, face)
    :type dof_info: dict
    :param variable_name: name given to variable, used as keyword for storage
    :type variable_name: str

    :return: Returns a 2-tuple containing the new objects. The second object will be None, if no mortar variable name is specified.
    :rtype: tuple(2)
    """

    # creating variables on each subdomain
    variables = list()
    for g, d in gb:

        if pp.PRIMARY_VARIABLES not in d:
            d[pp.PRIMARY_VARIABLES] = dict()

        d[pp.PRIMARY_VARIABLES].update({variable_name: dof_info})
        # TODO test if the order of variables is alright
        variables.append(pp.ad.Variable(
            variable_name, dof_info, 
            grids=[g])) 

    return pp.ad.MergedVariable(variables)


def create_merged_mortar_variable(
    gb: pp.GridBucket, dof_info: Dict[str, int], mortar_variable_name: str
    ) -> pp.ad.MergedVariable:
    """ 
    Creates domain-wide mortar variables for a given grid bucket.
    Stores information about the variable in the data dictionary associated with each subdomain. 
    The key :data:`porepy.PRIMARY_VARIABLES` and given variable names are used for storage.
    
    :param gb: grid bucket representing the whole computational domain
    :type gb: :class:`porepy.GridBucket`
    :param dof_info: number of DOFs per grid element (e.g. cell, face)
    :type dof_info: dict
    :param mortar_variable_name: (optional) name given to respective mortar variable, used as keyword for storage. If none, no mortar variable will be assigned.
    :type mortar_variable_name: str

    :return: the new mortar variable
    :rtype: :class: `~porepy.numerics.ad.operators.MergedVariable`
    """

    mortar_variables = list()
    for e, d in gb.edges():
        # FIXME: assure the data dict has the respective keys
        if d["mortar_grid"].codim == 2:  # no variables in points
            continue
        else:
            d[pp.PRIMARY_VARIABLES].update({mortar_variable_name: dof_info})
            # TODO test if the order of variables is alright
            mortar_variables.append(pp.ad.Variable(
                mortar_variable_name, dof_info,
                edges=[e], num_cells=d["mortar_grid"].num_cells))

    return pp.ad.MergedVariable(mortar_variables)