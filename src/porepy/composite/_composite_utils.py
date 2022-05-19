""" Utility functions for the composite submodule.

Data:
    - COMPUTATIONAL_VARIABLES

Functions:
    - create_merged_variable_on_gb

"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import porepy as pp

""" Contains an overview of computational variables.

The dictionary :data:`~porepy.params.computational_variables.COMPUTATIONAL_VARIABLES`
contains strings as keys and 2-tuples as values.

The key is a word describing the physical variable.
They are used for accessing the variable using
:class:`~pp.composite.material_subdomain.MaterialSubDomain` and
:class: `~porepy.compostie.computational_domain.ComputationalDomain`.

The value is a string, representing the mathematical symbol,

Above information is used to construct AD variables and to store respective information
in the grid data structure.
(e.g. if pressure has the symbol 'p', the relevant information will be found in
'data[pp.PRIMARY_VARIABLES]["p"]')

IMPORTANT: For more complex components, the symbol string will be augmented.
E.g. if component molar fractions in phase have the symbol 'chi', the final name for a
specific component in a specific phase will be 'chi_<component name>_<phase name>'.
EXAMPLE: the molar fraction of water in vapor form can have the name
    xi_H2O_vapor
if the modeler decides to call the substance class 'H20' and the phase 'vapor'

Assumed, default SI units of the respective variables are found below in comments.
"""
COMPUTATIONAL_VARIABLES: Dict[str, str] = {
    "mortar_prefix": "mortar",  # SYMBOL prefix for variables which appear on mortar grids
    "pressure": "p",  # [Pa] = [N / m^2] = [ kg / m / s^2]
    "enthalpy": "h",  # (specific, molar) [J / mol] = [kg m^2 / s^2 / mol]
    "temperature": "T",  # [K]
    "displacement": "u",  # [m]
    "component_overall_fraction": "zeta",  # (fractional, molar) [-]
    "component_fraction_in_phase": "chi",  # (fractional, molar) [-]
    "phase_molar_fraction": "xi",  # (fractional, molar) [-]
    "saturation": "S",  # (fractional, volumetric) [-]
}

""" Currently supported states of matter.
This influences the parameters for physical attributes, as well as the class
:class:`~porepy.composite.phase.PhysicalState`.
"""
STATES_OF_MATTER: Tuple = ("solid", "liquid", "gas")

"""
Universal molar gas constant.
        Math. Dimension:        scalar
        Phys. Dimension:        [kg m^2 / s K mol]
"""
IDEAL_GAS_CONSTANT: float = 8.31446261815324


def create_merged_variable(
    gb: "pp.GridBucket",
    dof_info: Dict[str, int],
    variable_name: str,
) -> "pp.ad.MergedVariable":
    """
    Creates domain-wide merged variables for a given grid bucket.
    Stores information about the variable in the data dictionary associated with each
    subdomain.

    :param gb: grid bucket representing the whole computational domain
    :type gb: :class:`~porepy.GridBucket`
    :param dof_info: number of DOFs per grid element (e.g. cells, faces, nodes)
    :type dof_info: dict
    :param variable_name: name given to variable, used as keyword for storage
    :type variable_name: str

    :return: Returns a new MergedVariable
    :rtype: :class:`~pore.ad.MergedVariable`
    """
    # creating variables on each subdomain
    variables = list()
    for g, d in gb:
        # store DOF information about variable
        if pp.PRIMARY_VARIABLES not in d.keys():
            d[pp.PRIMARY_VARIABLES] = dict()

        d[pp.PRIMARY_VARIABLES].update({variable_name: dof_info})

        # create grid-specific variable
        variables.append(pp.ad.Variable(variable_name, dof_info, grids=[g]))

        # initiate state and iterative state as zero
        if pp.STATE not in d:
            d[pp.STATE] = {}
        if pp.ITERATE not in d[pp.STATE]:
            d[pp.STATE][pp.ITERATE] = {}

        d[pp.STATE].update({variable_name: np.zeros(g.num_cells)})
        d[pp.STATE][pp.ITERATE].update({variable_name: np.zeros(g.num_cells)})
        # TODO for variables with not only cell values, above is wrong/incomplete

    return pp.ad.MergedVariable(variables)


def create_merged_mortar_variable(
    gb: "pp.GridBucket", dof_info: Dict[str, int], mortar_variable_name: str
) -> "pp.ad.MergedVariable":
    """
    Creates domain-wide mortar variables for a given grid bucket.
    Stores information about the variable in the data dictionary associated with each
    subdomain.
    The key :data:`porepy.PRIMARY_VARIABLES` and given variable names are used for storage.

    :param gb: grid bucket representing the whole computational domain
    :type gb: :class:`porepy.GridBucket`
    :param dof_info: number of DOFs per grid element (e.g. cell, face)
    :type dof_info: dict
    :param mortar_variable_name: (optional) name given to respective mortar variable,
    used as keyword for storage. If none, no mortar variable will be assigned.
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
            mortar_variables.append(
                pp.ad.Variable(
                    mortar_variable_name,
                    dof_info,
                    edges=[e],
                    num_cells=d["mortar_grid"].num_cells,
                )
            )

    return pp.ad.MergedVariable(mortar_variables)


def operator_sum(operators: List["pp.ad.Operator"]) -> "pp.ad.Operator":
    """This function is just for surpassing Typing error when using built-in sum().
    """
    out = operators[0]
    if len(operators) > 1:
        for op in operators[1:]:
            out += op
    return out


class ConvergenceError(Exception):
    """Error class to be raised when numerical algorithms do not converge."""

    def __init__(self, *args) -> None:
        """Constructor for catching arguments passed by the error stack."""
        if args:
            self._msg = args[0]
        else:
            self._msg = None

    def __str__(self) -> str:
        """Construct the actual error message."""

        if self._msg:
            return "ConvergenceError: %s" % (str(self._msg))
        else:
            return "ConvergenceError."
