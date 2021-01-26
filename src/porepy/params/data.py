""" Contains class for storing data / parameters associated with a grid.

At present, the Parameters class is a simple wrapper around a dictionary.

The Parameters will be stored as a dictionary identified by pp.PARAMETERS in an
"outer" dictionary (e.g. the data on the grid bucket nodes). In the Parameters object,
there will be one dictionary containing parameters for each keyword. The keywords link
parameters to discretization operators. For example, the operator

discr = pp.Tpfa(keyword="flow")

will access parameters under the keyword "flow". If outer_dictionary is the above
mentioned outer dictionary, these parameters will be found in

outer_dictionary[pp.PARAMETERS]["flow'],

and the boundary values are extracted from this dictionary as

bc = outer_dictionary[pp.PARAMETERS]["flow']["bc_values"]


There is a (not entirely clear) distinction between two types of parameters:
"Mathematical" parameters are those operated on by the discretization objects, and
should be thought of as corresponding to the terms of the mathematical equation.
"Physical" parameters are the actual physical properties of the media.
As an example, the standard incompressible convection-diffusion equation for temperature

    c \rho dT/dt + v \cdot \nabla T - \nabla \cdot (D \nabla T) = f

has the physical parameters c (specific heat capacity) and \rho (density). But from the
mathematical point of view, these combine to the parameter "mass_weight". Similarly,
the heat diffusion tensor ("physical" name) corresponds to the "second_order_tensor"
("mathematical" name).
If we consider the Darcy equation as another example, the "second_order_tensor" is
commonly termed the permeability ("physical"). Since the discretization schemes
do not know the physical terminology, the dictionary passed to these has to have the
_mathematical_ parameters defined. Solving (systems of) equations with multiple
instances of the same mathematical parameter (e.g. both thermal diffusivity and
permeability) is handled by the use of multiple keywords (e.g. "transport" and "flow").

Some default inner dictionaries are provided in pp.params.parameter_dictionaries.

For most instances, a convenient way to set up the parameters is:

    specified_parameters = {pm_1: val_1, ..., pm_n: val_n}
    data = pp.initialize_default_data(grid, {}, keyword, specified_parameters)

This will assign val_i to the specified parameters pm_i and default parameters to other
required parameters. If the data directory already exists as d (e.g. in the grid
bucket), consider:

    pp.initialize_default_data(grid, d, keyword, specified_parameters)


Also contains a function for setting the state. The state is all data associated with
the previous time step or iteration, and is stored in data[pp.STATE]. The solution of a
variable is stored in

data[pp.STATE][variable_name],

whereas data such as BC values are stored similarly to in the Parameters class, in

data[pp.STATE][keyword]["bc_values"].
"""
import numbers
import warnings
from typing import Dict, List, Optional

import numpy as np

import porepy as pp
import porepy.params.parameter_dictionaries as dicts

module_sections = ["parameters"]


class Parameters(dict):
    """Class to store all physical parameters used by solvers.

    The intention is to provide a unified way of passing around parameters, and
    also circumvent the issue of using a solver for multiple physical
    processes (e.g. different types of boundary conditions in multi-physics
    applications). The keyword assigned to parameters and discretization operators ensures
    that the right data is used: An operator will always use the parameters stored
    with under the keyword it has been assigned.

    The parameter class is a thin wrapper around a dictionary. This dictionary contains
    one sub-dictionary for each keyword.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, g=None, keywords=None, dictionaries=None):
        """Initialize Data object.

        Parameters:

        g - grid:
            Grid where the data is valid. Currently, only number of cells and
            faces are accessed.
        keywords: List of keywords to set parameters for. If none is passed, a
            parameter class without specified keywords is initialized.
        dictionaries: List of dictionaries with specified parameters, one for each
            keyword in keywords.
        """
        if not keywords:
            keywords = []
        if not dictionaries:
            dictionaries = []
        self.update_dictionaries(keywords, dictionaries)
        self.grid = g

    @pp.time_logger(sections=module_sections)
    def __repr__(self):
        s = "Data object for physical processes "
        s += ", ".join(str(k) for k in self.keys())
        for k in self.keys():
            s += '\nThe keyword "{}" has the following parameters specified: '.format(k)
            s += ", ".join(str(p) for p in self[k].keys())
        return s

    @pp.time_logger(sections=module_sections)
    def update_dictionaries(self, keywords, dictionaries=None):
        """Update the dictionaries corresponding to some keywords.

        Use either the dictionaries OR the property_ids / values.
        Properties:
            keywords - list of n_phys different physical processes.
            dictionaries - list of n_phys dictionaries with the properties to be
                updated. If not provided, empty dictionaries are used for all keywords.

        Example:
            keywords = ['flow', 'heat']
            ones = np.ones(g.num_cells)
            dicts = [{'porosity': 0.3 * ones, 'density': 42 * ones},
                    {'heat_capacity': 1.5 * np.ones}]
            param.upate(keywords, dicts)
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        if dictionaries is None:
            dictionaries = [{} for _ in range(len(keywords))]

        for (i, key) in enumerate(keywords):
            if key in self:
                self[key].update(dictionaries[i])
            else:
                self[key] = dictionaries[i]

    @pp.time_logger(sections=module_sections)
    def set_from_other(self, keyword_add, keyword_get, parameters):
        """Add parameters from existing values for a different keyword.

        Typical usage: Ensure parameters like aperture and porosity are consistent
        between keywords, by making reference the same object. Subsequent calls to
        modify_parameters should update the parameters for both keywords. Note that this
        will not work for Numbers, which are immutable in Python.

        Parameters:
            keyword_add: The keyword to whose dictionary the parameters are to be
                added.
            keyword_get: The keyword from whose dictionary the parameters are to be
            obtained.
            parameters: List of parameters to be set.
        """
        for p in parameters:
            self[keyword_add][p] = self[keyword_get][p]

    @pp.time_logger(sections=module_sections)
    def overwrite_shared_parameters(self, parameters, values):
        """Updates the given parameter for all keywords.

        Brute force method to ensure a parameter is updated/overwritten for all
        keywords where they are defined.
        parameters: List of (existing) parameters to be overwritten.
            values: List of new values.
        """
        for kw in self.keys():
            for (p, v) in zip(parameters, values):
                if p in self[kw]:
                    self[kw][p] = v

    @pp.time_logger(sections=module_sections)
    def modify_parameters(self, keyword, parameters, values):
        """Modify the values of some parameters of a given keyword.

        Usage: Ensure consistent parameter updates, see set_from_other. Does not work
        on Numbers.
        Parameters:
            parameters: List of (existing) parameters to be updated.
            values: List of new values. There are implicit assumptions on the values;
                in particular that the type and length of the new and old values agree,
                see modify_variable.
        """
        for (p, v) in zip(parameters, values):
            modify_variable(self[keyword][p], v)

    @pp.time_logger(sections=module_sections)
    def expand_scalars(
        self,
        n_vals: int,
        keyword: str,
        parameters: List[str],
        defaults: Optional[List] = None,
    ) -> List:
        """Expand parameters assigned as a single scalar to n_vals arrays.
        Used e.g. for parameters which may be heterogeneous in space (cellwise),
        but are often homogeneous and assigned as a scalar.
        Parameters:
            n_vals: Size of the expanded arrays. E.g. g.num_cells
            keyword: The parameter keyword.
            parameters: List of parameters.
            @pp.time_logger(sections=module_sections)
            defaults (optional): List of default values, one for each parameter.
                If not set, no default values will be provided and an error
                will ensue if one of the listed parameters is not present in
                the dictionary. This avoids assigning None to unset mandatory
                parameters.
        """
        values = []
        if defaults is None:
            defaults = [None] * len(parameters)
        for p, d in zip(parameters, defaults):
            if d is None:
                val = self[keyword].get(p)
            else:
                val = self[keyword].get(p, d)
            if np.asarray(val).size == 1:
                val *= np.ones(n_vals)
            values.append(val)
        return values


"""
Utility methods for handling of dictionaries.
TODO: Improve/add/remove methods based on experience with setting up problems using the
new Parameters class.
"""


@pp.time_logger(sections=module_sections)
def initialize_default_data(
    g, data, parameter_type, specified_parameters=None, keyword=None
):
    """Initialize a data dictionary for a single keyword.

    The initialization consists of adding a parameter dictionary and initializing a
    matrix dictionary in the proper fields of data. Default data are added for a certain
    set of "basic" parameters, depending on the type chosen.

    Args:
        g: Grid object with computed geometry.
        data: Outer data dictionary, to which the parameters will be added.
        parameter_type: Which type of parameters to use for the default assignment.
            Must be one of the following:
                "flow", "transport" and "mechanics".
        specified_parameters: A dictionary with specified parameters, overriding the
            @pp.time_logger(sections=module_sections)
            default values. Defualts to an empty dictionary (only default values).
        keyword: String to identify the parameters. Defaults to the parameter type.

     Returns:
        data: The filled dictionary.

    Raises:
        KeyError if an unknown parameter type is passed.
    """
    if not specified_parameters:
        specified_parameters = {}
    if not keyword:
        keyword = parameter_type
    if parameter_type == "flow":
        d = dicts.flow_dictionary(g, specified_parameters)
    elif parameter_type == "transport":
        d = dicts.transport_dictionary(g, specified_parameters)
    elif parameter_type == "mechanics":
        d = dicts.mechanics_dictionary(g, specified_parameters)
    else:
        raise KeyError(
            'Default dictionaries only exist for the parameter types "flow", '
            + '"transport" and "mechanics", not for '
            + parameter_type
            + "."
        )
    return initialize_data(g, data, keyword, d)


@pp.time_logger(sections=module_sections)
def initialize_data(
    g, data: Dict, keyword: str, specified_parameters: Optional[Dict] = None
) -> Dict:
    """Initialize a data dictionary for a single keyword.

    The initialization consists of adding a parameter dictionary and initializing a
    matrix dictionary in the proper fields of data. If there is a Parameters object
    in data, the new keyword is added using the update_dictionaries method.

    Args:
        g: The grid. Can be either standard grid, or mortar grid.
        data: Outer data dictionary, to which the parameters will be added.
        keyword: String identifying the parameters.
        specified_parameters: A dictionary with specified parameters, defaults to empty
            dictionary.

    Returns:
        data: The filled dictionary.
    """
    if not specified_parameters:
        specified_parameters = {}
    add_discretization_matrix_keyword(data, keyword)
    if pp.PARAMETERS in data:
        data[pp.PARAMETERS].update_dictionaries([keyword], [specified_parameters])
    else:
        data[pp.PARAMETERS] = pp.Parameters(g, [keyword], [specified_parameters])
    return data


@pp.time_logger(sections=module_sections)
def set_state(data: Dict, state: Optional[Dict] = None) -> Dict:
    """Initialize or update a state dictionary.

    The initialization consists of adding a state dictionary in the proper field of the
    data dictionary. If there is a state dictionary in data, the new state is added
    using the update method of dictionaries.

    Args:
        data: Outer data dictionary, to which the parameters will be added.
        state: A dictionary with the state, set to an empty dictionary if not provided.

    Returns:
        data: The filled dictionary.
    """
    state = state or {}
    if pp.STATE in data:
        data[pp.STATE].update(state)
    else:
        data[pp.STATE] = state
    return data


@pp.time_logger(sections=module_sections)
def set_iterate(data: Dict, iterate: Optional[Dict] = None) -> Dict:
    """Initialize or update an iterate dictionary.

    Same as set_state for subfield pp.ITERATE
    Also checks whether pp.STATE field is set, and adds it if not, see set_state.
    """
    if pp.STATE not in data:
        set_state(data)
    iterate = iterate or {}
    if pp.ITERATE in data[pp.STATE]:
        data[pp.STATE][pp.ITERATE].update(iterate)
    else:
        data[pp.STATE][pp.ITERATE] = iterate
    return data


@pp.time_logger(sections=module_sections)
def modify_variable(variable, new_value):
    """Changes the value (not id) of the stored parameter.

    Mutes the value of a variable to new_value.
    Note that this method cannot be extended to cover Numbers, as these are
    immutable in Python.
    Note that there are implicit assumptions on the arguments, in particular that
    the new value is of the same type as the variable. Further, if variable is a
        list, the lists should have the same length
        np.ndarray, the arrays should have the same shape, and new_value must be
            convertible to variable.dtype
    Args:
        variable: The variable.
        new_value: The new value to be assigned to the variable.

    Raises:
        TypeError if the variable is a number.
        NotImplementedError if a variable of unknown type is passed.
    """
    if isinstance(variable, np.ndarray):
        if variable.dtype != new_value.dtype:
            warnings.warn("Modifying array: new and old values have different dtypes.")
        variable.setfield(new_value, variable.dtype)
    elif isinstance(variable, list):
        variable[:] = new_value
    elif isinstance(variable, pp.SecondOrderTensor):
        variable.values[:] = new_value.values
    elif isinstance(variable, numbers.Number):
        raise TypeError("Numbers are immutable.")
    else:
        raise NotImplementedError(
            "No mute method implemented for variable of type " + str(type(variable))
        )


@pp.time_logger(sections=module_sections)
def add_nonpresent_dictionary(dictionary, key):
    """
    Check if key is in the dictionary, if not add it with an empty dictionary.
    """
    if key not in dictionary:
        dictionary[key] = {}


@pp.time_logger(sections=module_sections)
def add_discretization_matrix_keyword(dictionary, keyword):
    """Ensure presence of sub-dictionaries.

    Specific method ensuring that there is a sub-dictionary for discretization matrices,
    and that this contains a sub-sub-dictionary for the given key. Called previous to
    discretization matrix storage in discretization operators (e.g. the storage of
    "flux" by the Tpfa().discretize function).

    Parameters:
        dictionary: Main dictionary, typically stored on a grid bucket node.
        keyword: The keyword used for linking parameters and discretization operators.
    """
    add_nonpresent_dictionary(dictionary, pp.DISCRETIZATION_MATRICES)
    add_nonpresent_dictionary(dictionary[pp.DISCRETIZATION_MATRICES], keyword)
    return dictionary[pp.DISCRETIZATION_MATRICES][keyword]
