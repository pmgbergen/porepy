r"""Contains a function to initialize a dictionary for storing parameters associated
with a single grid.

The structure of the dictionary is the following:
```
outer_dictionary = {
    pp.PARAMETERS: {
        "flow": {
            # Parameters corresponding to the keyword "flow".
        },
        "darcy": {
            # Parameters corresponding to the keyword "darcy".
        },
        # and more keywords and inner dicts here.
    },
    pp.DISCRETIZATION_MATRICES: {
        "flow": {
            # Parameters corresponding to the keyword "flow".
        },
        "darcy": {
            # Parameters corresponding to the keyword "darcy".
        },
        # and more keywords and inner dicts here.
    },
    pp.TIME_STEP_SOLUTIONS: {
        # The time step solution is all data associated with the previous time step.
        "variable_name": {
            "time_step_index": ...  # The time step solution of a specific variable.
        },
        "some_keyword": {
            "bc_values": ...  # Data such as BC values is also stored in this dict.
        }
    },
    pp.ITERATE_SOLUTIONS: {
        # Data associated with the previous nonlinear iterates.
    },
    # and possibly more dicts here.
}
```
The keywords link parameters to discretization operators. For example, the operator
`discr = pp.Tpfa(keyword="flow")` will access parameters under the keyword "flow". For
instance, the boundary values are extracted from this dictionary as
`bc = outer_dictionary[pp.PARAMETERS]["flow"]["bc_values"]`.

There is a (not entirely clear) distinction between two types of parameters:
"Mathematical" parameters are those operated on by the discretization objects, and
should be thought of as corresponding to the terms of the mathematical equation.
"Physical" parameters are the actual physical properties of the media. As an example,
the standard incompressible convection-diffusion equation for temperature

    c \rho dT/dt + v \cdot \nabla T - \nabla \cdot (D \nabla T) = f

has the physical parameters c (specific heat capacity) and \rho (density). But from the
mathematical point of view, these combine to the parameter "mass_weight". Similarly, the
heat diffusion tensor ("physical" name) corresponds to the "second_order_tensor"
("mathematical" name). If we consider the Darcy equation as another example, the
"second_order_tensor" is commonly termed the permeability ("physical"). Since the
discretization schemes do not know the physical terminology, the dictionary passed to
these has to have the _mathematical_ parameters defined. Solving (systems of) equations
with multiple instances of the same mathematical parameter (e.g. both thermal
diffusivity and permeability) is handled by the use of multiple keywords (e.g.
"transport" and "flow").

For most instances, a convenient way to initialize the parameters is:
```
specified_parameters = {pm_1: val_1, ..., pm_n: val_n}
data = {}  # Or existing data dictionary.
data = pp.initialize_data(data, keyword, specified_parameters)
```

"""

from __future__ import annotations

import warnings
from typing import Any, Optional, overload

import porepy as pp


# The new signature. To only one the users should use, and the only one to be preserved.
@overload
def initialize_data(
    data: dict,
    keyword: str,
    specified_parameters: Optional[dict] = None,
) -> dict:
    pass


# The old signature to be deprecated. Using it will result in a warning now.
@overload
@warnings.deprecated(
    "Signature of pp.initialize_data has changed. Remove the first parameter (grid)."
)
def initialize_data(
    grid: Any,
    data: dict,
    keyword: str,
    specified_parameters: Optional[dict] = None,
) -> dict:
    pass


def initialize_data(*args, **kwargs) -> dict:
    """Initialize or update a data dictionary for a single keyword.

    This ensures that the proper nested structure sub-dictionaries is created and
    updates the sub-dictionary corresponding to the passed `keyword` with the
    `specified_parameters`. It can be called multiple times on the same dictionary to
    update it incrementally. The resulting dictionary will be updated by:
    ```
    {
        pp.PARAMETERS: {
            keyword: specified_parameters,
            # And old values
        },
        pp.DISCRETIZATION_MATRICES: {
            keyword: {
                # Only old values.
            }
        },
    }
    ```

    Parameters:
        data: Outer data dictionary, to which the parameters will be added. Can be empty
            if creating a new data dictionary. The same object will be returned from
            this function.
        keyword: String identifying the parameters.
        specified_parameters: A dictionary with specified parameters, defaults to empty
            dictionary.

    Note:
        For historical reasons, there was the first parameter "grid". It is not needed
        anymore, so we decided to change the signature. Chances are that this function
        is extensively used in user scripts, so just changing the signature will break
        things. Therefore, we temporarily support both signatures, and show a warning if
        the old one is used.

    Returns:
        The modified dictionary, same object as passed in `data` parameter.
    """
    # Parsing the passed parameters. We do not aim to cover all the combinations of
    # args and kwargs, as this is a temporary solution.
    data: dict
    keyword: str
    specified_parameters: Optional[dict] = None
    error_msg = (
        "Signature of pp.initialize_data has changed. Remove the first parameter (grid)"
        "."
    )
    warning_msg = error_msg + " This warning will become an error in future releases."
    if len(args) == 4 and len(kwargs) == 0:
        # initialize_data(grid, data, keyword, specified_parameters) - old signature.
        _, data, keyword, specified_parameters = args
        warnings.warn(warning_msg)
    elif len(args) == 3 and len(kwargs) == 1:
        # initialize_data(
        #     grid, data, keyword, specified_parameters=specified_parameters
        # ) - old signature.
        _, data, keyword = args
        specified_parameters = kwargs["specified_parameters"]
        warnings.warn(warning_msg)
    elif len(args) == 3 and len(kwargs) == 0:
        # Probably initialize_data(data, keyword, specified_parameters) - new signature.
        data, keyword, specified_parameters = args

        # Can also be initialize_data(grid, data, keyword) - old signature.
        if not isinstance(data, dict):
            _, data, keyword = args
            specified_parameters = None
            assert isinstance(data, dict)
            warnings.warn(warning_msg)

    elif len(args) == 2:
        # initialize_data(data, keyword) - new signature.
        data, keyword = args
        # or initialize_data(data, keyword, specified_parameters=specified_parameters)
        if "specified_parameters" in kwargs:
            specified_parameters = kwargs["specified_parameters"]
    else:
        # Anything else is exotic.
        raise TypeError(error_msg)

    if not specified_parameters:
        specified_parameters = {}
    add_discretization_matrix_keyword(data, keyword)

    add_nonpresent_dictionary(data, pp.PARAMETERS)
    add_nonpresent_dictionary(data[pp.PARAMETERS], keyword)
    data[pp.PARAMETERS][keyword].update(specified_parameters)
    return data


def add_nonpresent_dictionary(dictionary: dict, key: str) -> None:
    """Check if key is in the dictionary, if not add it with an empty dictionary.

    Parameters:
        dictionary: Dictionary to be updated.
        key: Keyword to be added to the dictionary if missing.
    """
    if key not in dictionary:
        dictionary[key] = {}


def add_discretization_matrix_keyword(dictionary: dict, keyword: str) -> dict:
    """Ensure presence of sub-dictionaries related to discretization.

    Specific method ensuring that there is a sub-dictionary for discretization matrices,
    and that this contains a sub-sub-dictionary for the given key. Called previous to
    discretization matrix storage in discretization operators (e.g. the storage of
    "flux" by the Tpfa().discretize function).

    Parameters:
        dictionary: Main dictionary, typically stored on a subdomain.
        keyword: The keyword used for linking parameters and discretization operators.

    Returns:
        Matrix dictionary of discretization matrices.
    """
    add_nonpresent_dictionary(dictionary, pp.DISCRETIZATION_MATRICES)
    add_nonpresent_dictionary(dictionary[pp.DISCRETIZATION_MATRICES], keyword)
    return dictionary[pp.DISCRETIZATION_MATRICES][keyword]
