"""This module contains helper functions for setting, getting and shifting values in the
data dictionary of a grid. This functionality is used by Ad Variables (including
MixedDimensionalVariables) and TimeDependentDenseArrays.

The main functions are:
    - :func:`set_solution_values`
    - :func:`get_solution_values`
    - :func:`shift_solution_values`

"""

from functools import lru_cache
from typing import Any, Optional

import numpy as np

import porepy as pp

__all__ = [
    "set_solution_values",
    "get_solution_values",
    "shift_solution_values",
]


def set_solution_values(
    name: str,
    values: np.ndarray,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
    additive: bool = False,
    reference: bool = False,
) -> None:
    """Function for setting values in the data dictionary, for some time-dependent or
    iterative term.

    Parameters:
        name: Name of the quantity that is to be assigned values.
        values: The values that are set in the data dictionary.
        data: Data dictionary corresponding to the subdomain or interface in question.
        time_step_index:

            Determines the key of where ``values`` are to be stored in
            ``data[pp.TIME_STEP_SOLUTIONS][name]``.
            0 is the most recent time step, 1 the one before that and so on.
        iterate_index:

            Determines the key of where ``values`` are to be stored in
            ``data[pp.ITERATE_SOLUTIONS][name]``.
            0 is the current iterate, 1 the previous iterate, and so on.
        additive:

            Flag to decide whether the values already stored in the data dictionary
            should be added to or overwritten.
        reference:

            Flag to decide whether reference values should be set instead of time step
            or iterate values. If ``True``, the setter will store values in
            ``data[pp.REFERENCE_SOLUTIONS][name]``.

    Raises:
        ValueError: In the case of inconsistent usage of indices (both None, or negative
            values).
        ValueError: If the user attempts to set values additively at an index where no
            values were set before.

    """
    if reference:
        _set_reference_values(name, values, data, additive)
        return

    loc_index = _validate_indices(time_step_index, iterate_index)

    for loc, index in loc_index:
        if loc not in data:
            data[loc] = {}
        if name not in data[loc]:
            data[loc][name] = {}

        if additive:
            if index not in data[loc][name]:
                raise ValueError(
                    f"Cannot set value additively for {name} at {(loc, index)}:"
                    + " No values stored to add to."
                )
            data[loc][name][index] += values
        else:
            data[loc][name][index] = values.copy()


def get_solution_values(
    name: str,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
    reference: bool = False,
) -> np.ndarray:
    """Function for fetching values stored in the data dictionary.

    The data dictionary can store values at different time steps, iterations, as well as
    reference values, and this function handles the fetching of values for all of these
    cases through the use of indices and flags. The following rules apply:
      1. If the ``reference`` flag is ``True``, values are fetched from the reference
         values, and the indices are ignored.
      2. If the ``reference`` flag is ``False``, values are fetched from the time step
         or iterate values, depending on which index is passed. If both are passed,
         an error is raised.

    Note:
        Compared to :func:`set_solution_values` the getter works only for 1 defined
        index, whereas the setter can take both a time and iterate index.

    Parameters:
        name: Name of the parameter whose values we are interested in. data: The data
        dictionary.

        time_step_index:

            Determines the key of where ``values`` are to be stored in
            ``data[pp.TIME_STEP_SOLUTIONS][name]``. 0 is the most recent time step, 1
            the one before that and so on.
        iterate_index:

            Determines the key of where ``values`` are to be stored in
            ``data[pp.ITERATE_SOLUTIONS][name]``. 0 is the current iterate, 1 the
            previous iterate, and so on.
        reference:
            Flag to decide whether reference values should be fetched instead of time
            step or iterate values. If ``True``, the getter will look for values in
            ``data[pp.REFERENCE_SOLUTIONS][name]``.

    Raises:
        ValueError: In the case of inconsistent usage of indices for time step and
            iterate (both None or negative values).
        ValueError: If the user attempts to get multiple iterate and time step values
            simultanously. Only 1 index is permitted in the getter.
        KeyError: If no values are stored for the passed index.

    Returns:
        A copy of the values stored at the passed index.

    """
    if reference:
        return _get_reference_values(name, data)
    # Get the location and index in the data dictionary where the values are stored. If
    # both time step and iterate indices are passed, this will give both, but we raise
    # an error in that case since we have not formally settled which to prioritize.
    loc_index = _validate_indices(time_step_index, iterate_index)
    if len(loc_index) != 1:
        raise ValueError(
            "Cannot get value from both iterate and time step at once. Call separately."
        )

    loc, index = loc_index[0]

    try:
        value = data[loc][name][index].copy()
    except KeyError as err:
        raise KeyError(
            f"No values stored for {name} at {(loc, index)}: {str(err)}."
        ) from err

    return value


def shift_solution_values(
    name: str, data: dict, location: Any, max_index: Optional[int] = None
) -> None:
    """Function to shift numerical values stored in the data dictionary.

    The shift is implemented s.t. values at index ``i`` are copied to index ``i + 1``.

    Note:
        The data stored must have support for ``.copy()`` in order to avoid faulty
        referencing (e.g., numpy arrays or sparse matrices).

    Note:
        After this operation, values at index 0 and 1 will be the same.
        Use :meth:`set_solution_values` to update the latest values at index 0.

    Parameters:
        name: Key in ``data`` for which quantity the shift should be performed.
        data: A grid data dictionary.
        location: Either :data:`~porepy.utils.common_constants.TIME_STEP_SOLUTIONS`
            or :data:`~porepy.utils.common_constants.ITERATE_SOLUTIONS`.
        max_index: ``default=None``

            A non-negative integer, capping the range of the shift operation to
            ``i -> max_index``.
            If called repeatedly with ``None``, the depth in ``location`` keeps
            increasing. To be used in schemes with a defined maximal depth of stored
            iterate or time step values.

    Raises:
        ValueError: If unsupported ``location`` is passed.
        ValueError: if ``max_index`` is negative.

    """
    if location == pp.REFERENCE_SOLUTIONS:
        _shift_to_reference_solutions(name, data)
        return

    if location not in [pp.ITERATE_SOLUTIONS, pp.TIME_STEP_SOLUTIONS]:
        raise ValueError(f"Shifting values not implemented for location {location}")

    # NOTE return because nothing to be shifted. Avoid confusion by introducing data
    # dictionaries for values which were never set using pp.set_solution_values.
    if location not in data:
        return
    if name not in data[location]:
        return

    num_stored = len(data[location][name])

    if max_index is not None:
        if max_index < 0:
            raise ValueError("Maximal index must be non-negative.")

        # Allow the number of stored values to increase
        if max_index > num_stored:
            range_ = range(num_stored, 0, -1)
        # don't allow it to increase
        else:
            range_ = range(max_index - 1, 0, -1)
            # TODO What should we do if for some reason already more stored?
    else:
        range_ = range(num_stored, 0, -1)

    for i in range_:
        data[location][name][i] = data[location][name][i - 1].copy()


def _set_reference_values(
    name: str, values: np.ndarray, data: dict, additive: bool = False
) -> None:
    """Function for setting reference values in the data dictionary.

    Parameters:
        name: Name of the quantity that is to be assigned values.
        values: The values that are set in the data dictionary.
        data: Data dictionary corresponding to the subdomain or interface in question.
        additive: ``default=False``

            Flag to decide whether the values already stored in the data dictionary
            should be added to or overwritten.

    Raises:
        ValueError: If the user attempts to set values additively at an index where no
            values were set before.

    """
    if pp.REFERENCE_SOLUTIONS not in data:
        data[pp.REFERENCE_SOLUTIONS] = {}

    if additive:
        if name not in data[pp.REFERENCE_SOLUTIONS]:
            raise ValueError(
                f"Cannot set value additively for {name} at reference values:"
                + " No values stored to add to."
            )
        data[pp.REFERENCE_SOLUTIONS][name] += values
    else:
        data[pp.REFERENCE_SOLUTIONS][name] = values.copy()


def _get_reference_values(name: str, data: dict) -> np.ndarray:
    """Function for fetching reference values stored in the data dictionary.

    Parameters:
        name: Name of the parameter whose values we are interested in.
        data: The data dictionary.

    Returns:
        A copy of the values stored at the passed category and name.
        0 if no values are stored for the passed category and name.

    """
    try:
        value = data[pp.REFERENCE_SOLUTIONS][name].copy()
    except KeyError as err:
        # No reference is set, we should return a zero array of the right shape. Look
        # for the shape in the iterate solutions, since these are most likely to be set.
        # If there are cases where no iterate solution is available, we will need to
        # think of a different approach.
        value = 0.0 * data[pp.ITERATE_SOLUTIONS][name][0]  # type: ignore
    return value


def _shift_to_reference_solutions(name: str, data: dict) -> None:
    """Shift the current iterate to reference values for specific name.

    This function shifts the reference values stored in the data dictionary for the
    specified category and name by one time step or iteration.

    Parameters:
        name: The name of the reference values.
        data: The data dictionary.

    """
    # Sanity checks.
    if pp.ITERATE_SOLUTIONS not in data:
        return
    elif name not in data[pp.ITERATE_SOLUTIONS]:
        return

    # Initialize data structure.
    if pp.REFERENCE_SOLUTIONS not in data:
        data[pp.REFERENCE_SOLUTIONS] = {}
    if name not in data[pp.REFERENCE_SOLUTIONS]:
        data[pp.REFERENCE_SOLUTIONS][name] = {}

    # Shift current iterate to the reference values.
    data[pp.REFERENCE_SOLUTIONS][name] = data[pp.ITERATE_SOLUTIONS][name][0].copy()


@lru_cache
def _validate_indices(
    time_step_index: Optional[int] = None, iterate_index: Optional[int] = None
) -> list[tuple[Any, int]]:
    """Helper method to validate the indexation of getter and setter methods for
    values in a grid's data dictionary.

    See :func:`set_solution_values` and :func:`get_solution_values`.

    """
    if time_step_index is None and iterate_index is None:
        raise ValueError(
            "At least one of time_step_index and iterate_index needs to be different"
            " from None."
        )

    out = []

    if iterate_index is not None:
        # Some valid iterate value.
        if iterate_index >= 0:
            out.append((pp.ITERATE_SOLUTIONS, iterate_index))
        # Negative iterate indices are not supported
        else:
            raise ValueError(
                "Use increasing, non-negative integers for iterate indices."
            )

    if time_step_index is not None:
        # Some previous time.
        if time_step_index >= 0:
            out.append((pp.TIME_STEP_SOLUTIONS, time_step_index))
        # Negative time step indices are not supported.
        else:
            raise ValueError(
                "Use increasing, non-negative integers for time step indices."
            )

    return out
