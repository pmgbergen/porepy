""" Parameter dictionaries.

Here, we store various parameter dictionaries with "sensible" (typically unitary or
zero) default values for the parameters required by the discretization objects.
"""

import numpy as np

import porepy as pp


def flow_dictionary(g, in_data=None):
    """Dictionary with parameters for standard flow problems.

    All parameters listed below which are not specified in in_data are assigned default
    values. Additional parameters may be passed in in_data.

    Args:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.

    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that the standard flow parameters are present in d. Values from in_data
    # have priority over the default values.
    d = {
        "source": np.zeros(g.num_cells),
        "mass_weight": np.ones(g.num_cells),
        "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
        "bc": pp.BoundaryCondition(g),
        "bc_values": np.zeros(g.num_faces),
        "time_step": 1,
    }

    # Ensure that parameters not handled above are copied. Values from in_data have
    # priority over the default values.
    if not in_data:
        in_data = {}
    d.update(in_data)
    return d


def transport_dictionary(g, in_data=None):
    """Dictionary with parameters for standard transport problems.

    All parameters listed below which are not specified in in_data are assigned default
    values. Additional parameters may be passed in in_data.

    Args:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.

    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that the standard transport parameters are present in d. Values from
    # in_data have priority over the default values.
    d = {
        "source": np.zeros(g.num_cells),
        "mass_weight": np.ones(g.num_cells),
        "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
        "bc": pp.BoundaryCondition(g),
        "bc_values": np.zeros(g.num_faces),
        "darcy_flux": np.zeros(g.num_faces),
        "time_step": 1,
    }

    # Ensure that parameters not handled above are copied. Values from in_data have
    # priority over the default values.
    if not in_data:
        in_data = {}
    d.update(in_data)
    return d


def mechanics_dictionary(g, in_data=None):
    """Dictionary with parameters for standard mechanics problems.

    All parameters listed below which are not specified in in_data are assigned default
    values. Additional parameters may be passed in in_data.

    Args:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.

    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that the standard mechanics parameters are present in d.
    d = {
        "porosity": np.ones(g.num_cells),
        "source": np.zeros(g.dim * g.num_cells),
        "mass_weight": np.ones(g.num_cells),
        "fourth_order_tensor": pp.FourthOrderTensor(
            np.ones(g.num_cells), np.ones(g.num_cells)
        ),
        "bc": pp.BoundaryConditionVectorial(g),
        "bc_values": np.zeros(g.dim * g.num_faces),
        "slip_distance": np.zeros(g.num_faces * g.dim),
        "time_step": 1,
    }

    # Ensure that parameters not handled above are copied. Values from in_data have
    # priority over the default values.
    if not in_data:
        in_data = {}
    d.update(in_data)
    return d
