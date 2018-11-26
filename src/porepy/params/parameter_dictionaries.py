""" Parameter dictionaries.

Here, we store various parameter dictionaries with "sensible" default values for the
parameters required by the discretization objects.
Note that the values of the output dictionary are set according to the relationship
between the physical process and the equation. Thus, the mass_weight ("mathematical
term") of the flow dictionary is set to the value of the porosity ("pysical term").
The same goes for the tensors.
"""

import numpy as np
import porepy as pp


def flow_dictionary(g, in_data={}):
    """ Dictionary with parameters for standard flow problems.

    All parameters listed below which are not specified in in_data are assigned unitary
    values. Additional parameters may be passed in in_data.
    Parameters:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.
        kw: Keyword. The keyword is the identification connecting parameters and
            discretizations.
    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that parameters not handled below are copied.
    d = in_data.copy()
    # Ensure that the standard flow parameters are present in d. Values from in_data
    # have priority over the default values.
    d["aperture"] = in_data.get("aperture", np.ones(g.num_cells))
    d["porosity"] = in_data.get("porosity", np.ones(g.num_cells))
    d["fluid_compressibility"] = in_data.get(
        "fluid_compressibility", np.ones(g.num_cells)
    )
    d["mass_weight"] = in_data.get("mass_weight", d["porosity"])
    d["source"] = in_data.get("source", np.zeros(g.num_cells))
    d["time_step"] = in_data.get("time_step", 1)
    # Set the second order tensor from the permeability.
    d["second_order_tensor"] = in_data.get(
        "permeability", pp.SecondOrderTensor(g.dim, np.ones(g.num_cells))
    )
    d["bc"] = in_data.get("bc", pp.BoundaryCondition(g))
    d["bc_values"] = in_data.get("bc_values", np.zeros(g.num_faces))
    return d


def transport_dictionary(g, in_data={}):
    """ Dictionary with parameters for standard transport problems.

    All parameters listed below which are not specified in in_data are assigned unitary
    values. Additional parameters may be passed in in_data.
    Parameters:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.
        kw: Keyword. The keyword is the identification connecting parameters and
            discretizations.
    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that parameters not handled below are copied.
    d = in_data.copy()
    # Ensure that the standard flow parameters are present in d. Values from in_data
    # have priority over the default values.
    d = in_data.copy()
    d["aperture"] = in_data.get("aperture", np.ones(g.num_cells))
    d["porosity"] = in_data.get("porosity", np.ones(g.num_cells))
    d["source"] = in_data.get("source", np.zeros(g.num_cells))
    d["time_step"] = in_data.get("time_step", 1)
    # Set the second order tensor from the conductivity.
    d["second_order_tensor"] = in_data.get(
        "conductivity", pp.SecondOrderTensor(g.dim, np.ones(g.num_cells))
    )
    d["bc"] = in_data.get("bc", pp.BoundaryCondition(g))
    d["bc_values"] = in_data.get("bc_values", np.zeros(g.num_faces))
    d["discharge"] = in_data.get(
        "discharge", pp.Upwind().discharge(g, [0, 0, 0], d["aperture"])
    )
    d["mass_weight"] = in_data.get("mass_weight", d["porosity"])
    return d


def mechanics_dictionary(g, in_data={}):
    """ Dictionary with parameters for standard mechanics problems.

    All parameters listed below which are not specified in in_data are assigned unitary
    values. Additional parameters may be passed in in_data.
    Parameters:
        g: Grid.
        in_data: Dictionary containing any custom (non-default) parameter values.
        kw: Keyword. The keyword is the identification connecting parameters and
            discretizations.
    Returns:
        Dictionary with the "mathematical" parameters required by various flow
            discretization objects specified.
    """
    # Ensure that parameters not handled below are copied.
    d = in_data.copy()
    # Ensure that the standard flow parameters are present in d. Values from in_data
    # have priority over the default values.
    d["aperture"] = in_data.get("aperture", np.ones(g.num_cells))
    d["porosity"] = in_data.get("porosity", np.ones(g.num_cells))
    d["source"] = in_data.get("source", np.zeros(g.dim * g.num_cells))
    d["time_step"] = in_data.get("time_step", 1)
    # Set the fourth order tensor from the stiffness.
    d["fourth_order_tensor"] = in_data.get(
        "stiffness",
        pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells)),
    )
    d["bc"] = in_data.get("bc", pp.BoundaryConditionVectorial(g))
    d["bc_values"] = in_data.get("bc_values", np.zeros(g.dim * g.num_faces))
    d["slip_distance"] = in_data.get("slip_distance", np.zeros(g.num_faces * g.dim))
    return d
