"""Test functionality of the ``DofManager``.

The ``DofManager`` is on its way to be phased out. The only functionality tested here is
the ``get_variable_values`` function, which was not working correctly.

"""

import numpy as np
import pytest

import porepy as pp


def test_get_variable_values() -> None:
    """Check that the ``get_variable_values`` function works.

    This is  only tested for a single-dimensional grid and one variable. Different
    combinations of variables or a multi-dimensional grid are NOT tested.
    """
    # Create an mdg without fractures.
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fracture_network = pp.FractureNetwork2d(domain=domain)
    mdg = fracture_network.mesh(
        mesh_args={"mesh_size_frac": 0.1, "mesh_size_bound": 0.1, "mesh_size_min": 0.1}
    )
    # Set dof and initial values for a test variable and initialize the ``DofManager``.
    # Note, that the for loop is only over a single grid.
    for sd, data in mdg.subdomains(return_data=True):
        data[pp.PRIMARY_VARIABLES] = {"test_var": {"cells": 1}}
        pp.set_solution_values(
            name="test_var",
            values=np.full(sd.num_cells, 0.0),
            data=data,
            time_step_index=0,
        )
    dof_manager = pp.DofManager(mdg)

    variable_values = dof_manager.get_variable_values(time_step_index=0)
    assert np.all(variable_values == np.full(mdg.subdomains()[0].num_cells, 0.0))
