""" Tests of methods from porepy.vis.plot_grid.
"""

import os
import pytest

import numpy as np

import porepy as pp
from porepy.grids.standard_grids import md_grids_2d, md_grids_3d


SCALAR_VARIABLE = "scalar"
VECTOR_VARIABLE = "vector_cell"


@pytest.fixture(
    params=[
        md_grids_2d.single_horizontal(mesh_args=[5, 5], simplex=False),
        md_grids_2d.single_vertical(simplex=True),
        md_grids_3d.single_horizontal(mesh_args=[3, 3, 3], simplex=False),
    ],
)
def mdg(request):
    """Helper for initialization of the parametrized mdgs"""
    mdg_ = request.param[0]
    _initialize_mdg(mdg_)
    return mdg_


def test_plot_grid_mdg(mdg):
    """Tests that no error is raised if we plot mdg and provide variable names."""
    pp.plot_grid(
        mdg,
        cell_value=SCALAR_VARIABLE,
        vector_variable=VECTOR_VARIABLE,
        vector_scale=10,
        info="CNFO",
    )


def test_plot_grid_simple_grid(mdg):
    """Tests that no error is risen if we plot the single and provide variable arrays.
    This use case requires the user to reshape the vector array to the shape (3 x n).
    The redundant dimensions are filled with zeros."""
    grid, data = mdg.subdomains(return_data=True)[0]
    scalar_data = data[pp.STATE][SCALAR_VARIABLE]
    vector_data = data[pp.STATE][VECTOR_VARIABLE].reshape((mdg.dim_max(), -1), order="F")
    vector_data = np.vstack([vector_data, np.zeros((3 - mdg.dim_max(), vector_data.shape[1]))])
    pp.plot_grid(
        grid,
        cell_value=scalar_data,
        vector_variable=vector_data,
        vector_scale=10,
        info="CNFO",
    )


def test_save_img():
    """Tests that `save_img` saves some image."""

    mdg_ = md_grids_2d.single_horizontal(mesh_args=[5, 5], simplex=False)[0]
    _initialize_mdg(mdg_)

    image_name = "test_save_img.png"
    assert not os.path.exists(image_name)

    pp.save_img(
        image_name,
        mdg,
        cell_value=SCALAR_VARIABLE,
        vector_variable=VECTOR_VARIABLE,
    )

    assert os.path.exists(image_name)
    os.remove(image_name)


def _initialize_mdg(mdg_):
    """Initializes mdg with an arbitrary state.
    The state contains one scalar and one vector variable at cell centres."""

    for sd, data in mdg_.subdomains(return_data=True):
        if sd.dim in (mdg_.dim_max(), mdg_.dim_max() - 1):
            data[pp.STATE] = {
                SCALAR_VARIABLE: np.ones(sd.num_cells),
                VECTOR_VARIABLE: np.ones((sd.dim, sd.num_cells)).ravel(order="F"),
            }
        else:
            data[pp.PRIMARY_VARIABLES] = {}

    for intf, data in mdg_.interfaces(return_data=True):
        if intf.dim == mdg_.dim_max() - 1:
            data[pp.STATE] = {
                SCALAR_VARIABLE: np.ones(intf.num_cells),
                VECTOR_VARIABLE: np.ones((intf.dim, intf.num_cells)).ravel(order="F"),
            }
        else:
            data[pp.PRIMARY_VARIABLES] = {}
