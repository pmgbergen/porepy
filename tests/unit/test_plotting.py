""" Tests of methods from porepy.vis.plot_grid.
"""

import os
import pytest

import numpy as np

import porepy as pp
from porepy.grids.standard_grids import md_grids_2d, md_grids_3d

plt = pytest.importorskip("matplotlib.pyplot")


SCALAR_VARIABLE = "scalar"
VECTOR_VARIABLE_CELL = "vector_cell"
VECTOR_VARIABLE_FACE = "vector_face"


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


@pytest.mark.parametrize("vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE])
def test_plot_grid_mdg(mdg, vector_variable):
    """Tests that no error is raised if we plot mdg and provide variable names."""
    pp.plot_grid(
        mdg,
        cell_value=SCALAR_VARIABLE,
        vector_value=vector_variable,
        vector_scale=10,
        info="CNFO",
        if_plot=False,
    )
    plt.close()


@pytest.mark.parametrize("vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE])
def test_plot_grid_simple_grid(mdg, vector_variable):
    """Tests that no error is raised if we plot a single dimension grid and provide variable arrays.
    This use case requires the user to reshape the vector array to the shape (3 x n).
    The redundant dimensions are filled with zeros."""
    grid, data = mdg.subdomains(return_data=True)[0]
    scalar_data = data[pp.STATE][SCALAR_VARIABLE]
    vector_data = data[pp.STATE][vector_variable].reshape((mdg.dim_max(), -1), order="F")
    vector_data = np.vstack(
        [vector_data, np.zeros((3 - vector_data.shape[0], vector_data.shape[1]))]
    )
    pp.plot_grid(
        grid,
        cell_value=scalar_data,
        vector_value=vector_data,
        vector_scale=10,
        info="CNFO",
        if_plot=False,
    )
    plt.close()


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
        vector_value=VECTOR_VARIABLE_CELL,
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
                VECTOR_VARIABLE_CELL: np.ones((mdg_.dim_max(), sd.num_cells)).ravel(order="F"),
                VECTOR_VARIABLE_FACE: np.ones((mdg_.dim_max(), sd.num_faces)).ravel(order="F"),
            }
        else:
            data[pp.STATE] = {}
