""" Tests of methods from porepy.vis.plot_grid.
"""

import os

import numpy as np
import pytest

import porepy as pp
from porepy.grids.standard_grids import md_grids_2d, md_grids_3d
from porepy.numerics.ad.equation_system import set_time_dependent_value

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


@pytest.mark.parametrize(
    "vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE]
)
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


@pytest.mark.parametrize(
    "vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE]
)
def test_plot_grid_simple_grid(mdg, vector_variable):
    """Tests that no error is raised if we plot a single dimension grid and provide variable arrays.
    This use case requires the user to reshape the vector array to the shape (3 x n).
    The redundant dimensions are filled with zeros."""
    grid, data = mdg.subdomains(return_data=True)[0]
    scalar_data = data['stored_solutions'][SCALAR_VARIABLE][0]
    vector_data = data['stored_solutions'][vector_variable][0].reshape(
        (mdg.dim_max(), -1), order="F"
    )
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
    """Initializes mdg with an arbitrary solution.
    The solution contains one scalar and one vector variable at cell centres."""

    for sd, data in mdg_.subdomains(return_data=True):
        if sd.dim in (mdg_.dim_max(), mdg_.dim_max() - 1):
            variables = np.array([SCALAR_VARIABLE, 
                                  VECTOR_VARIABLE_CELL, 
                                  VECTOR_VARIABLE_FACE])

            vals_scalar = np.ones(sd.num_cells)
            vals_vect_cell = np.ones((mdg_.dim_max(), sd.num_cells)).ravel(order="F")
            vals_vect_face = np.ones((mdg_.dim_max(), sd.num_faces)).ravel(order="F")
            values = np.array([vals_scalar, vals_vect_cell, vals_vect_face])

            for i in range(len(variables)):
                data = set_time_dependent_value(name=variables[i], values=values[i], data=data, solution_index=0)

        else:
            data['stored_solutions'] = {}
