""" Tests of methods from porepy.vis.plot_grid.
"""

import os

import matplotlib
import numpy as np
import pytest

import porepy as pp
from porepy.grids.md_grid import MixedDimensionalGrid

# Setting a non-interactive backend that does not open new windows.
matplotlib.use("agg")

SCALAR_VARIABLE = "scalar"
VECTOR_VARIABLE_CELL = "vector_cell"
VECTOR_VARIABLE_FACE = "vector_face"


@pytest.fixture(
    params=[
        pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size": 0.2},
            fracture_indices=[1],  # Horizontal fracture
        ),
        pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size": 0.2},
            fracture_indices=[0],  # Vertical fracture
        ),
        pp.mdg_library.cube_with_orthogonal_fractures(
            "cartesian",
            {"cell_size": 1 / 3},
            fracture_indices=[2],  # Horizontal (constant z) fracture
        ),
    ],
)
def mdg(request: pytest.FixtureRequest) -> pp.MixedDimensionalGrid:
    """Initializes the mdg.

    The state contains one scalar variable at cell centers and two vector variables at
    cell centres and cell faces, respectively.

    """
    mdg = request.param[0]

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim in (mdg.dim_max(), mdg.dim_max() - 1):
            variables = np.array(
                [SCALAR_VARIABLE, VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE]
            )

            vals_scalar = np.ones(sd.num_cells)
            vals_vect_cell = np.ones((mdg.dim_max(), sd.num_cells)).ravel(order="F")
            vals_vect_face = np.ones((mdg.dim_max(), sd.num_faces)).ravel(order="F")

            values = [vals_scalar, vals_vect_cell, vals_vect_face]

            for i in range(len(variables)):
                pp.set_solution_values(
                    name=variables[i], values=values[i], data=data, time_step_index=0
                )

        else:
            data[pp.TIME_STEP_SOLUTIONS] = {}

    return mdg


@pytest.mark.parametrize(
    "vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE]
)
def test_plot_grid_mdg(mdg: MixedDimensionalGrid, vector_variable: str):
    """Testing that no error is raised if we plot mdg and provide variable names."""
    pp.plot_grid(
        mdg,
        cell_value=SCALAR_VARIABLE,
        vector_value=vector_variable,
        vector_scale=10,
        info="CNFO",
    )


@pytest.mark.parametrize(
    "vector_variable", [VECTOR_VARIABLE_CELL, VECTOR_VARIABLE_FACE]
)
def test_plot_grid_simple_grid(mdg: MixedDimensionalGrid, vector_variable: str):
    """Tests that no error is raised if we plot a single dimension grid and provide
    variable arrays.
    This use case requires the user to reshape the vector array to the shape (3 x n).
    The redundant dimensions are filled with zeros."""
    grid, data = mdg.subdomains(return_data=True)[0]
    scalar_data = pp.get_solution_values(
        name=SCALAR_VARIABLE, data=data, time_step_index=0
    )
    vector_data = pp.get_solution_values(
        name=vector_variable, data=data, time_step_index=0
    )
    vector_data = vector_data.reshape((mdg.dim_max(), -1), order="F")
    vector_data = np.vstack(
        [vector_data, np.zeros((3 - vector_data.shape[0], vector_data.shape[1]))]
    )
    pp.plot_grid(
        grid,
        cell_value=scalar_data,
        vector_value=vector_data,
        vector_scale=10,
        info="CNFO",
    )


@pytest.fixture
def image_name() -> str:
    image_name = "test_save_img.png"
    assert not os.path.exists(image_name)
    yield image_name

    # Tear down
    os.remove(image_name)


def test_save_img(mdg: MixedDimensionalGrid, image_name: str):
    """Testing that `save_img` works."""
    pp.save_img(
        image_name,
        mdg,
        cell_value=SCALAR_VARIABLE,
        vector_value=VECTOR_VARIABLE_CELL,
    )

    assert os.path.exists(image_name)
