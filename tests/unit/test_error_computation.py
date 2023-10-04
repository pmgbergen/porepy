"""Module containing tests for error computation."""

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.mdg_library import square_with_orthogonal_fractures


@pytest.fixture(scope="module")
def grids() -> list[pp.Grid, pp.MortarGrid]:
    """Create a mixed-dimensional grid on a unit square with a single fracture.

    Returns:
        A list containing one subdomain grid (a Cartesian 2x2 grid) and one mortar
        grid (a one dimensional mortar grid with 4 mortar cells).

    """
    mdg, _ = square_with_orthogonal_fractures(
        grid_type="cartesian",
        meshing_args={"cell_size": 0.5},
        fracture_indices=[0],
        size=1.0,
    )
    return [mdg.subdomains()[0], mdg.interfaces()[0]]


@pytest.mark.parametrize("is_relative", [False, True])
@pytest.mark.parametrize(
    "is_sd, is_cc, is_scalar",
    [
        (True, True, True),  # subdomain scalar cell-centered quantity
        (True, False, True),  # subdomain scalar face-centered quantity
        (True, True, False),  # subdomain vector cell-centered quantity
        (True, False, False),  # subdomain vector face-centered quantity
        (False, True, True),  # interface scalar cell-centered quantity
        (False, True, False),  # interface vector cell-centered quantity
    ],
)
def test_l2_error(
    is_sd: bool,
    is_scalar: bool,
    is_cc: bool,
    is_relative: bool,
    grids: list[pp.Grid, pp.MortarGrid],
) -> None:
    """Test whether the discrete L2-error is computed correctly.

    The test sets arrays of ones as for the true array, and arrays of zeros for the
    approximate arrays. The absolute l2-error is thus the square root of the sum of
    the measure of each element (cell_volume when is_cc=True and face_area when
    is_cc=False) in each grid. The relative l2-error is always 1.0 in all cases.

    Parameters:
        is_sd: Whether the error should be evaluated in a subdomain grid. False
            implies evaluation in an interface grid.
        is_scalar: Whether the array is corresponds to a scalar quantity. False
            implies a vector quantity.
        is_cc: Whether the array is a cell-centered quantity. False implies a
            face-centered quantity.
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """

    # Retrieve grid
    if is_sd:
        grid = grids[0]  # subdomain grid
    else:
        grid = grids[1]  # interface grid

    # Retrieve number of degrees of freedom and set the true array
    if is_cc:
        if is_scalar:
            ndof = grid.num_cells
            true_l2_error = np.sqrt(np.sum(grid.cell_volumes))
        else:
            if is_sd:
                ndof = 2 * grid.num_cells
                true_l2_error = np.sqrt(2 * np.sum(grid.cell_volumes))
            else:
                ndof = grid.num_cells
                true_l2_error = np.sqrt(np.sum(grid.cell_volumes))
    else:
        if is_scalar:
            ndof = grid.num_faces
            true_l2_error = np.sqrt(np.sum(grid.face_areas))
        else:
            ndof = 2 * grid.num_faces
            true_l2_error = np.sqrt(2 * np.sum(grid.face_areas))

    # Compute actual error
    actual_l2_error = pp.error_computation.l2_error(
        grid=grid,
        true_array=np.ones(ndof),
        approx_array=np.zeros(ndof),
        is_cc=is_cc,
        is_scalar=is_scalar,
        relative=is_relative,
    )

    # Compare
    if not is_relative:
        assert np.isclose(actual_l2_error, true_l2_error)
    else:
        assert np.isclose(actual_l2_error, 1.0)


def test_l2_error_division_by_zero_error(grids: list[pp.Grid, pp.MortarGrid]) -> None:
    """Test whether a division by zero error is raised.

    This error should be raised when the denominator is zero while computing the
    relative error.

    Parameters:
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """
    msg = "Attempted division by zero."
    with pytest.raises(ZeroDivisionError) as excinfo:
        # Attempt to obtain L2-relative error with true array of zeros
        pp.error_computation.l2_error(
            grid=grids[0],
            true_array=np.zeros(4),
            approx_array=np.random.random(4),
            is_cc=True,
            is_scalar=True,
            relative=True,
        )
    assert msg in str(excinfo.value)


def test_l2_error_not_implemented_error(grids: list[pp.Grid, pp.MortarGrid]) -> None:
    """Test whether a not implemented error is raised.

    The error should be raised when a face-centered quantity is passed together with
    a mortar grid.

    Parameters:
        grids: List of grids. The first element is a two-dimensional subdomain grid,
            and the second element is an interface grid. See the fixture grids().

    """
    msg = "Interface variables can only be cell-centered."
    with pytest.raises(NotImplementedError) as excinfo:
        # Attempt to compute the error for a face-centered quantity on a mortar grid
        pp.error_computation.l2_error(
            grid=grids[1],
            true_array=np.ones(6),
            approx_array=np.random.random(6),
            is_cc=False,
            is_scalar=True,
        )
    assert msg in str(excinfo.value)
