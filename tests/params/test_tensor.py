import numpy as np
import pytest

import porepy as pp


@pytest.mark.parametrize(
    "kxx, kyy, kxy, kyz, nd, expected_diagonal, expected_isotropic",
    [
        (np.ones(4) * 2, None, None, None, 3, True, True),
        (np.array([1, 2, 3, 4]), None, None, None, 3, True, True),
        (np.ones(4), np.ones(4) * 2, None, None, 3, True, False),
        (np.ones(4), None, None, np.ones(4) * 0.5, 3, False, False),
        (np.ones(4), None, np.ones(4) * 0.5, None, 2, False, False),
        (np.ones(4), None, None, None, 1, True, True),
    ],
)
def test_is_diagonal_and_is_isotropic(
    kxx, kyy, kxy, kyz, nd, expected_diagonal, expected_isotropic
):
    """Tests the two methods is_diagonal and is_isotropic of the second order tensor.

    The methods return whether the tensor is diagonal or isotropic. This test tests that
    the methods return the expected result for different second order permeability
    tensors. Specifically we consider here isotropic, anisotropic, diagonal,
    non-diagonal, homogeneous and heterogeneous tensors. All three spatial dimensions
    are tested.

    Parameters:
        kxx: Array with cell-wise values of kxx permeability.
        kyy: Array of kyy.
        kzz: Array of kzz.
        kxy: Array of kxy.
        kyz: Array of kyz.

    """
    kwargs = {"kxx": kxx}
    if kyy is not None:
        kwargs["kyy"] = kyy
    if kxy is not None:
        kwargs["kxy"] = kxy
    if kyz is not None:
        kwargs["kyz"] = kyz

    perm = pp.SecondOrderTensor(**kwargs)

    assert perm.is_diagonal(nd) == expected_diagonal
    assert perm.is_isotropic(nd) == expected_isotropic


def test_restrict_to_cells():
    """Testing the restriction of a tensor to a chosen set of cells.

    The method restrict_to_cells is used to restrict e.g. the permeability tensor to a
    region within the simulation domain. This method checks that this restriction is
    done correctly.

    TODO: Make a test also for the fourth order tensor (or perhaps only for the fourth
    order tensor). This is needed to check that the restriction happens as expected also
    for tensors constructed by several fields (e.g. lambda and mu, or potentially even
    more.)

    """
    np.random.seed(42)

    # Defining an arbitrary anisotropic second order tensor:
    kxx = np.random.uniform(0.5, 1.0, 16)
    kyy = np.random.uniform(0.5, 1.0, 16)
    kxy = np.random.uniform(0.1, 0.5, 16)

    k = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)

    # Define some cells which we want to restrict the tensor k to:
    active_cells = np.array([2, 1, 14, 10, 15])

    # First copy k, and then restrict the copied k to the cell we chose above.
    k_copy = k.copy()
    k_copy.restrict_to_cells(active_cells)

    # Looping through the permeability tensors to check that the restriction is done
    # correctly.
    for ind, cell in enumerate(active_cells):
        original = k.values[:, :, cell]
        copied_restriction = k_copy.values[:, :, ind]

        assert np.array_equal(original, copied_restriction)
