from typing import Tuple
import numpy as np
import pytest

import porepy as pp


def fourth_order_tensor_for_testing(
    num_cells: int,
) -> Tuple[pp.FourthOrderTensor, np.ndarray]:
    """Construct a fourth order tensor with custom fields for testing.

    The tensor which is constructed here is for an anisotropic and homogeneous medium.
    Specifically, the medium is represented by five material parameters and represents a
    transversely isotropic medium.

    Parameters:
        num_cells: The number of cells the tensor is constructed for.

    Returns:
        The fourth order tensor with custom fields and the matrix representation of the
        tensor for one cell.

    """
    # The various matrices serve as the foundation for building a tensor for specific
    # parameters. The cell-wise parameter value for each basis is determined by a 1D
    # array, referred to as a "field". The matrices that form the basis for different
    # parameters, along with their corresponding fields, are organized as tuples. These
    # tuples are later provided to FourthOrderTensor when constructing the tensor. As
    # the stiffness tensor for the transversely isotropic media we consider here is
    # determined by five independent material parameters, we need five matrices and five
    # fields:
    matrix_and_field_1 = (
        np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
            ],
        ),
        np.ones(num_cells),
    )

    matrix_and_field_2 = (
        np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        np.ones(num_cells) * 2,
    )

    matrix_and_field_3 = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
        ),
        np.ones(num_cells) * 3,
    )

    matrix_and_field_4 = (
        np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        np.ones(num_cells) * 4,
    )

    matrix_and_field_5 = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
            ],
        ),
        np.ones(num_cells) * 5,
    )

    matrices_and_fields = {
        "field_1_name": matrix_and_field_1,
        "field_2_name": matrix_and_field_2,
        "field_3_name": matrix_and_field_3,
        "field_4_name": matrix_and_field_4,
        "field_5_name": matrix_and_field_5,
    }

    c = pp.FourthOrderTensor(
        mu=np.zeros(num_cells),
        lmbda=np.zeros(num_cells),
        other_fields=matrices_and_fields,
    )

    # The 9x9 matrix representation of the tensor for one cell is the following:
    c_one_cell = (
        matrix_and_field_1[0]
        + 2 * matrix_and_field_2[0]
        + 3 * matrix_and_field_3[0]
        + 4 * matrix_and_field_4[0]
        + 5 * matrix_and_field_5[0]
    )
    return c, c_one_cell


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
        kxy: Array of kxy.
        kyz: Array of kyz.
        expected_diagonal: True if the tensor is expected to be diagonal, False if not.
        expected_isotropic: True if the tensor is expected to be isotropic, False if
            not.

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


@pytest.mark.parametrize(
    "second_order",
    [(True), (False)],
)
def test_restrict_to_cells(second_order):
    """Testing the restriction of a tensor to a chosen set of cells.

    The method restrict_to_cells is used to restrict e.g. the permeability tensor to a
    region within the simulation domain. This method checks that this restriction is
    done correctly.

    """
    num_cells = 16
    if second_order:
        np.random.seed(42)

        # Defining an arbitrary anisotropic second order tensor:
        kxx = np.random.uniform(0.5, 1.0, num_cells)
        kyy = np.random.uniform(0.5, 1.0, num_cells)
        kxy = np.random.uniform(0.1, 0.5, num_cells)

        tensor = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)
    else:
        tensor, _ = fourth_order_tensor_for_testing(num_cells=num_cells)

    # Define some cells which we want to restrict the tensor k to:
    active_cells = np.array([2, 1, 14, 10, 15])

    # Restrict the tensor k to the cells we chose above.
    tensor_copy = tensor.restrict_to_cells(active_cells)

    # Looping through the permeability tensors to check that the restriction is done
    # correctly.
    for ind, cell in enumerate(active_cells):
        original = tensor.values[:, :, cell]
        copied_restriction = tensor_copy.values[:, :, ind]

        assert np.array_equal(original, copied_restriction)


def test_custom_field_tensor_generation():
    """Test generation of tensors based on custom fields.

    Utilizes the function fourth_order_tensor_for_testing() to generate a tensor with
    custom fields. The tensor is assumed to be homogeneous, and the function returns
    both the tensor and a hardcoded matrix representation of the tensor for one cell.
    The matrix representation is used as a blueprint for checking if the tensor is
    constructed correctly.

    """
    num_cells = 5
    tensor, matrix = fourth_order_tensor_for_testing(num_cells=num_cells)

    # Double checking that the tensor has the expected "length".
    assert tensor.values.shape[2] == num_cells

    for i in range(num_cells):
        assert np.array_equal(tensor.values[:, :, i], matrix)
