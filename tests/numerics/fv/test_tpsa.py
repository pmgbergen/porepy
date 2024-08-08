import pytest
import porepy as pp
import numpy as np


NUM_CELLS = 2
KEYWORD = "mechanics"


# @pytest.fixture(scope="module")
def g():
    g = pp.CartGrid([NUM_CELLS, 1])
    g.nodes = np.array(
        [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
    ).T
    g.compute_geometry()
    g.face_centers[0, 3] = 1.5
    g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T
    return g


# @pytest.fixture(scope="module")
def data():
    lmbda = np.array([1, 1])
    mu = np.array([1, 2])
    C = pp.FourthOrderTensor(mu, lmbda)
    cosserat = np.array([1, 3])

    d = {
        pp.PARAMETERS: {
            KEYWORD: {"fourth_order_tensor": C, "cosserat_parameter": cosserat}
        },
        pp.DISCRETIZATION_MATRICES: {KEYWORD: {}},
    }

    return d


def set_bc(grid, d, bc_type):
    face_ind = grid.get_all_boundary_faces()
    nf = face_ind.size
    match bc_type:
        case "dir":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["dir"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["dir"])
        case "neu":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["neu"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["neu"])
        case "robin":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["rob"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["rob"])
        case _:
            raise ValueError(f"Unknown boundary condition type {bc_type}")

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_disp
    d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot


def discretize_get_matrices(grid, d):
    # Common function to discretize and return the matrices
    discr = pp.Tpsa(KEYWORD)
    discr.discretize(grid, d)
    return d[pp.DISCRETIZATION_MATRICES][KEYWORD]


def compare_matrices(
    g, matrices, known_values, target_faces_scalar, target_faces_vector
):
    for key, known in known_values.items():

        computed = matrices[key].toarray()

        if computed.shape[0] == g.num_faces * 2:
            target_faces = target_faces_vector
        else:
            target_faces = target_faces_scalar
        assert np.allclose(computed[target_faces], known)


def test_discretization_interior_cells(g, data):
    # Construct a tpsa discretization on a grid consisting of two cells, compare the
    # computed coefficients with hand-coded values.

    # This test considers only the discretization on interior cells, but we still need
    # to give some boundary values to the discretization. Assign Dirichlet conditions,
    # more or less arbitrarily.
    set_bc(g, data, "dir")
    matrices = discretize_get_matrices(g, data)

    n = np.array([2, 1])
    n_nrm = np.sqrt(5)

    # The only interior face
    target_faces_scalar = np.array([1])
    target_faces_vector = np.array([2, 3])

    # The distance from cell center to face center, projected onto the normal, is
    #   3 / (2 * sqrt(5)) for both cells.
    d_0_1 = 3 / (2 * n_nrm)
    d_1_1 = 3 / (2 * n_nrm)
    d = d_0_1 + d_1_1

    # Shear moduli
    mu_0 = 1
    mu_1 = 2
    # Weighted sum of the shear moduli
    mu_w = mu_0 / d_0_1 + mu_1 / d_1_1

    # The stress coefficient is twice the harmonic average of the two shear moduli.
    # Multiply by the length of the face (sqrt(5)).
    stress = 2 * (mu_0 * mu_1 / (d_0_1 * d_1_1) / mu_w) * n_nrm

    # The Cosserat parameter is the harmonic average of the two Cosserat parameters.
    # Multiply by the length of the face (sqrt(5)).
    cos_0 = 1
    cos_1 = 3
    cosserat = cos_0 * cos_1 / (d_0_1 * d_1_1) / (cos_0 / d_0_1 + cos_1 / d_1_1) * n_nrm

    r_0 = mu_0 / d_0_1 / mu_w
    r_1 = mu_1 / d_1_1 / mu_w

    cr_0 = 1 - r_0
    cr_1 = 1 - r_1

    known_values = {
        "stress": np.array([[-stress, 0, stress, 0], [0, -stress, 0, stress]]),
        "bound_stress": np.zeros((2, 4)),
        "stress_rotation": -np.array(
            [[cr_0 * n[1], cr_1 * n[1]], [-cr_0 * n[0], -cr_1 * n[0]]]
        ),
        "stress_total_pressure": np.array(
            [[cr_0 * n[0], cr_1 * n[0]], [cr_0 * n[1], cr_1 * n[1]]],
        ),
        "rotation_displacement": -np.array(
            [-r_0 * n[1], r_0 * n[0], -r_1 * n[1], r_1 * n[0]]
        ),
        "bound_rotation_displacement": np.zeros((1, 4)),
        "rotation_diffusion": np.array([-cosserat, cosserat]),
        "bound_rotation_diffusion": np.zeros((1, 2)),
        "solid_mass_displacement": np.array(
            [r_0 * n[0], r_0 * n[1], r_1 * n[0], r_1 * n[1]]
        ),
        "bound_mass_displacement": np.zeros((1, 4)),
        "solid_mass_total_pressure": np.array([-1 / (2 * mu_w), 1 / (2 * mu_w)])
        * n_nrm,
    }

    compare_matrices(
        g, matrices, known_values, target_faces_scalar, target_faces_vector
    )


def test_dirichlet_bcs(g, data):
    # Set Dirichlet boundary conditions on all faces, check that the implementation of
    # the boundary conditions are correct.
    set_bc(g, data, "dir")
    matrices = discretize_get_matrices(g, data)

    # We test the discretization on face 6, as this has two non-trivial components of
    # the normal vector, and in the vector from cell to face center.
    target_faces_scalar = np.array([0, 6])
    target_faces_vector = np.array([0, 1, 12, 13])

    n_0 = np.array([1, 0])
    n_0_nrm = 1
    n_6 = np.array([1, 2])
    n_6_nrm = np.sqrt(5)

    # The distance from cell center to face center, projected onto the normal.
    d_0_0 = 1
    d_1_6 = 3 / (2 * n_6_nrm)

    mu_0 = 1
    mu_1 = 2
    cos_0 = 1
    cos_1 = 3

    stress_0 = 2 * mu_0 / d_0_0 * n_0_nrm
    stress_6 = 2 * mu_1 / d_1_6 * n_6_nrm

    r_0 = 0
    r_6 = 0

    cr_0 = 1 - r_0
    cr_6 = 1 - r_6

    # EK note to self: The coefficients in bound_stress should be negative those of the
    # stress matrix so that translation results in a stress-free configuration.
    bound_stress = np.zeros((4, 14))
    bound_stress[0, 0] = -stress_0
    bound_stress[1, 1] = -stress_0
    bound_stress[2, 12] = stress_6
    bound_stress[3, 13] = stress_6

    bound_rotation_diffusion = np.zeros((2, 7))
    bound_rotation_diffusion[0, 0] = -cos_0 / d_0_0 * n_0_nrm
    bound_rotation_diffusion[1, 6] = cos_1 / d_1_6 * n_6_nrm

    bound_rotation_displacement = np.zeros((2, 14))

    bound_mass_displacement = np.zeros((2, 14))
    assert False

    known_values = {
        # Positive sign on the first two rows, since the normal vector is pointing
        # into that cell. Oposite sign on the two last rows, as the normal vector is
        # pointing out of the cell.
        "stress": np.array(
            [
                [stress_0, 0, 0, 0],
                [0, stress_0, 0, 0],
                [0, 0, -stress_6, 0],
                [0, 0, 0, -stress_6],
            ]
        ),
        "bound_stress": bound_stress,
        # Minus sign for the full expression (see paper).
        "stress_rotation": -np.array(
            [
                [cr_0 * n_0[1], 0],
                [-cr_0 * n_0[0], 0],
                [0, cr_6 * n_6[1]],
                [0, -cr_6 * n_6[0]],
            ]
        ),
        "stress_total_pressure": np.array(
            [
                [cr_0 * n_0[0], 0],
                [cr_0 * n_0[1], 0],
                [0, cr_6 * n_6[0]],
                [0, cr_6 * n_6[1]],
            ]
        ),
        "rotation_displacement": np.zeros((2, 4)),
        "bound_rotation_displacement": np.zeros((2, 14)),
        # Minus sign on the second face, since the normal vector is pointing out of the
        # cell.
        "rotation_diffusion": np.array(
            [[cos_0 / d_0_0 * n_0_nrm, 0], [0, -cos_1 / d_1_6 * n_6_nrm]]
        ),
        "bound_rotation_diffusion": bound_rotation_diffusion,
        "solid_mass_displacement": np.zeros((2, 4)),
        "bound_mass_displacement": np.zeros((2, 14)),
        "solid_mass_total_pressure": np.zeros((2, 2)),
    }

    compare_matrices(
        g, matrices, known_values, target_faces_scalar, target_faces_vector
    )


def test_neumann_bcs(g, data):
    # Set Neumann boundary conditions on all faces, check that the implementation of
    # the boundary conditions are correct.
    set_bc(g, data, "neu")
    matrices = discretize_get_matrices(g, data)

    # We test the discretization on face 6, as this has two non-trivial components of
    # the normal vector, and in the vector from cell to face center.
    target_faces_scalar = np.array([0, 6])
    target_faces_vector = np.array([0, 1, 12, 13])

    n_0 = np.array([1, 0])
    n_0_nrm = 1
    n_6 = np.array([1, 2])
    n_6_nrm = np.sqrt(5)

    # The distance from cell center to face center, projected onto the normal.
    d_0_0 = 1
    d_1_6 = 3 / (2 * n_6_nrm)

    mu_0 = 1
    mu_1 = 2
    cos_0 = 1
    cos_1 = 3

    stress_0 = 2 * mu_0 / d_0_0 * n_0_nrm
    stress_6 = 2 * mu_1 / d_1_6 * n_6_nrm

    r_0 = 1
    r_6 = 1

    # Boundary stress: The coefficients in bound_stress should be negative those of the
    # stress matrix so that translation results in a stress-free configuration.
    bound_stress = np.zeros((4, 14))
    bound_stress[0, 0] = -1
    bound_stress[1, 1] = -1
    bound_stress[2, 12] = 1
    bound_stress[3, 13] = 1

    bound_rotation_diffusion = np.zeros((2, 7))
    bound_rotation_diffusion[0, 0] = -1
    bound_rotation_diffusion[1, 6] = 1

    bound_rotation_displacement = np.zeros((2, 14))
    bound_rotation_displacement[0, 0] = -d_0_0 * n_0[1] / (2 * mu_0)
    bound_rotation_displacement[0, 1] = d_0_0 * n_0[0] / (2 * mu_0)
    bound_rotation_displacement[1, 12] = -d_1_6 * n_6[1] / (2 * mu_1)
    bound_rotation_displacement[1, 13] = d_1_6 * n_6[0] / (2 * mu_1)

    bound_mass_displacement = np.zeros((2, 14))
    bound_mass_displacement[0, 0] = d_0_0 * n_0[0] / (2 * mu_0)
    bound_mass_displacement[0, 1] = d_0_0 * n_0[1] / (2 * mu_0)
    bound_mass_displacement[1, 12] = d_1_6 * n_6[0] / (2 * mu_1)
    bound_mass_displacement[1, 13] = d_1_6 * n_6[1] / (2 * mu_1)

    known_values = {
        # The stress is prescribed, thus no contribution from the interior cells for
        # any of the stress-related matrices.
        "stress": np.zeros((4, 4)),
        "stress_rotation": np.zeros((4, 2)),
        "stress_total_pressure": np.zeros((4, 2)),
        # The boundary stress is defined above.
        "bound_stress": bound_stress,
        # The outer minus sign is part of the analytical expression. The inner minus
        # signs, and the coefficients, follow from the definition of R_k^n and the
        # coefficients of the discretization (see paper).
        "rotation_displacement": -np.array(
            [[-r_0 * n_0[1], r_0 * n_0[0], 0, 0], [0, 0, -r_6 * n_6[1], r_6 * n_6[0]]]
        ),
        "bound_rotation_displacement": bound_rotation_displacement,
        "rotation_diffusion": np.zeros((2, 2)),
        "bound_rotation_diffusion": bound_rotation_diffusion,
        "solid_mass_displacement": np.array(
            [[r_0 * n_0[0], r_0 * n_0[1], 0, 0], [0, 0, r_6 * n_6[0], r_6 * n_6[1]]]
        ),
        "bound_mass_displacement": bound_mass_displacement,
        "solid_mass_total_pressure": np.array(
            [[d_0_0 / (2 * mu_0) * n_0_nrm, 0], [0, -d_1_6 / (2 * mu_1) * n_6_nrm]]
        ),
    }

    compare_matrices(
        g, matrices, known_values, target_faces_scalar, target_faces_vector
    )


test_neumann_bcs(g(), data())


def test_robin_bcs(g):
    # Set Robin boundary conditions on all faces, check that the discretization stencil
    # for internal faces, as well as the implementation of the boundary conditions are
    # correct.
    pass


def test_mixed_bcs(g):
    # Set mixed boundary conditions (e.g. type A in one direction, B in a different
    # direction) on all faces, check that the discretization stencil for internal faces,
    # as well as the implementation of the boundary conditions are correct. Note that it
    # is not necessary to consider interaction between different types of boundary
    # conditions on different faces, since a two-point stencil does not allow for such
    # interactions.
    pass


def test_no_cosserat(g):
    # Set up a problem without Cosserat effects, check that the rotation diffusion
    # matrix is zero.
    pass


def test_cosserat_3d():
    # Set up a 3d problem, check that the rotation diffusion matrix has the right
    # dimension, and that the boundary conditions are correctly implemented. The 3d case
    # is considered for the Cosserat term only, since all other terms have negligible
    # differences between 2d and 3d.
    g = pp.CartGrid([2, 1, 1])


def test_translation(g):
    # Set boundary conditions that corresponds to a translation of the grid, check that
    # the interior cells follow the translation, and that the resulting system is
    # stress-free.
    pass
