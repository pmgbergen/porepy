import pytest
import porepy as pp
import numpy as np
import scipy.sparse as sps

NUM_CELLS = 2
KEYWORD = "mechanics"

def _rot_dim(g):
    if g.dim == 2:
        return 1
    else:
        return g.dim


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


def _set_uniform_bc(grid, d, bc_type, include_rot=True):
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
        case "rob":
            bc_disp = pp.BoundaryConditionVectorial(
                grid, faces=face_ind, cond=nf * ["rob"]
            )
            bc_rot = pp.BoundaryCondition(grid, faces=face_ind, cond=nf * ["rob"])
        case _:
            raise ValueError(f"Unknown boundary condition type {bc_type}")

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_disp
    if include_rot:
        d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot

class TestTpsaTailoredGrid:

    @pytest.fixture(scope="class")
    def g(self):
        g = pp.CartGrid([NUM_CELLS, 1])
        g.nodes = np.array(
            [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
        ).T
        g.compute_geometry()
        g.face_centers[0, 3] = 1.5
        g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T
        return g

    @pytest.fixture(scope="class")
    def data(self):
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

    def test_discretization_interior_cells(self, g, data):
        # Construct a tpsa discretization on a grid consisting of two cells, compare the
        # computed coefficients with hand-coded values.

        # This test considers only the discretization on interior cells, but we still need
        # to give some boundary values to the discretization. Assign Dirichlet conditions,
        # more or less arbitrarily.
        _set_uniform_bc(g, data, "dir")
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
            "bound_stress": np.zeros((2, 14)),
            "stress_rotation": -np.array(
                [[cr_0 * n[1], cr_1 * n[1]], [-cr_0 * n[0], -cr_1 * n[0]]]
            ),
            "stress_total_pressure": np.array(
                [[cr_0 * n[0], cr_1 * n[0]], [cr_0 * n[1], cr_1 * n[1]]],
            ),
            "rotation_displacement": -np.array(
                [-r_0 * n[1], r_0 * n[0], -r_1 * n[1], r_1 * n[0]]
            ),
            "bound_rotation_displacement": np.zeros((1, 14)),
            "rotation_diffusion": np.array([-cosserat, cosserat]),
            "bound_rotation_diffusion": np.zeros((1, 7)),
            "solid_mass_displacement": np.array(
                [r_0 * n[0], r_0 * n[1], r_1 * n[0], r_1 * n[1]]
            ),
            "bound_mass_displacement": np.zeros((1, 14)),
            "solid_mass_total_pressure": np.array([-1 / (2 * mu_w), 1 / (2 * mu_w)])
            * n_nrm,
        }

        compare_matrices(
            g, matrices, known_values, target_faces_scalar, target_faces_vector
        )


    def test_dirichlet_bcs(self, g, data):
    #def test_dirichlet_bcs(self):
    #    g = self.g()
    #    data = self.data()
        # Set Dirichlet boundary conditions on all faces, check that the implementation of
        # the boundary conditions are correct.
        _set_uniform_bc(g, data, "dir")
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
        # From the definition of \bar{R}, we get [-n[1], n[0]]. There is an additional
        # minus sign in the analytical expression, which is included in the known values.
        bound_rotation_displacement[0, 0] = cr_0 * n_0[1]
        bound_rotation_displacement[0, 1] = -cr_0 * n_0[0]
        bound_rotation_displacement[1, 12] = cr_6 * n_6[1]
        bound_rotation_displacement[1, 13] = -cr_6 * n_6[0]

        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] = cr_0 * n_0[0]
        bound_mass_displacement[0, 1] = cr_0 * n_0[1]
        bound_mass_displacement[1, 12] = cr_6 * n_6[0]
        bound_mass_displacement[1, 13] = cr_6 * n_6[1]

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
            "bound_rotation_displacement": bound_rotation_displacement,
            # Minus sign on the second face, since the normal vector is pointing out of the
            # cell.
            "rotation_diffusion": np.array(
                [[cos_0 / d_0_0 * n_0_nrm, 0], [0, -cos_1 / d_1_6 * n_6_nrm]]
            ),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.zeros((2, 4)),
            "bound_mass_displacement": bound_mass_displacement,
            "solid_mass_total_pressure": np.zeros((2, 2)),
        }

        compare_matrices(
            g, matrices, known_values, target_faces_scalar, target_faces_vector
        )


    def test_neumann_bcs(self, g, data):

        # Set Neumann boundary conditions on all faces, check that the implementation of
        # the boundary conditions are correct.
        _set_uniform_bc(g, data, "neu")
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


    def test_robin_bcs(self, g, data):
        # Set Robin boundary conditions on all faces, check that the discretization stencil
        # for internal faces, as well as the implementation of the boundary conditions are
        # correct.
        # Set Robin boundary conditions on all faces, check that the implementation of
        # the boundary conditions are correct.
        _set_uniform_bc(g, data, "rob")

        # Modify the Robin weight in the displacement boundary condition
        # Robin weights
        rw_0_x = 2
        rw_0_y = 1
        rw_6 = 1
        bc_disp = data[pp.PARAMETERS][KEYWORD]['bc']
        # Set an anisotropic weight for face 0
        bc_disp.robin_weight[0, 0, 0] = rw_0_x
        bc_disp.robin_weight[1, 1, 0] = rw_0_y
        bc_disp.robin_weight[0, 0, 6] = rw_6
        bc_disp.robin_weight[1, 1, 6] = rw_6
        

        # Robin boundaries have not yet been implemented for the rotation variable, so
        # set this to Neumann and ignore the computed values.
        bc_rot = data[pp.PARAMETERS][KEYWORD]['bc_rot']
        # This will actually set Neumann conditions also on internal faces, but that
        # should not be a problem.
        bc_rot.is_neu[:] = True
        bc_rot.is_rob[:] = False

        matrices = discretize_get_matrices(g, data)

        # We test the discretization on face 6, as this has two non-trivial components of
        # the normal vector, and in the vector from cell to face center.
        target_faces_scalar = np.array([0, 6])
        target_faces_vector = np.array([0, 1, 12, 13])

        n_0 = np.array([1, 0])
        n_0_nrm = 1
        n_6 = np.array([1, 2])
        n_6_nrm = np.sqrt(5)

        # The distance from cell center to face center, projected onto the normal. Robin
        # conditions can be interpreted as a distance of 1 / alpha, where alpha is the
        # Biot parameter in the sense of PorePy.
        
        d_0_0 = 1  # Distance from cell 0 to face 0
        # The Robin condition translated to 'distances'
        d_0_x_bound = rw_0_x  # In the x-direction, the Robin coefficient is 2
        d_0_y_bound = rw_0_y  # In the y-direction, the Robin coefficient is 1
        # The total distances
        d_0_0_x = d_0_0 + d_0_x_bound
        d_0_0_y = d_0_0 + d_0_y_bound
        # For face 6, cell 1, the distance is 2/(3 * nrm_6), the Robin coefficient is 1
        d_6_bound = 1
        d_1_6 = 3 / (2 * n_6_nrm) 
        #d_1_6 = d_1_6_int + d_6_bound

        mu_0 = 1
        mu_1 = 2
        cos_0 = 1
        cos_1 = 3

        # Stress discretization, use distances that incorporate the Robin condition
        stress_0_x = 2 * n_0_nrm * ((mu_0 / d_0_0) * rw_0_x) / (mu_0 / d_0_0 + rw_0_x)
        stress_0_y = 2 * n_0_nrm * ((mu_0 / d_0_0) * rw_0_y) / (mu_0 / d_0_0 + rw_0_y)
        stress_6 = 2 * n_6_nrm * ((mu_1 / d_1_6) * rw_6) / (mu_1 / d_1_6 + rw_6)

        # Averaging coefficient for the interior cell
        r_0_x = (mu_0 / d_0_0) / (mu_0 / d_0_0 + rw_0_x)
        r_0_y = (mu_0 / d_0_0) / (mu_0 / d_0_0 + rw_0_y)
        r_6 = (mu_1 / d_1_6) / (mu_1 / d_1_6 + rw_6)
        # And the complement
        cr_0_x = 1 - r_0_x
        cr_0_y = 1 - r_0_y
        cr_6 = 1 - r_6

        # Averaging coefficients for the boundary term
        r_0_x_bound = rw_0_x / (mu_0 / d_0_0 + rw_0_x)
        r_0_y_bound = rw_0_y / (mu_0 / d_0_0 + rw_0_y)
        r_6_bound = rw_6 / (mu_1 / d_1_6 + rw_6)
        # And the complement
        cr_0_x_bound = 1 - r_0_x_bound
        cr_0_y_bound = 1 - r_0_y_bound
        cr_6_bound = 1 - r_6_bound


        # The term delta_k^mu
        delta_0_x = 1 / (2 * (mu_0 / d_0_0 + 2))
        delta_0_y = 1 / (2 * (mu_0 / d_0_0 + 1))
        delta_6 = 1 / (2 * (mu_1 / d_1_6 + 1))


        # Boundary stress. The first term in the below expressions corresponds to the
        # Neumann part of the Robin condition, the second is the Dirichlet part. For the
        # Neumann part, the expression includes the distance measure (or rather 1 -
        # distance, since the discretization includes the complimentary averaging
        # operator).
        bound_stress = np.zeros((4, 14))
        # The first term is the 'Neumann part', second is Dirichlet. The signs of the
        # respective parts are the same as in the Dirichlet and Neumann tests.
        bound_stress[0, 0] = -cr_0_x_bound - stress_0_x
        bound_stress[1, 1] = -cr_0_y_bound - stress_0_y
        bound_stress[2, 12] = cr_6_bound + stress_6
        bound_stress[3, 13] = cr_6_bound + stress_6

        # This is set to Neumann conditions (see top of this method), so copy these
        # conditions from the relevant test
        bound_rotation_diffusion = np.zeros((2, 7))
        bound_rotation_diffusion[0, 0] = -1
        bound_rotation_diffusion[1, 6] = 1

        # From the definition of \bar{R}, we get [-n[1], n[0]]. There is an additional
        # minus sign in the analytical expression, which is included in the known values.
        # First term is the Neumann part, second is Dirichlet.
        bound_rotation_displacement = np.zeros((2, 14))
        bound_rotation_displacement[0, 0] = -n_0[1] * delta_0_x + r_0_x_bound * n_0[1]
        bound_rotation_displacement[0, 1] = n_0[0]  * delta_0_y -r_0_y_bound * n_0[0]
        bound_rotation_displacement[1, 12] = -n_6[1] * delta_6 + r_6_bound * n_6[1]
        bound_rotation_displacement[1, 13] = n_6[0] * delta_6 -r_6_bound * n_6[0]

        bound_mass_displacement = np.zeros((2, 14))
        bound_mass_displacement[0, 0] =  n_0[0] * delta_0_x
        bound_mass_displacement[0, 1] = n_0[1] * delta_0_y
        bound_mass_displacement[1, 12] = n_6[0] * delta_6
        bound_mass_displacement[1, 13] = n_6[1] * delta_6

        known_values = {
            # The stress discretization is the same as in the Dirichlet case
            "stress": np.array(
                [
                    [stress_0_x, 0, 0, 0],
                    [0, stress_0_y, 0, 0],
                    [0, 0, -stress_6, 0],
                    [0, 0, 0, -stress_6],
                ]
            ),
            "stress_rotation":  -np.array(
                [
                    [cr_0_x * n_0[1], 0],
                    [-cr_0_y * n_0[0], 0],
                    [0, cr_6 * n_6[1]],
                    [0, -cr_6 * n_6[0]],
                ]
            ),
            "stress_total_pressure": np.array(
                [
                    [cr_0_x * n_0[0], 0],
                    [cr_0_y * n_0[1], 0],
                    [0, cr_6 * n_6[0]],
                    [0, cr_6 * n_6[1]],
                ]
            ),
            # The boundary stress is defined above.
            "bound_stress": bound_stress,
            # The outer minus sign is part of the analytical expression. The inner minus
            # signs, and the coefficients, follow from the definition of R_k^n and the
            # coefficients of the discretization (see paper).
            "rotation_displacement": -np.array(
                [[-r_0_x * n_0[1], r_0_y * n_0[0], 0, 0], [0, 0, -r_6 * n_6[1], r_6 * n_6[0]]]
            ),
            "bound_rotation_displacement": bound_rotation_displacement,
            "rotation_diffusion": np.zeros((2, 2)),
            "bound_rotation_diffusion": bound_rotation_diffusion,
            "solid_mass_displacement": np.array(
                [[r_0_x * n_0[0], r_0_y * n_0[1], 0, 0], [0, 0, r_6 * n_6[0], r_6 * n_6[1]]]
            ),
            "bound_mass_displacement": bound_mass_displacement,
            "solid_mass_total_pressure": np.array(
                [[d_0_0 / (2 * mu_0) * n_0_nrm, 0], [0, -d_1_6 / (2 * mu_1) * n_6_nrm]]
            ),
        }

        compare_matrices(
            g, matrices, known_values, target_faces_scalar, target_faces_vector
        )





    def test_mixed_bcs(self, g):
        # Set mixed boundary conditions (e.g. type A in one direction, B in a different
        # direction) on all faces, check that the discretization stencil for internal faces,
        # as well as the implementation of the boundary conditions are correct. Note that it
        # is not necessary to consider interaction between different types of boundary
        # conditions on different faces, since a two-point stencil does not allow for such
        # interactions.
        pass


#TestTpsaTailoredGrid().test_dirichlet_bcs()

def test_no_cosserat():
    # Set up a problem without Cosserat effects, check that the rotation diffusion
    # matrix is zero.
    g = pp.CartGrid([2, 2])
    g.compute_geometry()

    d = _set_uniform_parameters(g)
    bf = g.get_all_boundary_faces()
    d[pp.PARAMETERS][KEYWORD]["bc"] = pp.BoundaryConditionVectorial(
        g, faces=bf, cond=bf.size * ["dir"]
    )

    # Discretize, assemble matrices
    matrices = discretize_get_matrices(g, d)

    assert np.allclose(matrices["rotation_diffusion"].toarray(), 0)
    assert np.allclose(matrices["bound_rotation_diffusion"].toarray(), 0)


def test_cosserat_3d():
    # Set up a 3d problem, check that the rotation diffusion matrix has the right
    # dimension, and that the boundary conditions are correctly implemented. The 3d case
    # is considered for the Cosserat term only, since all other terms have negligible
    # differences between 2d and 3d.
    g = pp.CartGrid([2, 1, 1])
    g.compute_geometry()
    d = _set_uniform_parameters(g)

    d[pp.PARAMETERS][KEYWORD]["cosserat_parameter"] = np.ones(g.num_cells)
    bf = g.get_all_boundary_faces()
    d[pp.PARAMETERS][KEYWORD]["bc"] = pp.BoundaryConditionVectorial(
        g, faces=bf, cond=bf.size * ["dir"]
    )

    bc_rot = pp.BoundaryConditionVectorial(g, faces=bf, cond=bf.size * ["dir"]    )
    bc_rot.is_dir[:, 0] = False
    bc_rot.is_neu[:, 0] = True
    d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot

    # Discretize, assemble matrices
    matrices = discretize_get_matrices(g, d)

    rot_mat = matrices["rotation_diffusion"]
    # Check size
    assert rot_mat.shape == (g.num_faces * 3, g.num_cells * 3)

    known_values = np.zeros((12, 6))
    
    # Face 0 has a Neumann condition, thus the coefficients are zero
    # Check coefficients on the inner face
    known_values[3, 0] = -1
    known_values[4, 1] = -1
    known_values[5, 2] = -1
    known_values[3, 3] = 1
    known_values[4, 4] = 1
    known_values[5, 5] = 1
    # Check coefficients on face 2, which has a Dirichlet condition
    known_values[6, 3] = -2
    known_values[7, 4] = -2
    known_values[8, 5] = -2
    # Face 3 has a Dirichlet condition, but with inwards pointing normal vector, thus the
    # coefficients should have negative sign.
    known_values[9, 0] = 2
    known_values[10, 1] = 2
    known_values[11, 2] = 2

    assert np.allclose(rot_mat[:12].toarray(), known_values)

    bc_rot = matrices["bound_rotation_diffusion"]
    known_values = np.zeros((12, 33))
    # Neumann condition on face 0
    known_values[0, 0] = -1
    known_values[1, 1] = -1
    known_values[2, 2] = -1
    # Dirichlet condition, outwards pointing normal vector on face 2
    known_values[6, 6] = 2
    known_values[7, 7] = 2
    known_values[8, 8] = 2
    # Dirichlet condition, inwards pointing normal vector on face 3
    known_values[9, 9] = -2
    known_values[10, 10] = -2
    known_values[11, 11] = -2

    assert np.allclose(bc_rot[:12].toarray(), known_values)


def _set_uniform_bc_values(g, bc_type):
    if g.dim == 2:
        val = np.array([1, -2])
    else:
        val = np.array([1, -2, 3])

    bc_values = np.zeros((g.dim, g.num_faces))
    bf = g.get_boundary_faces()

    sgn, _ = g.signs_and_cells_of_boundary_faces(bf)

    # Unit normal vectors
    n = g.face_normals / np.linalg.norm(g.face_normals, axis=0)
    for d in range(g.dim):
        if bc_type == "dir":
            bc_values[d, bf] = val[d]
        elif bc_type == "neu":
            bc_values[d, bf] = val[d] * sgn * n[d, bf]

    return bc_values, val

def _set_uniform_parameters(g):
    # EK note to self: For the 3d test of compression with a unit value for the elastic
    # moduli, there was an unexpected sign in the direction of the displacement, but
    # this changed when moving to a stiffer material. This does not sound unreasonable,
    # but is hereby noted for future reference.
    e = 100 * np.ones(g.num_cells)
    C = pp.FourthOrderTensor(e, e)

    d = {
        pp.PARAMETERS: {KEYWORD: {"fourth_order_tensor": C}},
        pp.DISCRETIZATION_MATRICES: {KEYWORD: {}},
    }
    return d

def _assemble_matrices(matrices, g):
    if g.dim == 2:
        n_rot_face = g.num_faces
        n_rot_cell = g.num_cells
        div_rot = pp.fvutils.scalar_divergence(g)
    else:
        n_rot_face = g.num_faces * g.dim
        n_rot_cell = g.num_cells * g.dim
        div_rot = pp.fvutils.vector_divergence(g)

    flux = sps.block_array(
        [
            [
                matrices["stress"],
                matrices["stress_rotation"],
                matrices["stress_total_pressure"],
            ],
            [
                matrices["rotation_displacement"],
                matrices["rotation_diffusion"],
                sps.csr_array((n_rot_face, g.num_cells)),
            ],
            [
                matrices["solid_mass_displacement"],
                sps.csr_array((g.num_faces, n_rot_cell)),
                matrices["solid_mass_total_pressure"],
            ],
        ],
    )

    rhs_matrix = sps.bmat(
        [
            [matrices["bound_stress"], sps.csr_array((g.num_faces * g.dim, n_rot_face))],
            [matrices["bound_rotation_displacement"], sps.csr_matrix((n_rot_face, n_rot_face))],
            [matrices["bound_mass_displacement"], sps.csr_matrix((g.num_faces, n_rot_face))],
        ]
    )

    div = sps.block_diag(
        [
            pp.fvutils.vector_divergence(g),
            div_rot,
            pp.fvutils.scalar_divergence(g),
        ], format="csr"
    )       

    accum = sps.block_diag(
        [
            sps.csr_matrix((g.num_cells * g.dim, g.num_cells * g.dim)),
            sps.eye(n_rot_cell),
            sps.eye(g.num_cells),
        ], format="csr"
    )

    return flux, rhs_matrix, div, accum



def _set_bc_by_direction(g, d, type_south, type_east, type_north, type_west, type_bottom=None, type_top=None):
    face_ind = g.get_all_boundary_faces()
    nf = face_ind.size

    min_coord, max_coord = pp.domain.grid_minmax_coordinates(g)

    south = np.where(g.face_centers[1, face_ind] == min_coord[1])
    east = np.where(g.face_centers[0, face_ind] == max_coord[0])
    north = np.where(g.face_centers[1, face_ind] == max_coord[1])
    west = np.where(g.face_centers[0, face_ind] == min_coord[0])
    if g.dim == 3:
        bottom = np.where(g.face_centers[2, face_ind] == min_coord[2])
        top = np.where(g.face_centers[2, face_ind] == max_coord[2])

    bc_str = np.zeros(nf, dtype='object')

    directions = [south, east, north, west]
    types = [type_south, type_east, type_north, type_west]
    if g.dim == 3:
        directions += [bottom, top]
        types += [type_bottom, type_top]

    for fi, bc_type in zip(directions, types):

        match bc_type:
            case "dir":
                bc_str[fi] = "dir"
            case "neu":
                bc_str[fi] = "neu"
            case "robin":
                bc_str[fi] = "rob"
            case _:
                raise ValueError(f"Unknown boundary condition type {bc_type}")

    bc_disp = pp.BoundaryConditionVectorial(
        g, faces=face_ind, cond=bc_str
    )
    if g.dim == 2:
        bc_rot = pp.BoundaryCondition(g, faces=face_ind, cond=bc_str)
    else:
        bc_rot = pp.BoundaryConditionVectorial(
            g, faces=face_ind, cond=bc_str
        )

    d[pp.PARAMETERS][KEYWORD]["bc"] = bc_disp
    d[pp.PARAMETERS][KEYWORD]["bc_rot"] = bc_rot

    values = np.zeros((g.dim, nf))
    values[1, south] = 0.1
    values[0, east] = -0.1
    if g.dim == 3:
        values[2, bottom] = 0.1
    
    bc_val = np.zeros((g.dim, g.num_faces))
    bc_val[:, face_ind] = values
    return bc_val


@pytest.mark.parametrize("g", [pp.CartGrid([2, 2]), pp.CartGrid([2, 2, 2])])
@pytest.mark.parametrize("driving_bc_type", ["dir", "neu"])
@pytest.mark.parametrize("extension", [False, True])
def test_compression(g: pp.Grid, driving_bc_type: str, extension: bool):
    # Assign a compressive force on the south and east faces, with fixed west and north
    # boundaries. Check that this results in displacement in the negative x-direction
    # and positive y-direction. The total pressure should be negative, while EK cannot
    # surmise the correct sign of the rotation by physical intuition. The grid is
    # assumed to be Cartesian (or else the discretization is inconsistent, and anything
    # can happen), in which case the face normal vectors will point into the domain on
    # the south boundary, out of the domain on the east boundary.
    g.compute_geometry()

    d = _set_uniform_parameters(g)
    if g.dim == 2:
        dir_list = [driving_bc_type, driving_bc_type, "dir", "dir"]
    else:
        dir_list = [driving_bc_type, driving_bc_type, "dir", "dir", driving_bc_type, "dir"]
    bc_values = _set_bc_by_direction(g, d, *dir_list)

    #bc_values[0] *= 0

    if extension:
        bc_values *= -1

    # Discretize, assemble matrices
    matrices = discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)

    if g.dim == 2:
        n_rot_face = g.num_faces
        rot_dim = 1
    else:
        n_rot_face = g.num_faces * g.dim
        rot_dim = g.dim

    # Boundary values, map to right-hand side
    bound_vec = np.hstack((bc_values.ravel("F"), np.zeros(n_rot_face)))
    b = -div @ rhs_matrix @ bound_vec

    A = div @ flux - accum

    x = sps.linalg.spsolve(A, b)

    if extension:
        assert np.all(x[:g.dim * g.num_cells: g.dim] > 0)
        assert np.all(x[1 :g.dim * g.num_cells: g.dim] < 0)
        if g.dim == 3:
            assert np.all(x[2 :g.dim * g.num_cells: g.dim] < 0)

        assert np.all(x[(g.dim + rot_dim) * g.num_cells:] > 0)
    else:
        assert np.all(x[:g.dim * g.num_cells: g.dim]< 0)
        assert np.all(x[1 :g.dim * g.num_cells: g.dim]> 0)
        if g.dim == 3:
            assert np.all(x[2 :g.dim * g.num_cells: g.dim]> 0)

        assert np.all(x[(g.dim + rot_dim) * g.num_cells:] < 0)




def _test_uniform_force_bc(g):
    # Set a uniform force on the grid, check that the resulting stress is as expected.
    g.compute_geometry()

    d = _set_uniform_parameters(g)

    # Set type and values of boundary conditions
    _set_uniform_bc(g, d, "neu")
    bc_values, disp = _set_uniform_bc_values(g, 'neu')
    bc_values[1] *= 0

    # Discretize, assemble matrices
    matrices = discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)

    if g.dim == 2:
        n_rot_face = g.num_faces
        rot_dim = 1
    else:
        n_rot_face = g.num_faces * g.dim
        rot_dim = g.dim

    # Boundary values, map to right-hand side
    bound_vec = np.hstack((bc_values.ravel("F"), np.zeros(n_rot_face)))
    b = -div @ rhs_matrix @ bound_vec

    assert np.sum(b[:g.dim * g.num_cells]) == 0

    A = (div @ flux - accum)

    C = np.zeros((g.dim, (g.dim + rot_dim + 1) * g.num_cells))
    C[0, :g.dim * g.num_cells: g.dim] = 1
    C[1, 1 :g.dim * g.num_cells: g.dim] = 1
    if g.dim == 3:
        C[2, 2 :g.dim * g.num_cells: g.dim] = 1

    D = np.zeros((g.dim, g.dim))
    D[0, 0] = -1 / g.num_cells
    D[1, 1] = -1 / g.num_cells
    if g.dim == 3:
        D[2, 2] = -1 / g.num_cells

    Ct = np.zeros(C.T.shape)
    

    M = np.block([[A.toarray(), Ct], [C, D]])

    b_ext = np.hstack((b, np.zeros(g.dim)))

    np.linalg.solve(M, b_ext)
    x = sps.linalg.spsolve(A, b)



    assert np.allclose(x[:g.dim * g.num_cells: g.dim], x[0])
    assert np.allclose(x[1 :g.dim * g.num_cells: g.dim], x[1])
    if g.dim == 3:
        assert np.allclose(x[2 :g.dim * g.num_cells: g.dim], x[2])

    # Both the rotation and the total pressure should be zero
    assert np.allclose(x[g.dim * g.num_cells:], 0)

@pytest.mark.parametrize("g", [pp.CartGrid([2, 2]), pp.CartGrid([2, 2, 2])])
def test_translation(g):
    # Set boundary conditions that corresponds to a translation of the grid, check that
    # the interior cells follow the translation, and that the resulting system is
    # stress-free.
    g.compute_geometry()

    d = _set_uniform_parameters(g)

    # Set type and values of boundary conditions. No rotation.
    _set_uniform_bc(g, d, "dir", include_rot=False)
    bc_values, disp = _set_uniform_bc_values(g, 'dir')

    # Discretize, assemble matrices
    matrices = discretize_get_matrices(g, d)
    flux, rhs_matrix, div, accum = _assemble_matrices(matrices, g)

    if g.dim == 2:
        n_rot_face = g.num_faces
    else:
        n_rot_face = g.num_faces * g.dim

    # Boundary values, map to right-hand side
    bound_vec = np.hstack((bc_values.ravel("F"), np.zeros(n_rot_face)))
    b = -div @ rhs_matrix @ bound_vec

    # Check: Uniform translation should result in a stress-free state on all faces. This
    # is not the case for rotation and solid mass, which are only zero when integrated
    # over individual cells. This is checked below, after having solved the system.
    disp_cells = np.zeros(g.num_cells * g.dim)
    disp_cells[:: g.dim] = disp[0]
    disp_cells[1 :: g.dim] = disp[1]
    if g.dim == 3:
        disp_cells[2 :: g.dim] = disp[2]
    v = matrices['stress'] @ disp_cells + matrices["bound_stress"] @ bc_values.ravel("F")
    assert np.allclose(v, 0)

    A = div @ flux - accum

    x = sps.linalg.spsolve(A, b)

    assert np.allclose(x[:g.dim * g.num_cells: g.dim], disp[0])
    assert np.allclose(x[1 :g.dim * g.num_cells: g.dim], disp[1])
    if g.dim == 3:
        assert np.allclose(x[2 :g.dim * g.num_cells: g.dim], disp[2])

    # Both the rotation and the total pressure should be zero
    assert np.allclose(x[g.dim * g.num_cells:], 0)

#test_cosserat_3d()
#test_compression(pp.CartGrid([2, 2]), 'dir')
#test_compression(pp.CartGrid([2, 2]), 'neu')
#test_uniform_force_bc(pp.CartGrid([2, 2]))
#test_dirichlet_bcs(g(), data())
#test_translation(pp.CartGrid([3, 3]))
#test_neumann_bcs(g(), data())
