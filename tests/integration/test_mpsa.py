""" Various integration tests for mpsa methods. Contains the classes

    TestMpsaExactReproduction: Cases where mpsa should be exact, e.g. uniform strain.
    TestUpdateMpsaDiscretization: Partial updates of the discretization
    MpsaReconstructBoundaryDisplacement: Check discretization for recovery of boundary values.
    TestMpsaDiscretizeAssemble: Check the discretize and assemble_matrix_rhs functions
    TestMpsaBoundRhs: Checks of method to create rhs vector for local systems
    TestMpsaRotation: Various checks of local rotations of coornidate systems.
    RobinBoundTest: Boundary conditions of the Robin type
    TestAsymmetricNeumann: Neumann conditions.

"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp
from tests.integration import setup_grids_mpfa_mpsa_tests as setup_grids


def setup_stiffness(g, mu=1, l=1):
    mu = np.ones(g.num_cells) * mu
    l = np.ones(g.num_cells) * l
    return pp.FourthOrderTensor(mu, l)


class TestMpsaExactReproduction(unittest.TestCase):
    # Test that the discretization reproduces the expected behavior for uniform strain,
    # homogeneous conditions and other cases where the method should be exact
    def test_uniform_strain(self):
        g_list = setup_grids.setup_2d()

        for g in g_list:
            bound_faces = np.argwhere(
                np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1
            )
            bound = pp.BoundaryConditionVectorial(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            mu = 1
            l = 1
            constit = setup_stiffness(g, mu, l)

            xc = g.cell_centers
            xf = g.face_centers

            gx = np.random.rand(1)
            gy = np.random.rand(1)

            dc_x = np.sum(xc * gx, axis=0)
            dc_y = np.sum(xc * gy, axis=0)
            df_x = np.sum(xf * gx, axis=0)
            df_y = np.sum(xf * gy, axis=0)

            d_bound = np.zeros((g.dim, g.num_faces))

            d_bound[0, bound.is_dir[0]] = df_x[bound.is_dir[0]]
            d_bound[1, bound.is_dir[1]] = df_y[bound.is_dir[1]]

            bc_values = d_bound.ravel("F")

            keyword = "mechanics"

            specified_data = {
                "fourth_order_tensor": constit,
                "bc": bound,
                "inverter": "python",
                "bc_values": bc_values,
            }
            data = pp.initialize_default_data(
                g, {}, keyword, specified_parameters=specified_data
            )

            discr = pp.Mpsa(keyword)
            discr.discretize(g, data)
            A, b = discr.assemble_matrix_rhs(g, data)

            d = np.linalg.solve(A.A, b)

            stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
            bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_stress_matrix_key
            ]
            traction = stress * d + bound_stress * d_bound.ravel("F")

            s_xx = (2 * mu + l) * gx + l * gy
            s_xy = mu * (gx + gy)
            s_yx = mu * (gx + gy)
            s_yy = (2 * mu + l) * gy + l * gx

            n = g.face_normals
            traction_ex_x = s_xx * n[0] + s_xy * n[1]
            traction_ex_y = s_yx * n[0] + s_yy * n[1]

            self.assertTrue(np.max(np.abs(d[::2] - dc_x)) < 1e-8)
            self.assertTrue(np.max(np.abs(d[1::2] - dc_y)) < 1e-8)
            self.assertTrue(np.max(np.abs(traction[::2] - traction_ex_x)) < 1e-8)
            self.assertTrue(np.max(np.abs(traction[1::2] - traction_ex_y)) < 1e-8)

    def test_uniform_displacement(self):

        g_list = setup_grids.setup_2d()

        for g in g_list:
            bound_faces = np.argwhere(
                np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1
            )
            bound = pp.BoundaryConditionVectorial(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            constit = setup_stiffness(g)

            d_x = np.random.rand(1)
            d_y = np.random.rand(1)
            d_bound = np.zeros((g.dim, g.num_faces))
            d_bound[0, bound.is_dir[0]] = d_x
            d_bound[1, bound.is_dir[1]] = d_y

            bc_values = d_bound.ravel("F")

            keyword = "mechanics"

            specified_data = {
                "fourth_order_tensor": constit,
                "bc": bound,
                "inverter": "python",
                "bc_values": bc_values,
            }
            data = pp.initialize_default_data(
                g, {}, keyword, specified_parameters=specified_data
            )

            discr = pp.Mpsa(keyword)
            discr.discretize(g, data)
            A, b = discr.assemble_matrix_rhs(g, data)

            d = np.linalg.solve(A.A, b)

            stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
            bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_stress_matrix_key
            ]

            traction = stress * d + bound_stress * d_bound.ravel("F")

            self.assertTrue(np.max(np.abs(d[::2] - d_x)) < 1e-8)
            self.assertTrue(np.max(np.abs(d[1::2] - d_y)) < 1e-8)
            self.assertTrue(np.max(np.abs(traction)) < 1e-8)

    def test_uniform_displacement_neumann(self):
        physdims = [1, 1]
        g_size = [4, 8]
        g_list = [pp.CartGrid([n, n], physdims=physdims) for n in g_size]
        [g.compute_geometry() for g in g_list]
        for g in g_list:
            bot = np.ravel(np.argwhere(g.face_centers[1, :] < 1e-10))
            left = np.ravel(np.argwhere(g.face_centers[0, :] < 1e-10))
            dir_faces = np.hstack((left, bot))
            bound = pp.BoundaryConditionVectorial(
                g, dir_faces.ravel("F"), ["dir"] * dir_faces.size
            )
            constit = setup_stiffness(g)
            d_x = np.random.rand(1)
            d_y = np.random.rand(1)
            d_bound = np.zeros((g.dim, g.num_faces))

            d_bound[0, bound.is_dir[0]] = d_x
            d_bound[1, bound.is_dir[1]] = d_y

            bc_values = d_bound.ravel("F")

            keyword = "mechanics"

            specified_data = {
                "fourth_order_tensor": constit,
                "bc": bound,
                "inverter": "python",
                "bc_values": bc_values,
            }
            data = pp.initialize_default_data(
                g, {}, keyword, specified_parameters=specified_data
            )

            discr = pp.Mpsa(keyword)
            discr.discretize(g, data)
            A, b = discr.assemble_matrix_rhs(g, data)

            d = np.linalg.solve(A.A, b)

            stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
            bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_stress_matrix_key
            ]

            traction = stress * d + bound_stress * d_bound.ravel("F")
            self.assertTrue(np.max(np.abs(d[::2] - d_x)) < 1e-8)
            self.assertTrue(np.max(np.abs(d[1::2] - d_y)) < 1e-8)
            self.assertTrue(np.max(np.abs(traction)) < 1e-8)

    def test_conservation_of_momentum(self):
        pts = np.random.rand(3, 9)
        corners = [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
        pts = np.hstack((corners, pts))
        gt = pp.TetrahedralGrid(pts)
        gc = pp.CartGrid([3, 3, 3], physdims=[1, 1, 1])
        g_list = [gt, gc]
        [g.compute_geometry() for g in g_list]
        for g in g_list:
            g.compute_geometry()
            bot = np.ravel(np.argwhere(g.face_centers[1, :] < 1e-10))
            left = np.ravel(np.argwhere(g.face_centers[0, :] < 1e-10))
            dir_faces = np.hstack((left, bot))
            bound = pp.BoundaryConditionVectorial(
                g, dir_faces.ravel("F"), ["dir"] * dir_faces.size
            )
            constit = setup_stiffness(g)

            bndr = g.get_all_boundary_faces()
            d_x = np.random.rand(bndr.size)
            d_y = np.random.rand(bndr.size)
            d_bound = np.zeros((g.dim, g.num_faces))
            d_bound[0, bndr] = d_x
            d_bound[1, bndr] = d_y

            bc_values = d_bound.ravel("F")

            keyword = "mechanics"

            specified_data = {
                "fourth_order_tensor": constit,
                "bc": bound,
                "inverter": "python",
                "bc_values": bc_values,
            }
            data = pp.initialize_default_data(
                g, {}, keyword, specified_parameters=specified_data
            )

            discr = pp.Mpsa(keyword)
            discr.discretize(g, data)
            A, b = discr.assemble_matrix_rhs(g, data)

            d = np.linalg.solve(A.A, b)

            stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
            bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_stress_matrix_key
            ]
            traction = stress * d + bound_stress * d_bound.ravel("F")
            traction_2d = traction.reshape((g.dim, -1), order="F")
            for cell in range(g.num_cells):
                fid, _, sgn = sps.find(g.cell_faces[:, cell])
                self.assertTrue(
                    np.all(np.sum(traction_2d[:, fid] * sgn, axis=1) < 1e-10)
                )


class TestUpdateMpsaDiscretization(unittest.TestCase):
    """
    Class for testing updating the discretization, including the reconstruction
    of gradient displacements. Given a discretization we want
    to rediscretize parts of the domain. This will typically be a change of boundary
    conditions, fracture growth, or a change in aperture.
    """

    def test_no_change_input(self):
        """
        The input matrices should not be changed
        """
        g = pp.CartGrid([4, 4], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        keyword = "mechanics"
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        stress_old = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ].copy()
        bound_stress_old = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ].copy()
        hf_cell_old = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ].copy()
        hf_bound_old = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ].copy()

        # Update should not change anything
        faces = np.array([0, 4, 5, 6])
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = faces

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_old).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_old).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_old).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_old).data, 0))

    def test_cart_2d(self):
        """
        When not changing the parameters the output should equal the input
        """
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        keyword = "mechanics"
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)
        stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell_full = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound_full = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        # Update should not change anything
        faces = np.array([0, 3])
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = faces

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_full).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_full).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_full).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_full).data, 0))

    def test_changing_bc(self):
        """
        We test that we can change the boundary condition
        """
        g = pp.StructuredTriangleGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        # Neumann conditions everywhere
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        keyword = "mechanics"
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        # Now change the type of boundary condition on one face
        faces = 0  # g.get_all_boundary_faces()
        bc.is_dir[:, faces] = True
        bc.is_neu[bc.is_dir] = False

        # Full discretization
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        # New data dictionary,
        data_d_full = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )
        # Full discretization of the new problem
        discr.discretize(g, data_d_full)
        stress_dir = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        bound_stress_dir = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell_dir = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound_dir = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = np.array([faces])

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_dir).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_dir).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_dir).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_dir).data, 0))

    def test_changing_bc_by_cells(self):
        """
        Test that we can change the boundary condition by specifying the boundary cells
        """
        g = pp.StructuredTetrahedralGrid([2, 2, 2], physdims=(1, 1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        keyword = "mechanics"
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        faces = g.face_centers[2] < 1e-10

        bc.is_rob[:, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Partiall should give same result as full
        cells = np.argwhere(g.cell_faces[faces, :])[:, 1].ravel()
        cells = np.unique(cells)

        # Full discretization
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        # New data dictionary,
        data_d_full = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )
        # Full discretization of the new problem
        discr.discretize(g, data_d_full)
        stress_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        bound_stress_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_cells"] = np.array([cells])

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_rob).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_rob).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_rob).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_rob).data, 0))

    def test_mixed_bc(self):
        """
        We test that we can change the boundary condition in given direction
        """
        g = pp.StructuredTriangleGrid([2, 2], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        keyword = "mechanics"
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        faces = np.where(g.face_centers[0] > 1 - 1e-10)[0]

        bc.is_rob[1, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Full discretization
        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}
        # New data dictionary,
        data_d_full = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        # Go back to the old data dictionary, update a single face
        data[pp.PARAMETERS][keyword]["update_discretization"] = True
        data[pp.PARAMETERS][keyword]["specified_faces"] = np.array([faces])

        # Full discretization of the new problem
        discr.discretize(g, data_d_full)
        stress_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        bound_stress_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacement_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_rob).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_rob).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_rob).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_rob).data, 0))


class MpsaReconstructBoundaryDisplacement(unittest.TestCase):
    def test_cart_2d(self):
        """
        Test that mpsa gives out the correct matrices for
        reconstruction of the displacement at the faces
        """
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[2, 2])
        g.compute_geometry()

        lam = np.array([1])
        mu = np.array([2])
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        keyword = "mechanics"

        discr = pp.Mpsa(keyword)

        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        hf2f = pp.fvutils.map_hf_2_f(sd=g)
        num_subfaces = hf2f.sum(axis=1).A.ravel()
        scaling = sps.dia_matrix(
            (1.0 / num_subfaces, 0), shape=(hf2f.shape[0], hf2f.shape[0])
        )

        hf2f = (scaling * hf2f).toarray()

        grad_bound_known = np.array(
            [
                [0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.10416667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02083333],
                [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.02083333, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333],
                [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                [-0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.02083333, 0.0, 0.0, 0.10416667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.02083333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10416667],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, -0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667],
            ]
        )

        grad_cell_known = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        self.assertTrue(np.all(np.abs(grad_bound - hf2f.dot(grad_bound_known)) < 1e-7))
        self.assertTrue(np.all(np.abs(grad_cell - hf2f.dot(grad_cell_known)) < 1e-12))

    def test_simplex_3d_dirichlet(self):
        """
        Test that we retrieve a linear solution exactly
        """
        nx = 2
        ny = 2
        nz = 2
        g = pp.StructuredTetrahedralGrid([nx, ny, nz], physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        lam = np.ones(g.num_cells)
        mu = np.ones(g.num_cells)
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False

        x0 = np.array([[1, 2, 3]]).T
        u_b = g.face_centers + x0

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        bc_val = u_b.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * u_b.ravel("F")).reshape(
            (g.dim, -1), order="F"
        )

        U = U.reshape((g.dim, -1), order="F")

        self.assertTrue(np.all(np.abs(U - g.cell_centers - x0) < 1e-10))
        self.assertTrue(np.all(np.abs(U_f - g.face_centers - x0) < 1e-10))

    def test_simplex_3d_boundary(self):
        """
        Even if we do not get exact solution at interiour we should be able to
        retrieve the boundary conditions
        """
        nx = 2
        ny = 2
        nz = 2
        g = pp.StructuredTetrahedralGrid([nx, ny, nz], physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        lam = 10 * np.random.rand(g.num_cells)
        mu = 10 * np.random.rand(g.num_cells)
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        dir_ind = g.get_all_boundary_faces()[[0, 2, 5, 8, 10, 13, 15, 21]]
        bc.is_dir[:, dir_ind] = True
        bc.is_neu[bc.is_dir] = False

        u_b = np.random.randn(g.face_centers.shape[0], g.face_centers.shape[1])

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        bc_val = u_b.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacement_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacement_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * u_b.ravel("F")).reshape(
            (g.dim, -1), order="F"
        )

        self.assertTrue(np.all(np.abs(U_f[:, dir_ind] - u_b[:, dir_ind]) < 1e-10))


class TestMpsaDiscretizeAssembly(unittest.TestCase):
    # Test of discretization and assembly
    def test_matrix_rhs(self):
        g_list = setup_grids.setup_2d()
        kw = "mechanics"
        for g in g_list:
            solver = pp.Mpsa(kw)
            data = pp.initialize_default_data(g, {}, kw)

            solver.discretize(g, data)
            A, b = solver.assemble_matrix_rhs(g, data)
            self.assertTrue(
                np.all(A.shape == (g.dim * g.num_cells, g.dim * g.num_cells))
            )
            self.assertTrue(b.size == g.dim * g.num_cells)

    def test_matrix_rhs_no_disc(self):
        g_list = setup_grids.setup_2d()
        kw = "mechanics"
        for g in g_list:
            solver = pp.Mpsa(kw)
            data = pp.initialize_default_data(g, {}, kw)
            cell_dof = g.dim * g.num_cells
            face_dof = g.dim * g.num_faces
            stress = sps.csc_matrix((face_dof, cell_dof))
            bound_stress = sps.csc_matrix((face_dof, face_dof))
            data[pp.DISCRETIZATION_MATRICES][kw] = {
                "stress": stress,
                "bound_stress": bound_stress,
            }

            A, b = solver.assemble_matrix_rhs(g, data)
            self.assertTrue(np.sum(A != 0) == 0)
            self.assertTrue(np.all(b == 0))


class TestMpsaBoundRhs(unittest.TestCase):
    """
    Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs
    for handling boundary conditions expressed in a vectorial form
    """

    def test_neu(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g)
        self.run_test(g, basis, bc)

    def test_dir(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "dir")
        self.run_test(g, basis, bc)

    def test_mix(self):
        nx = 2
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        basis = np.random.rand(g.dim, g.dim, g.num_faces)

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_neu[:] = False

        bc.is_dir[0, left] = True
        bc.is_neu[1, left] = True

        bc.is_rob[0, right] = True
        bc.is_dir[1, right] = True

        bc.is_neu[0, bot] = True
        bc.is_rob[1, bot] = True

        bc.is_rob[0, top] = True
        bc.is_dir[1, top] = True

        self.run_test(g, basis, bc)

    def run_test(self, g, basis, bc):
        g.compute_geometry()
        g = true_2d(g)

        st = pp.fvutils.SubcellTopology(g)
        bc_sub = pp.fvutils.boundary_to_sub_boundary(bc, st)
        be = pp.fvutils.ExcludeBoundaries(st, bc_sub, g.dim)

        bound_rhs = pp.Mpsa("")._create_bound_rhs(bc_sub, be, st, g, True)

        bc.basis = basis
        bc_sub = pp.fvutils.boundary_to_sub_boundary(bc, st)
        be = pp.fvutils.ExcludeBoundaries(st, bc_sub, g.dim)
        bound_rhs_b = pp.Mpsa("")._create_bound_rhs(bc_sub, be, st, g, True)

        # rhs should not be affected by basis transform
        self.assertTrue(np.allclose(bound_rhs_b.A, bound_rhs.A))


class TestMpsaRotation(unittest.TestCase):
    """
    Rotating the basis should not change the answer. This unittest test that Mpsa
    with and without change of basis gives the same answer
    """

    def test_dir(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "dir")
        self.run_test(g, basis, bc)

    def test_rob(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "rob")
        self.run_test(g, basis, bc)

    def test_neu(self):
        g = pp.StructuredTriangleGrid([2, 2])
        basis = np.random.rand(g.dim, g.dim, g.num_faces)
        bc = pp.BoundaryConditionVectorial(g, g.get_all_boundary_faces(), "neu")
        # Add a Dirichlet condition so the system is well defined
        bc.is_dir[:, g.get_boundary_faces()[0]] = True
        bc.is_neu[bc.is_dir] = False
        self.run_test(g, basis, bc)

    def test_flip_axis(self):
        """
        We want to test the rotation of mixed conditions, but this is not so easy to
        compare because rotating the basis will change the type of boundary condition
        that is applied in the different directions. This test overcomes that by flipping
        the x- and y-axis and the flipping the boundary conditions also.
        """
        g = pp.CartGrid([2, 2], [1, 1])
        g.compute_geometry()
        nf = g.num_faces
        basis = np.array([[[0] * nf, [1] * nf], [[1] * nf, [0] * nf]])
        bc = pp.BoundaryConditionVectorial(g)
        west = g.face_centers[0] < 1e-7
        south = g.face_centers[1] < 1e-7
        north = g.face_centers[1] > 1 - 1e-7
        east = g.face_centers[0] > 1 - 1e-7
        bc.is_dir[0, west] = True
        bc.is_rob[1, west] = True
        bc.is_rob[0, north] = True
        bc.is_neu[1, north] = True
        bc.is_dir[0, south] = True
        bc.is_neu[1, south] = True
        bc.is_dir[:, east] = True
        bc.is_neu[bc.is_dir + bc.is_rob] = False
        k = pp.FourthOrderTensor(
            np.random.rand(g.num_cells), np.random.rand(g.num_cells)
        )
        # Solve without rotations
        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        u_bound = np.random.rand(g.dim, g.num_faces)
        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        u = np.linalg.solve(A.toarray(), b)

        # Solve with rotations
        bc_b = pp.BoundaryConditionVectorial(g)
        bc_b.basis = basis
        bc_b.is_dir[1, west] = True
        bc_b.is_rob[0, west] = True
        bc_b.is_rob[1, north] = True
        bc_b.is_neu[0, north] = True
        bc_b.is_dir[1, south] = True
        bc_b.is_neu[0, south] = True
        bc_b.is_dir[:, east] = True
        bc_b.is_neu[bc_b.is_dir + bc_b.is_rob] = False

        bc_val_b = np.sum(basis * u_bound, axis=1).ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc_b,
            "inverter": "python",
            "bc_values": bc_val_b,
        }
        data_b = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data_b)
        A_b, b_b = discr.assemble_matrix_rhs(g, data_b)
        u_b = np.linalg.solve(A_b.toarray(), b_b)

        # Assert that solutions are the same
        self.assertTrue(np.allclose(u, u_b))

    def run_test(self, g, basis, bc):
        g.compute_geometry()
        c = pp.FourthOrderTensor(
            np.random.rand(g.num_cells), np.random.rand(g.num_cells)
        )
        # Solve without rotations

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        u_bound = np.random.rand(g.dim, g.num_faces)
        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": c,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        u = np.linalg.solve(A.toarray(), b)

        # Solve with rotations
        bc.basis = basis
        data[pp.PARAMETERS][keyword]["bc"] = bc

        u_bound_b = np.sum(basis * u_bound, axis=1).ravel("F")
        data[pp.PARAMETERS][keyword]["bc_values"] = u_bound_b
        discr.discretize(g, data)
        A_b, b_b = discr.assemble_matrix_rhs(g, data)

        u_b = np.linalg.solve(A_b.toarray(), b_b)
        # Assert that solutions are the same
        self.assertTrue(np.allclose(u, u_b))


def true_2d(g, constit=None):
    if g.dim == 2:
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

    if constit is None:
        return g
    constit = constit.copy()
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=0)
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=1)
    return g, constit


class RobinBoundTest(unittest.TestCase):
    def test_dir_rob(self):
        nx = 2
        ny = 2
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = 1

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left + bot + top))
        neu_ind = np.ravel(np.argwhere([]))
        rob_ind = np.ravel(np.argwhere(right))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        self.assertTrue(np.allclose(u, u_ex(g.cell_centers).ravel("F")))
        self.assertTrue(np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F")))

    def test_dir_neu_rob(self):
        nx = 2
        ny = 3
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(left))
        neu_ind = np.ravel(np.argwhere(top))
        rob_ind = np.ravel(np.argwhere(right + bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[0], 0 * x[1]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[3, 0], [0, 1]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        self.assertTrue(np.allclose(u, u_ex(g.cell_centers).ravel("F")))
        self.assertTrue(np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F")))

    def test_structured_triang(self):
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(()))
        neu_ind = np.ravel(np.argwhere(()))
        rob_ind = np.ravel(np.argwhere(left + right + top + bot))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        self.assertTrue(np.allclose(u, u_ex(g.cell_centers).ravel("F")))
        self.assertTrue(np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F")))

    def test_unstruct_triang(self):
        corners = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        points = np.random.rand(2, 2)
        points = np.hstack((corners, points))
        g = pp.TriangleGrid(points)
        g.compute_geometry()
        c = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        robin_weight = np.pi

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(()))
        neu_ind = np.ravel(np.argwhere(()))
        rob_ind = np.ravel(np.argwhere((right + left + top + bot)))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(g, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2], [2, 0]])
            T_r = [np.dot(sigma, g.face_normals[:2, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((2, g.num_faces))

        sgn_n, _ = g.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = g.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(g.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(g.face_centers[:, rob_ind]) * g.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(g, c, robin_weight, bnd, u_bound)

        self.assertTrue(np.allclose(u, u_ex(g.cell_centers).ravel("F")))
        self.assertTrue(np.allclose(T, T_ex(np.arange(g.num_faces)).ravel("F")))

    def test_unstruct_tetrahedron(self):
        box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        domain = pp.Domain(box)
        network = pp.FractureNetwork3d([], domain=domain)
        mesh_args = {"mesh_size_frac": 3, "mesh_size_min": 3}
        mdg = network.mesh(mesh_args)
        sd = mdg.subdomains(dim=3)[0]
        c = pp.FourthOrderTensor(np.ones(sd.num_cells), np.ones(sd.num_cells))
        robin_weight = 1.0

        bot = sd.face_centers[2] < 1e-10
        top = sd.face_centers[2] > 1 - 1e-10
        west = sd.face_centers[0] < 1e-10
        east = sd.face_centers[0] > 1 - 1e-10
        north = sd.face_centers[1] > 1 - 1e-10
        south = sd.face_centers[1] < 1e-10

        dir_ind = np.ravel(np.argwhere(west + top))
        neu_ind = np.ravel(np.argwhere(bot))
        rob_ind = np.ravel(np.argwhere(east + north + south))

        names = ["dir"] * len(dir_ind) + ["rob"] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryConditionVectorial(sd, bnd_ind, names)

        def u_ex(x):
            return np.vstack((x[1], x[0], 0 * x[2]))

        def T_ex(faces):
            if np.size(faces) == 0:
                return np.atleast_2d(np.array([]))
            sigma = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
            T_r = [np.dot(sigma, sd.face_normals[:, f]) for f in faces]
            return np.vstack(T_r).T

        u_bound = np.zeros((3, sd.num_faces))

        sgn_n, _ = sd.signs_and_cells_of_boundary_faces(neu_ind)
        sgn_r, _ = sd.signs_and_cells_of_boundary_faces(rob_ind)

        u_bound[:, dir_ind] = u_ex(sd.face_centers[:, dir_ind])
        u_bound[:, neu_ind] = T_ex(neu_ind) * sgn_n
        u_bound[:, rob_ind] = (
            T_ex(rob_ind) * sgn_r
            + robin_weight * u_ex(sd.face_centers[:, rob_ind]) * sd.face_areas[rob_ind]
        )
        u, T = self.solve_mpsa(sd, c, robin_weight, bnd, u_bound)

        self.assertTrue(np.allclose(u, u_ex(sd.cell_centers).ravel("F")))
        self.assertTrue(np.allclose(T, T_ex(np.arange(sd.num_faces)).ravel("F")))

    def solve_mpsa(self, g, c, robin_weight, bnd, u_bound):
        bnd.robin_weight *= robin_weight

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        bc_val = u_bound.ravel("F")

        specified_data = {
            "fourth_order_tensor": c,
            "bc": bnd,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        u = np.linalg.solve(A.toarray(), b)
        T = matrix_dictionary[discr.stress_matrix_key] * u + matrix_dictionary[
            discr.bound_stress_matrix_key
        ] * u_bound.ravel("F")
        return u, T


class TestAsymmetricNeumann(unittest.TestCase):
    def test_cart_2d(self):
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        right = g.face_centers[0] > 1 - 1e-10
        top = g.face_centers[1] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, top] = True
        bc.is_dir[0, right] = True

        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        g, k = true_2d(g, k)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)
        _, igrad, _ = pp.Mpsa("")._create_inverse_gradient_matrix(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        data = np.array(
            [
                -0.75,
                0.25,
                -2.0,
                -2.0,
                0.25,
                -0.75,
                2.0,
                -2.0,
                -2.0,
                2.0,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                -2.0,
                -2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                -2.0,
                2.0,
            ]
        )
        indices = np.array(
            [
                0,
                9,
                3,
                4,
                0,
                9,
                10,
                2,
                6,
                6,
                8,
                10,
                1,
                15,
                13,
                5,
                13,
                15,
                11,
                12,
                7,
                12,
                14,
            ]
        )
        indptr = np.array([0, 2, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 19, 20, 22, 23])
        igrad_known = sps.csr_matrix((data, indices, indptr))
        self.assertTrue(np.all(np.abs(igrad - igrad_known).A < 1e-12))

    def test_cart_3d(self):
        g = pp.CartGrid([1, 1, 1], physdims=(1, 1, 1))
        g.compute_geometry()

        west = g.face_centers[0] < 1e-10
        east = g.face_centers[0] > 1 - 1e-10
        south = g.face_centers[1] < 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, west + east + south] = True
        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)

        _, igrad, _ = pp.Mpsa("")._create_inverse_gradient_matrix(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        data = np.array(
            [
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                4.0,
                -4.0,
                2.0,
                1.5,
                0.5,
                -0.5,
                -4.0,
                2.0,
                4.0,
                -0.5,
                -1.5,
                -0.5,
                2.0,
                4.0,
                -4.0,
                2.0,
                1.5,
                0.5,
                -0.5,
                -4.0,
                2.0,
                4.0,
                -0.5,
                -1.5,
                -0.5,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                4.0,
                4.0,
                2.0,
                1.5,
                -0.5,
                -0.5,
                4.0,
                2.0,
                4.0,
                -0.5,
                1.5,
                -0.5,
                2.0,
                4.0,
                4.0,
                2.0,
                1.5,
                -0.5,
                -0.5,
                4.0,
                2.0,
                4.0,
                -0.5,
                1.5,
                -0.5,
            ]
        )
        indices = np.array(
            [
                36,
                44,
                4,
                60,
                48,
                56,
                16,
                68,
                60,
                68,
                28,
                36,
                56,
                40,
                47,
                5,
                64,
                52,
                59,
                17,
                71,
                64,
                71,
                29,
                40,
                59,
                37,
                0,
                7,
                49,
                12,
                31,
                37,
                19,
                61,
                24,
                12,
                31,
                37,
                41,
                3,
                6,
                53,
                15,
                30,
                41,
                18,
                65,
                27,
                15,
                30,
                41,
                39,
                45,
                8,
                63,
                51,
                57,
                20,
                69,
                63,
                69,
                32,
                39,
                57,
                43,
                46,
                9,
                67,
                55,
                58,
                21,
                70,
                67,
                70,
                33,
                43,
                58,
                38,
                1,
                11,
                50,
                13,
                35,
                38,
                23,
                62,
                25,
                13,
                35,
                38,
                42,
                2,
                10,
                54,
                14,
                34,
                42,
                22,
                66,
                26,
                14,
                34,
                42,
            ]
        )
        indptr = np.array(
            [
                0,
                1,
                2,
                4,
                5,
                6,
                8,
                9,
                10,
                13,
                14,
                15,
                17,
                18,
                19,
                21,
                22,
                23,
                26,
                27,
                28,
                29,
                30,
                33,
                34,
                35,
                36,
                39,
                40,
                41,
                42,
                43,
                46,
                47,
                48,
                49,
                52,
                53,
                54,
                56,
                57,
                58,
                60,
                61,
                62,
                65,
                66,
                67,
                69,
                70,
                71,
                73,
                74,
                75,
                78,
                79,
                80,
                81,
                82,
                85,
                86,
                87,
                88,
                91,
                92,
                93,
                94,
                95,
                98,
                99,
                100,
                101,
                104,
            ]
        )

        igrad_known = sps.csr_matrix((data, indices, indptr))
        self.assertTrue(np.all(np.abs(igrad - igrad_known).A < 1e-12))


if __name__ == "__main__":
    unittest.main()
