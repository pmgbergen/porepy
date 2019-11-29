import numpy as np
import unittest

import porepy as pp


class TestDisplacementReconstruction(unittest.TestCase):
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
            discr.bound_displacment_cell_matrix_key
        ].copy()
        hf_bound_old = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound_full = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound_dir = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
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
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound_rob = data_d_full[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
        ]

        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]
        hf_cell = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_cell_matrix_key
        ]
        hf_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_displacment_face_matrix_key
        ]

        self.assertTrue(np.allclose((stress - stress_rob).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_rob).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_rob).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_rob).data, 0))


if __name__ == "__main__":
    TestDisplacementReconstruction().test_changing_bc()
    unittest.main()
