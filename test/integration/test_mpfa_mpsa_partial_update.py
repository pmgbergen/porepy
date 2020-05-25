""" Unit tests for the partial update features of mpfa and mpsa.

Split into three classes, for flow, mechanics and poro-mechanics, respectively.

"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestPartialMPFA(unittest.TestCase):
    """ Test various partial assembly for mpfa.
    """

    def setup(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))
        bnd = pp.BoundaryCondition(g)
        flux, bound_flux, _, _, vector_source, _ = pp.Mpfa("flow")._flux_discretization(
            g, perm, bnd, inverter="python"
        )
        return g, perm, bnd, flux, bound_flux, vector_source

    def test_inner_cell_node_keyword(self):
        # Compute update for a single cell in the interior.
        g, perm, bnd, flux, bound_flux, vector_source = self.setup()

        nodes_of_cell = np.array([14, 15, 20, 21])
        faces_of_cell = np.array([14, 15, 42, 47])

        specified_data = {
            "second_order_tensor": perm,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": nodes_of_cell,
        }

        keyword = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )
        discr = pp.Mpfa(keyword)
        discr.discretize(g, data)

        partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_flux_matrix_key
        ]
        partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.vector_source_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_flux = (flux - partial_flux).todense()
        diff_bound = (bound_flux - partial_bound).todense()
        diff_vc = (vector_source - partial_vector_source).todense()

        self.assertTrue(np.max(np.abs(diff_flux[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_vc[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        pp.fvutils.zero_out_sparse_rows(partial_flux, faces_of_cell)
        pp.fvutils.zero_out_sparse_rows(partial_bound, faces_of_cell)
        pp.fvutils.zero_out_sparse_rows(partial_vector_source, faces_of_cell)

        self.assertTrue(np.max(np.abs(partial_flux.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_vector_source.data)) == 0)

    def test_bound_cell_node_keyword(self):
        # Compute update for a single cell on the boundary
        g, perm, bnd, flux, bound_flux, vector_source = self.setup()

        # cell = 10
        nodes_of_cell = np.array([12, 13, 18, 19])
        faces_of_cell = np.array([12, 13, 40, 45])
        specified_data = {
            "second_order_tensor": perm,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": nodes_of_cell,
        }

        keyword = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpfa(keyword)
        discr.discretize(g, data)

        partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][discr.flux_matrix_key]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_flux_matrix_key
        ]
        partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.vector_source_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_flux = (flux - partial_flux).todense()
        diff_bound = (bound_flux - partial_bound).todense()
        diff_vc = (vector_source - partial_vector_source).todense()

        self.assertTrue(np.max(np.abs(diff_flux[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_vc[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        pp.fvutils.zero_out_sparse_rows(partial_flux, faces_of_cell)
        pp.fvutils.zero_out_sparse_rows(partial_bound, faces_of_cell)
        pp.fvutils.zero_out_sparse_rows(partial_vector_source, faces_of_cell)

        self.assertTrue(np.max(np.abs(partial_flux.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_vector_source.data)) == 0)

    def test_one_cell_a_time_node_keyword(self):
        # Update one and one cell, and verify that the result is the same as
        # with a single computation.
        # The test is similar to what will happen with a memory-constrained
        # splitting.
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        # Assign random permeabilities, for good measure
        np.random.seed(42)
        kxx = np.random.random(g.num_cells)
        kyy = np.random.random(g.num_cells)
        # Ensure positive definiteness
        kxy = np.random.random(g.num_cells) * kxx * kyy
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)

        flux = sps.csr_matrix((g.num_faces, g.num_cells))
        bound_flux = sps.csr_matrix((g.num_faces, g.num_faces))
        vc = sps.csr_matrix((g.num_faces, g.num_cells * g.dim))
        faces_covered = np.zeros(g.num_faces, np.bool)

        bnd = pp.BoundaryCondition(g)

        cn = g.cell_nodes()
        for ci in range(g.num_cells):
            ind = np.zeros(g.num_cells)
            ind[ci] = 1
            nodes = np.squeeze(np.where(cn * ind > 0))

            specified_data = {
                "second_order_tensor": perm,
                "bc": bnd,
                "inverter": "python",
                "specified_nodes": nodes,
            }

            keyword = "flow"
            data = pp.initialize_default_data(
                g, {}, keyword, specified_parameters=specified_data
            )

            discr = pp.Mpfa(keyword)
            discr.discretize(g, data)

            partial_flux = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.flux_matrix_key
            ]
            partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_flux_matrix_key
            ]
            partial_vector_source = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.vector_source_matrix_key
            ]

            active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

            if np.any(faces_covered):
                fi = np.where(faces_covered)[0]
                pp.fvutils.remove_nonlocal_contribution(
                    fi, 1, partial_flux, partial_bound, partial_vector_source
                )

            faces_covered[active_faces] = True

            flux += partial_flux
            bound_flux += partial_bound
            vc += partial_vector_source

        flux_full, bound_flux_full, *_, vc_full, _ = pp.Mpfa(
            "flow"
        )._flux_discretization(g, perm, bnd, inverter="python")

        self.assertTrue((flux_full - flux).max() < 1e-8)
        self.assertTrue((flux_full - flux).min() > -1e-8)
        self.assertTrue((bound_flux - bound_flux_full).max() < 1e-8)
        self.assertTrue((bound_flux - bound_flux_full).min() > -1e-8)
        self.assertTrue((vc - vc_full).max() < 1e-8)
        self.assertTrue((vc - vc_full).min() > -1e-8)


class TestPartialMPSA(unittest.TestCase):
    """ Test various partial assembly features for mpsa.
    """

    def setup(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        bnd = pp.BoundaryConditionVectorial(g)

        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
        }
        keyword = "mechanics"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)
        stress = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        return g, stiffness, bnd, stress, bound_stress

    def expand_indices_nd(self, ind, nd, direction="F"):
        dim_inds = np.arange(nd)
        dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
        new_ind = nd * ind + dim_inds
        new_ind = new_ind.ravel(direction)
        return new_ind

    def test_inner_cell_node_keyword(self):
        # Compute update for a single cell in the interior.
        g, stiffness, bnd, stress, bound_stress = self.setup()

        # inner_cell = 12  # The target cell
        nodes_of_cell = np.array([14, 15, 20, 21])
        faces_of_cell = np.array([14, 15, 42, 47])

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": np.array([nodes_of_cell]),
        }
        keyword = "mechanics"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_stress = (stress - partial_stress).todense()
        diff_bound = (bound_stress - partial_bound).todense()

        faces_of_cell = self.expand_indices_nd(faces_of_cell, g.dim)
        self.assertTrue(np.max(np.abs(diff_stress[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        pp.fvutils.remove_nonlocal_contribution(
            faces_of_cell, 1, partial_stress, partial_bound
        )
        self.assertTrue(np.max(np.abs(partial_stress.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_bound_cell_node_keyword(self):
        # Compute update for a single cell on the
        g, stiffness, bnd, stress, bound_stress = self.setup()

        # inner_cell = 10
        nodes_of_cell = np.array([12, 13, 18, 19])
        faces_of_cell = np.array([12, 13, 40, 45])

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": np.array([nodes_of_cell]),
        }
        keyword = "mechanics"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.stress_matrix_key
        ]
        partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        faces_of_cell = self.expand_indices_nd(faces_of_cell, g.dim)
        diff_stress = (stress - partial_stress).todense()
        diff_bound = (bound_stress - partial_bound).todense()

        self.assertTrue(np.max(np.abs(diff_stress[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be non-zero.
        pp.fvutils.remove_nonlocal_contribution(
            faces_of_cell, 1, partial_stress, partial_bound
        )
        self.assertTrue(np.max(np.abs(partial_stress.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_one_cell_a_time_node_keyword(self):
        # Update one and one cell, and verify that the result is the same as
        # with a single computation. The test is similar to what will happen
        # with a memory-constrained splitting.
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        # Assign random permeabilities, for good measure
        np.random.seed(42)
        mu = np.random.random(g.num_cells)
        lmbda = np.random.random(g.num_cells)
        stiffness = pp.FourthOrderTensor(mu=mu, lmbda=lmbda)

        stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
        bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))
        faces_covered = np.zeros(g.num_faces, np.bool)

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
        }
        keyword = "mechanics"
        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        discr.discretize(g, data)

        stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][discr.stress_matrix_key]
        bound_stress_full = data[pp.DISCRETIZATION_MATRICES][keyword][
            discr.bound_stress_matrix_key
        ]

        cn = g.cell_nodes()
        for ci in range(g.num_cells):
            ind = np.zeros(g.num_cells)
            ind[ci] = 1
            nodes = np.squeeze(np.where(cn * ind > 0))

            data[pp.PARAMETERS][keyword]["specified_nodes"] = nodes

            discr.discretize(g, data)

            partial_stress = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.stress_matrix_key
            ]
            partial_bound = data[pp.DISCRETIZATION_MATRICES][keyword][
                discr.bound_stress_matrix_key
            ]

            active_faces = data[pp.PARAMETERS][keyword]["active_faces"]

            if np.any(faces_covered):
                del_faces = self.expand_indices_nd(np.where(faces_covered)[0], g.dim)
                # del_faces is already expanded, set dimension to 1
                pp.fvutils.remove_nonlocal_contribution(
                    del_faces, 1, partial_stress, partial_bound
                )
            faces_covered[active_faces] = True

            stress += partial_stress
            bound_stress += partial_bound

        self.assertTrue((stress_full - stress).max() < 1e-8)
        self.assertTrue((stress_full - stress).min() > -1e-8)
        self.assertTrue((bound_stress - bound_stress_full).max() < 1e-8)
        self.assertTrue((bound_stress - bound_stress_full).min() > -1e-8)


class PartialBiotMpsa(TestPartialMPSA):
    """ Test various partial assembly for mpsa for poro-elasticity.
    """

    def setup_biot(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        bnd = pp.BoundaryConditionVectorial(g)

        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "biot_alpha": 1,
        }
        keyword_mech = "mechanics"
        keyword_flow = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword_mech, specified_parameters=specified_data
        )
        data = pp.initialize_default_data(g, data, keyword_flow)

        discr = pp.Biot()
        discr.discretize(g, data)
        div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][discr.div_u_matrix_key]
        bound_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.bound_div_u_matrix_key
        ]
        stab = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.stabilization_matrix_key
        ]
        grad_p = data[pp.DISCRETIZATION_MATRICES][keyword_mech][discr.grad_p_matrix_key]
        bound_pressure = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.bound_pressure_matrix_key
        ]

        return g, stiffness, bnd, div_u, bound_div_u, grad_p, stab, bound_pressure

    def test_inner_cell_node_keyword(self):
        # Compute update for a single cell in the interior.
        g, stiffness, bnd, div_u, bound_div_u, grad_p, stab, bound_pressure = (
            self.setup_biot()
        )

        inner_cell = 12  # The target cell
        nodes_of_cell = np.array([14, 15, 20, 21])
        faces_of_cell = np.array([14, 15, 42, 47])

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": np.array([nodes_of_cell]),
            "biot_alpha": 1,
        }
        keyword_mech = "mechanics"
        keyword_flow = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword_mech, specified_parameters=specified_data
        )
        data = pp.initialize_default_data(g, data, keyword_flow)

        discr = pp.Biot()
        discr.discretize(g, data)

        partial_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.div_u_matrix_key
        ]
        partial_bound_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.bound_div_u_matrix_key
        ]
        partial_grad_p = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.grad_p_matrix_key
        ]
        partial_stab = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.stabilization_matrix_key
        ]
        partial_bound_pressure = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.bound_pressure_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword_mech]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_div_u = (div_u - partial_div_u).todense()
        diff_bound_div_u = (bound_div_u - partial_bound_div_u).todense()
        diff_grad_p = (grad_p - partial_grad_p).todense()
        diff_stab = (stab - partial_stab).todense()
        diff_bound_pressure = (bound_pressure - partial_bound_pressure).todense()

        faces_of_cell_vec = self.expand_indices_nd(faces_of_cell, g.dim)
        self.assertTrue(np.max(np.abs(diff_div_u[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound_div_u[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_grad_p[faces_of_cell_vec])) == 0)
        self.assertTrue(np.max(np.abs(diff_stab[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound_pressure[faces_of_cell_vec])) == 0)

        # Only the faces of the central cell should be zero
        pp.fvutils.remove_nonlocal_contribution(
            inner_cell, 1, partial_div_u, partial_bound_div_u, partial_stab
        )
        pp.fvutils.remove_nonlocal_contribution(
            faces_of_cell, g.dim, partial_grad_p, partial_bound_pressure
        )

        self.assertTrue(np.max(np.abs(partial_div_u.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound_div_u.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_grad_p.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_stab.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound_pressure.data)) == 0)

    def test_bound_cell_node_keyword(self):
        # Compute update for a single cell on the
        g, stiffness, bnd, div_u, bound_div_u, grad_p, stab, bound_pressure = (
            self.setup_biot()
        )

        inner_cell = 10
        nodes_of_cell = np.array([12, 13, 18, 19])
        faces_of_cell = np.array([12, 13, 40, 45])

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "specified_nodes": np.array([nodes_of_cell]),
            "biot_alpha": 1,
        }
        keyword_mech = "mechanics"
        keyword_flow = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword_mech, specified_parameters=specified_data
        )
        data = pp.initialize_default_data(g, data, keyword_flow)

        discr = pp.Biot()
        discr.discretize(g, data)

        partial_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.div_u_matrix_key
        ]
        partial_bound_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.bound_div_u_matrix_key
        ]
        partial_grad_p = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.grad_p_matrix_key
        ]
        partial_stab = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.stabilization_matrix_key
        ]
        partial_bound_pressure = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.bound_pressure_matrix_key
        ]

        active_faces = data[pp.PARAMETERS][keyword_mech]["active_faces"]

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_div_u = (div_u - partial_div_u).todense()
        diff_bound_div_u = (bound_div_u - partial_bound_div_u).todense()
        diff_grad_p = (grad_p - partial_grad_p).todense()
        diff_stab = (stab - partial_stab).todense()
        diff_bound_pressure = (bound_pressure - partial_bound_pressure).todense()

        faces_of_cell_vec = self.expand_indices_nd(faces_of_cell, g.dim)
        self.assertTrue(np.max(np.abs(diff_div_u[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound_div_u[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_grad_p[faces_of_cell_vec])) == 0)
        self.assertTrue(np.max(np.abs(diff_stab[inner_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound_pressure[faces_of_cell_vec])) == 0)

        # Only the faces of the central cell should be zero
        pp.fvutils.remove_nonlocal_contribution(
            inner_cell, 1, partial_div_u, partial_bound_div_u, partial_stab
        )
        pp.fvutils.remove_nonlocal_contribution(
            faces_of_cell, g.dim, partial_grad_p, partial_bound_pressure
        )

        self.assertTrue(np.max(np.abs(partial_div_u.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound_div_u.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_grad_p.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_stab.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound_pressure.data)) == 0)

    def test_one_cell_a_time_node_keyword(self):
        # Update one and one cell, and verify that the result is the same as
        # with a single computation. The test is similar to what will happen
        # with a memory-constrained splitting.
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        # Assign random permeabilities, for good measure
        np.random.seed(42)
        mu = np.random.random(g.num_cells)
        lmbda = np.random.random(g.num_cells)
        stiffness = pp.FourthOrderTensor(mu=mu, lmbda=lmbda)

        nd = g.dim
        nf = g.num_faces
        nc = g.num_cells

        grad_p = sps.csr_matrix((nf * nd, nc))
        div_u = sps.csr_matrix((nc, nc * nd))
        bound_div_u = sps.csr_matrix((nc, nf * nd))
        stab = sps.csr_matrix((nc, nc))
        bound_displacement_pressure = sps.csr_matrix((nf * nd, nc))

        faces_covered = np.zeros(g.num_faces, np.bool)
        cells_covered = np.zeros(g.num_cells, np.bool)

        bnd = pp.BoundaryConditionVectorial(g)
        specified_data = {
            "fourth_order_tensor": stiffness,
            "bc": bnd,
            "inverter": "python",
            "biot_alpha": 1,
        }
        keyword_mech = "mechanics"
        keyword_flow = "flow"
        data = pp.initialize_default_data(
            g, {}, keyword_mech, specified_parameters=specified_data
        )
        data = pp.initialize_default_data(g, data, keyword_flow)

        discr = pp.Biot()
        discr.discretize(g, data)

        div_u_full = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.div_u_matrix_key
        ]
        bound_div_u_full = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.bound_div_u_matrix_key
        ]
        stab_full = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
            discr.stabilization_matrix_key
        ]
        grad_p_full = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.grad_p_matrix_key
        ]
        bound_pressure_full = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
            discr.bound_pressure_matrix_key
        ]

        cn = g.cell_nodes()
        for ci in range(g.num_cells):
            ind = np.zeros(g.num_cells)
            ind[ci] = 1
            nodes = np.squeeze(np.where(cn * ind > 0))

            data[pp.PARAMETERS][keyword_mech]["specified_nodes"] = nodes

            discr.discretize(g, data)

            partial_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
                discr.div_u_matrix_key
            ]
            partial_bound_div_u = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
                discr.bound_div_u_matrix_key
            ]
            partial_grad_p = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
                discr.grad_p_matrix_key
            ]
            partial_stab = data[pp.DISCRETIZATION_MATRICES][keyword_flow][
                discr.stabilization_matrix_key
            ]
            partial_bound_pressure = data[pp.DISCRETIZATION_MATRICES][keyword_mech][
                discr.bound_pressure_matrix_key
            ]

            active_faces = data[pp.PARAMETERS][keyword_mech]["active_faces"]

            if np.any(faces_covered):
                del_faces = self.expand_indices_nd(np.where(faces_covered)[0], g.dim)
                del_cells = np.where(cells_covered)[0]
                pp.fvutils.remove_nonlocal_contribution(
                    del_cells, 1, partial_div_u, partial_bound_div_u, partial_stab
                )
                # del_faces is already expanded, set dimension to 1
                pp.fvutils.remove_nonlocal_contribution(
                    del_faces, 1, partial_grad_p, partial_bound_pressure
                )

            faces_covered[active_faces] = True
            cells_covered[ci] = True

            div_u += partial_div_u
            bound_div_u += partial_bound_div_u
            grad_p += partial_grad_p
            stab += partial_stab
            bound_displacement_pressure += partial_bound_pressure

        self.assertTrue((div_u_full - div_u).max() < 1e-8)
        self.assertTrue((bound_div_u_full - bound_div_u).min() > -1e-8)
        self.assertTrue((grad_p_full - grad_p).max() < 1e-8)
        self.assertTrue((stab_full - stab).min() > -1e-8)
        self.assertTrue(
            (bound_displacement_pressure - bound_pressure_full).min() > -1e-8
        )


if __name__ == "__main__":
    unittest.main()
