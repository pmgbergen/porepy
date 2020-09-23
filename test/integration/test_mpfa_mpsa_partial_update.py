""" Unit tests for the partial update features of mpfa and mpsa.

Split into four classes:
    TestPartialMPFA, MPSA and Biot 
test the option of setting 'specified_{cells, faces, nodes}', for flow, mechanics and
 poro-mechanics, respectively.

The class UpdateDiscretizations tests the update_discretization() methods of 
Mpfa, Mpsa and Biot. These are effectively a second set of test of the specified_*
keyword, but in addition, updates of grid geometry etc. are also probed.

"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestPartialMPFA(unittest.TestCase):
    """ Test various partial assembly for mpfa.
    """

    def setup(self, sz=None):
        if sz is None:
            sz = [5, 5]
        g = pp.CartGrid(sz)
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
        (
            g,
            stiffness,
            bnd,
            div_u,
            bound_div_u,
            grad_p,
            stab,
            bound_pressure,
        ) = self.setup_biot()

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
        (
            g,
            stiffness,
            bnd,
            div_u,
            bound_div_u,
            grad_p,
            stab,
            bound_pressure,
        ) = self.setup_biot()

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
            #            "inverter": "python",
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


class UpdateDiscretizations(unittest.TestCase):
    """ Tests specifically for the update_discretization methods in the
     Mpfa, Mpsa and Biot classes.

    The tests are structured as follows:
        1. Generate a small (Cartesian 3x4) and a large (4x4) grid.
        2. Discretize the small problem
        3. Use the update_discretization() method to transfer the small discretization
           to the larger grid, and discretize those cells / faces that were not updated.
        4. Compare the result from 3. with a full discretization on the standard grid.

    """

    def setup(self):
        self.g = pp.CartGrid([3, 4])
        self.g.compute_geometry()
        self.g_larger = pp.CartGrid([4, 4])
        self.g_larger.compute_geometry()
        cell_map_index = np.array(
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], dtype=np.int
        )
        self.cell_map = sps.coo_matrix(
            (np.ones(self.g.num_cells), (cell_map_index, np.arange(self.g.num_cells))),
            shape=(self.g_larger.num_cells, self.g.num_cells),
        ).tocsr()

        face_map_index = np.array(
            [
                0,
                1,
                2,
                3,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                20,
                21,
                22,
                24,
                25,
                26,
                28,
                29,
                30,
                32,
                33,
                34,
                36,
                37,
                38,
            ]
        )
        self.face_map = sps.coo_matrix(
            (np.ones(self.g.num_faces), (face_map_index, np.arange(self.g.num_faces))),
            shape=(self.g_larger.num_faces, self.g.num_faces),
        ).tocsr()
        self.new_cells = np.array([3, 7, 11, 15])

    def _update_and_compare(
        self, data_small, data_partial, data_full, g_larger, keywords, discr
    ):

        for keyword in keywords:
            # Transfer discretized matrices from the small problem
            for key, val in data_small[pp.DISCRETIZATION_MATRICES][keyword].items():
                data_partial[pp.DISCRETIZATION_MATRICES][keyword][key] = val

        discr.update_discretization(g_larger, data_partial)
        # Update discretizations

        for keyword in keywords:
            dict_partial = data_partial[pp.DISCRETIZATION_MATRICES][keyword]

            # Compare
            for key, mat_full in data_full[pp.DISCRETIZATION_MATRICES][keyword].items():
                mat_partial = dict_partial[key]
                self.assertTrue(np.allclose(mat_full.shape, mat_partial.shape))
                self.assertTrue(np.max(mat_full - mat_partial) < 1e-8)
                self.assertTrue(np.min(mat_full - mat_partial) < 1e-8)

    def test_mpfa(self):
        self.setup()

        g, g_larger = self.g, self.g_larger

        specified_data = {
            "inverter": "python",
        }

        keyword = "flow"
        data_small = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpfa(keyword)
        # Discretization on a small problem
        discr.discretize(g, data_small)

        # Perturb one node
        g_larger.nodes[0, 2] += 0.2
        # Faces that have their geometry changed
        update_faces = np.array([2, 21, 22])

        # Perturb the permeability in some cells on the larger grid
        perm_larger = pp.SecondOrderTensor(np.ones(g_larger.num_cells))
        high_perm_cells = np.array([7, 12])
        perm_larger.values[:, :, high_perm_cells] *= 10
        specified_data_larger = {"second_order_tensor": perm_larger}

        # Do a full discretization on the larger grid
        data_full = pp.initialize_default_data(
            g_larger, {}, keyword, specified_parameters=specified_data_larger
        )
        discr.discretize(g_larger, data_full)

        # Cells that will be marked as updated, either due to changed parameters or
        # the newly defined topology
        update_cells = np.union1d(self.new_cells, high_perm_cells)

        updates = {
            "modified_cells": update_cells,
            #            "modified_faces": update_faces,
            "map_cells": self.cell_map,
            "map_faces": self.face_map,
        }

        # Data dictionary for the two-step discretization
        data_partial = pp.initialize_default_data(
            g_larger, {}, keyword, specified_parameters=specified_data_larger
        )
        data_partial["update_discretization"] = updates

        self._update_and_compare(
            data_small, data_partial, data_full, g_larger, [keyword], discr
        )

    def test_mpsa(self):
        self.setup()

        g, g_larger = self.g, self.g_larger

        specified_data = {
            "inverter": "python",
        }

        keyword = "mechanics"
        data_small = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr = pp.Mpsa(keyword)
        # Discretization on a small problem
        discr.discretize(g, data_small)

        # Perturb one node
        g_larger.nodes[0, 2] += 0.2
        # Faces that have their geometry changed
        update_faces = np.array([2, 21, 22])

        # Perturb the permeability in some cells on the larger grid
        mu, lmbda = np.ones(g_larger.num_cells), np.ones(g_larger.num_cells)

        high_coeff_cells = np.array([7, 12])
        stiff_larger = pp.FourthOrderTensor(mu, lmbda)

        specified_data_larger = {"fourth_order_tensor": stiff_larger}

        # Do a full discretization on the larger grid
        data_full = pp.initialize_default_data(
            g_larger, {}, keyword, specified_parameters=specified_data_larger
        )
        discr.discretize(g_larger, data_full)

        # Cells that will be marked as updated, either due to changed parameters or
        # the newly defined topology
        update_cells = np.union1d(self.new_cells, high_coeff_cells)

        updates = {
            "modified_cells": update_cells,
            #            "modified_faces": update_faces,
            "map_cells": self.cell_map,
            "map_faces": self.face_map,
        }

        # Data dictionary for the two-step discretization
        data_partial = pp.initialize_default_data(
            g_larger, {}, keyword, specified_parameters=specified_data_larger
        )
        data_partial["update_discretization"] = updates

        self._update_and_compare(
            data_small, data_partial, data_full, g_larger, [keyword], discr
        )

    def test_biot(self):
        self.setup()

        g, g_larger = self.g, self.g_larger

        specified_data = {"inverter": "python", "biot_alpha": 1}

        mechanics_keyword = "mechanics"
        flow_keyword = "flow"
        data_small = pp.initialize_default_data(
            g, {}, mechanics_keyword, specified_parameters=specified_data
        )

        def add_flow_data(g, d):
            d[pp.DISCRETIZATION_MATRICES][flow_keyword] = {}
            d[pp.PARAMETERS][flow_keyword] = {
                "bc": pp.BoundaryCondition(g),
                "second_order_tensor": pp.SecondOrderTensor(np.ones(g.num_cells)),
                "bc_values": np.zeros(g.num_faces),
                "inverter": "python",
                "mass_weight": np.ones(g.num_cells),
                "biot_alpha": 1,
            }

        discr = pp.Biot(mechanics_keyword=mechanics_keyword, flow_keyword=flow_keyword)

        add_flow_data(g, data_small)

        discr.discretize(g, data_small)
        # Discretization on a small problem

        # Perturb one node
        g_larger.nodes[0, 2] += 0.2
        # Faces that have their geometry changed
        update_faces = np.array([2, 21, 22])

        # Perturb the permeability in some cells on the larger grid
        mu, lmbda = np.ones(g_larger.num_cells), np.ones(g_larger.num_cells)

        high_coeff_cells = np.array([7, 12])
        stiff_larger = pp.FourthOrderTensor(mu, lmbda)

        specified_data_larger = {"fourth_order_tensor": stiff_larger, "biot_alpha": 1}

        # Do a full discretization on the larger grid
        data_full = pp.initialize_default_data(
            g_larger, {}, mechanics_keyword, specified_parameters=specified_data_larger
        )
        add_flow_data(g_larger, data_full)

        discr.discretize(g_larger, data_full)

        # Cells that will be marked as updated, either due to changed parameters or
        # the newly defined topology
        update_cells = np.union1d(self.new_cells, high_coeff_cells)

        updates = {
            "modified_cells": update_cells,
            #            "modified_faces": update_faces,
            "map_cells": self.cell_map,
            "map_faces": self.face_map,
        }

        # Data dictionary for the two-step discretization
        data_partial = pp.initialize_default_data(
            g_larger, {}, mechanics_keyword, specified_parameters=specified_data_larger
        )
        add_flow_data(g_larger, data_partial)
        data_partial["update_discretization"] = updates

        self._update_and_compare(
            data_small,
            data_partial,
            data_full,
            g_larger,
            keywords=[flow_keyword, mechanics_keyword],
            discr=discr,
        )


UpdateDiscretizations().test_mpfa()
UpdateDiscretizations().test_mpsa()
UpdateDiscretizations().test_biot()

if __name__ == "__main__":
    unittest.main()
