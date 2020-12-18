from __future__ import division

import unittest
from test.integration import _helper_test_upwind_coupling
from test.test_utils import permute_matrix_vector

import numpy as np

import porepy as pp

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_bottom_top(self):
        gb, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # assign parameters
        gb.add_node_props(["param"])
        tol = 1e-3
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(1e-2, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                top = bound_face_centers[1, :] > 1 - tol
                bottom = bound_face_centers[1, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(top, bottom)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(g, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [0, 1, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()
        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)
        U_known = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, -1.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )

        rhs_known = np.array([1, 0, 0, 0, 0])

        theta_known = [1, 1, 1, -1, 1]

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known))

    #        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right(self):
        gb, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        tol = 1e-3
        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # assign parameters

        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:

            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(g, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})


        add_constant_darcy_flux(gb, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)

        U_known = np.array(
            [
                [0.5, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.5, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.01, -1.0, -1.0],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )

        rhs_known = np.array([0.5, 0.5, 1e-2, 0, 0])
        theta_known = np.array([1, 1, 1, 0, 0])

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    #        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right_cross(self):
        gb, _ = pp.grid_buckets_2d.two_intersecting([2, 2], simplex=False)

        # Enforce node orderning because of Python 3.5 and 2.7.
        # Don't do it in general.
        cell_centers_1 = np.array(
            [
                [7.50000000e-01, 2.5000000e-01],
                [0.50000000e00, 0.50000000e00],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )
        cell_centers_2 = np.array(
            [
                [0.50000000e00, 0.50000000e00],
                [7.50000000e-01, 2.5000000e-01],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )

        for g, d in gb:
            if g.dim == 2:
                d["node_number"] = 0
            elif g.dim == 1:
                if np.allclose(g.cell_centers, cell_centers_1):
                    d["node_number"] = 1
                elif np.allclose(g.cell_centers, cell_centers_2):
                    d["node_number"] = 2
                else:
                    raise ValueError("Grid not found")
            elif g.dim == 0:
                d["node_number"] = 3
            else:
                raise ValueError

        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            n1 = gb.node_props(g1, "node_number")
            n2 = gb.node_props(g2, "node_number")
            if n1 == 1 and n2 == 0:
                d["edge_number"] = 0
            elif n1 == 2 and n2 == 0:
                d["edge_number"] = 1
            elif n1 == 3 and n2 == 1:
                d["edge_number"] = 2
            elif n1 == 3 and n2 == 2:
                d["edge_number"] = 3
            else:
                raise ValueError

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # define parameters
        tol = 1e-3
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(g, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})
            else:
                bound = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
                specified_parameters.update({"bc": bound})
            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"
        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)
        U_known = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
            ]
        )

        theta_known = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                -0.5,
                -0.5,
                0.5,
                0.5,
                0.01,
                -0.01,
                0,
                0,
            ]
        )

        rhs_known = np.array(
            [
                0.5,
                0.0,
                0.5,
                0.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )

        rtol = 1e-15
        atol = rtol

        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    #        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_bottom_top(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        gb = pp.meshing.cart_grid([f], [1, 1, 2], **{"physdims": [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # add parameters
        tol = 1e-3
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                top = bound_face_centers[2, :] > 1 - tol
                bottom = bound_face_centers[2, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(top, bottom)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(g, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})
            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [0, 0, 1], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)

        U_known = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, -1.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
        rhs_known = np.array([1, 0, 0, 0, 0])

        theta_known = np.array([1, 1, 1, -1, 1])

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    #       self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_left_right(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        gb = pp.meshing.cart_grid([f], [1, 1, 2], **{"physdims": [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(left, right)] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)
        U_known = np.array(
            [
                [0.5, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.5, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.01, -1.0, -1.0],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
        rhs_known = np.array([0.5, 0.5, 0.01, 0.0, 0.0])

        theta_known = np.array([1, 1, 1, 0, 0])

        deltaT_known = 5 * 1e-1

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    #        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_1d_0d(self):
        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        f2 = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])

        gb = pp.meshing.cart_grid([f1, f2, f3], [2, 2, 2], **{"physdims": [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        cell_centers1 = np.array(
            [[0.25, 0.75, 0.25, 0.75], [0.25, 0.25, 0.75, 0.75], [0.5, 0.5, 0.5, 0.5]]
        )
        cell_centers2 = np.array(
            [[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.75, 0.25, 0.75, 0.25]]
        )
        cell_centers3 = np.array(
            [[0.25, 0.75, 0.25, 0.75], [0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
        )
        cell_centers4 = np.array([[0.5], [0.25], [0.5]])
        cell_centers5 = np.array([[0.5], [0.75], [0.5]])
        cell_centers6 = np.array([[0.75], [0.5], [0.5]])
        cell_centers7 = np.array([[0.25], [0.5], [0.5]])
        cell_centers8 = np.array([[0.5], [0.5], [0.25]])
        cell_centers9 = np.array([[0.5], [0.5], [0.75]])

        for g, d in gb:
            if np.allclose(g.cell_centers[:, 0], cell_centers1[:, 0]):
                d["node_number"] = 1
            elif np.allclose(g.cell_centers[:, 0], cell_centers2[:, 0]):
                d["node_number"] = 2
            elif np.allclose(g.cell_centers[:, 0], cell_centers3[:, 0]):
                d["node_number"] = 3
            elif np.allclose(g.cell_centers[:, 0], cell_centers4[:, 0]):
                d["node_number"] = 4
            elif np.allclose(g.cell_centers[:, 0], cell_centers5[:, 0]):
                d["node_number"] = 5
            elif np.allclose(g.cell_centers[:, 0], cell_centers6[:, 0]):
                d["node_number"] = 6
            elif np.allclose(g.cell_centers[:, 0], cell_centers7[:, 0]):
                d["node_number"] = 7
            elif np.allclose(g.cell_centers[:, 0], cell_centers8[:, 0]):
                d["node_number"] = 8
            elif np.allclose(g.cell_centers[:, 0], cell_centers9[:, 0]):
                d["node_number"] = 9
            else:
                pass

        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            n1 = gb.node_props(g1, "node_number")
            n2 = gb.node_props(g2, "node_number")
            if n1 == 1 and n2 == 0:
                d["edge_number"] = 0
            elif n1 == 2 and n2 == 0:
                d["edge_number"] = 1
            elif n1 == 3 and n2 == 0:
                d["edge_number"] = 2
            elif n1 == 4 and n2 == 1:
                d["edge_number"] = 3
            elif n1 == 5 and n2 == 1:
                d["edge_number"] = 4
            elif n1 == 6 and n2 == 1:
                d["edge_number"] = 5
            elif n1 == 7 and n2 == 1:
                d["edge_number"] = 6
            elif n1 == 4 and n2 == 2:
                d["edge_number"] = 7
            elif n1 == 5 and n2 == 2:
                d["edge_number"] = 8
            elif n1 == 8 and n2 == 2:
                d["edge_number"] = 9
            elif n1 == 9 and n2 == 2:
                d["edge_number"] = 10
            elif n1 == 6 and n2 == 3:
                d["edge_number"] = 11
            elif n1 == 7 and n2 == 3:
                d["edge_number"] = 12
            elif n1 == 8 and n2 == 3:
                d["edge_number"] = 13
            elif n1 == 9 and n2 == 3:
                d["edge_number"] = 14
            elif n1 == 10 and n2 == 4:
                d["edge_number"] = 15
            elif n1 == 10 and n2 == 5:
                d["edge_number"] = 16
            elif n1 == 10 and n2 == 6:
                d["edge_number"] = 17
            elif n1 == 10 and n2 == 7:
                d["edge_number"] = 18
            elif n1 == 10 and n2 == 8:
                d["edge_number"] = 19
            elif n1 == 10 and n2 == 9:
                d["edge_number"] = 20
            else:
                raise ValueError

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:

                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(g, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)

        assembler.discretize()
        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"
        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = np.linalg.solve(U.A, rhs)
        #        deltaT = solver.cfl(gb)
        (
            U_known,
            rhs_known,
        ) = (
            _helper_test_upwind_coupling.matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d()
        )

        theta_known = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1,
                1,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.25,
                -0.25,
                -0.25,
                -0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -5.0e-03,
                5.0e-03,
                -5.0e-03,
                5.0e-03,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -5e-03,
                5e-03,
                -5e-03,
                5e-03,
                0,
                0,
                -1e-04,
                1e-04,
                0,
                0,
            ]
        )

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    #        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_beta_positive(self):

        f = np.array([[2, 2], [0, 2]])
        gb = pp.meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(gb, upwind, upwind_coupling, variable)

        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            bf = g.tags["domain_boundary_faces"].nonzero()[0]
            bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
            specified_parameters = {"aperture": aperture, "bc": bc}
            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [2, 0, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        M, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        # add generic mass matrix to solve system
        I_diag = np.zeros(M.shape[0])
        I_diag[: gb.num_cells()] = 1
        I = np.diag(I_diag)
        theta = np.linalg.solve(M + I, I_diag + rhs)
        M_known = np.array(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
        rhs_known = np.zeros(14)
        theta_known = np.array(
            [
                0.33333333,
                0.55555556,
                0.80246914,
                2.60493827,
                0.33333333,
                0.55555556,
                0.80246914,
                2.60493827,
                0.7037037,
                0.7037037,
                -1.40740741,
                -1.40740741,
                1.11111111,
                1.11111111,
            ]
        )

        rtol = 1e-8
        atol = rtol

        self.assertTrue(np.allclose(M.A, M_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_full_beta_bc_dir(self):

        f = np.array([[2, 2], [0, 2]])
        gb = pp.meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()

        # define discretization
        key = "transport"
        variable = "T"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        assign_discretization(gb, upwind, upwind_coupling, variable)

        a = 1e-1
        for g, d in gb:
            aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            labels = np.array(["dir"] * bound_faces.size)
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = 3

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
            pp.initialize_default_data(g, d, "transport", specified_parameters)

        for e, d in gb.edges():
            pp.initialize_data(d["mortar_grid"], d, "transport", {})

        add_constant_darcy_flux(gb, upwind, [1, 1, 0], a)

        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = np.empty(gb.num_graph_nodes() + gb.num_graph_edges(), dtype=np.object)
        variables = np.empty_like(grids)
        for g, d in gb:
            grids[d["node_number"]] = g
            variables[d["node_number"]] = variable
        for e, d in gb.edges():
            grids[d["edge_number"] + gb.num_graph_nodes()] = e
            variables[d["edge_number"] + gb.num_graph_nodes()] = "lambda_u"

        M, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = np.linalg.solve(M.A, rhs)
        M_known = np.array(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    -0.1,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )

        rhs_known = np.array(
            [6.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        )

        theta_known = np.array(
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, -3.0, -3.0, 3.0, 3.0]
        )

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M.todense(), M_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))


# ------------------------------------------------------------------------------#


def assign_discretization(gb, disc, coupling_disc, variable):
    # Identifier of the advection term
    term = "advection"
    for _, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        d[pp.DISCRETIZATION] = {variable: {term: disc}}

    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {"lambda_u": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            variable: {
                g1: (variable, term),
                g2: (variable, term),
                e: ("lambda_u", coupling_disc),
            }
        }


def add_constant_darcy_flux(gb, upwind, flux, a):
    """
    Adds the constant darcy_flux to all internal and mortar faces, the latter
    in the "mortar_solution" field.
    gb - grid bucket
    upwind- upwind discretization class
    flux - 3 x 1 flux at all faces [u, v, w]
    a - cross-sectional area of the fractures.
    """
    for g, d in gb:
        aperture = np.ones(g.num_cells) * np.power(a, gb.dim_max() - g.dim)
        d[pp.PARAMETERS]["transport"]["darcy_flux"] = upwind.darcy_flux(
            g, flux, aperture
        )
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        p_h = gb.node_props(g_h, pp.PARAMETERS)
        darcy_flux = p_h["transport"]["darcy_flux"]
        sign = np.zeros(g_h.num_faces)
        boundary_faces = g_h.get_all_boundary_faces()
        boundary_signs, _ = g_h.signs_and_cells_of_boundary_faces(boundary_faces)
        sign[boundary_faces] = boundary_signs

        mg = d["mortar_grid"]
        sign = mg.primary_to_mortar_avg() * sign
        #        d["param"] = pp.Parameters(g_h)
        darcy_flux_e = sign * (d["mortar_grid"].primary_to_mortar_avg() * darcy_flux)
        if pp.PARAMETERS not in d:
            d[pp.PARAMETERS] = pp.Parameters(
                mg, ["transport"], [{"darcy_flux": darcy_flux_e}]
            )
        else:
            d[pp.PARAMETERS]["transport"]["darcy_flux"] = darcy_flux_e


# #------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
