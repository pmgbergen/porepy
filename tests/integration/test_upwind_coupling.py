from __future__ import division

import unittest
from tests.integration import _helper_test_upwind_coupling
from tests.test_utils import permute_matrix_vector

import numpy as np
from scipy.sparse.linalg import spsolve as sparse_solver

import porepy as pp

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_bottom_top(self):
        mdg, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(1e-2, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = sd.face_centers[:, bound_faces]

                top = bound_face_centers[1, :] > 1 - tol
                bottom = bound_face_centers[1, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(top, bottom)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [0, 1, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()
        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd, data in mdg.subdomains(return_data=True):
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = sparse_solver(U, rhs)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right(self):
        mdg, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        tol = 1e-3
        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):

            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = sd.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf, data in mdg.interfaces(return_data=True):
            grids.append(intf)
            variables.append("lambda_u")

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = sparse_solver(U, rhs)

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

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right_cross(self):
        mdg, _ = pp.md_grids_2d.two_intersecting([2, 2], simplex=False)

        # Enforce node ordering because of Python 3.5 and 2.7.
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

        # for sd, data in mdg.subdomains(return_data=True):
        #     if sd.dim == 2:
        #         data["node_number"] = 0
        #     elif sd.dim == 1:
        #         if np.allclose(sd.cell_centers, cell_centers_1):
        #             data["node_number"] = 1
        #         elif np.allclose(sd.cell_centers, cell_centers_2):
        #             data["node_number"] = 2
        #         else:
        #             raise ValueError("Grid not found")
        #     elif sd.dim == 0:
        #         data["node_number"] = 3
        #     else:
        #         raise ValueError
        # #
        # for intf, data in mdg.interfaces(return_data=True):
        #     sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
        #     n1 = mdg.subdomain_data(sd_secondary)["node_number"]
        #     n2 = mdg.subdomain_data(sd_primary)["node_number"]
        #     if n1 == 1 and n2 == 0:
        #         data["edge_number"] = 0
        #     elif n1 == 2 and n2 == 0:
        #         data["edge_number"] = 1
        #     elif n1 == 3 and n2 == 1:
        #         data["edge_number"] = 2
        #     elif n1 == 3 and n2 == 2:
        #         data["edge_number"] = 3
        #     else:
        #         raise ValueError

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # define parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = sd.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})
            else:
                bound = pp.BoundaryCondition(sd, np.empty(0), np.empty(0))
                specified_parameters.update({"bc": bound})
            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")
        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = sparse_solver(U, rhs)
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
        mdg = pp.meshing.cart_grid([f], [1, 1, 2], **{"physdims": [1, 1, 1]})
        mdg.compute_geometry()


        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # add parameters
        tol = 1e-3
        a = 1e-2
        for g, d in mdg.subdomains(return_data=True):
            aperture = np.ones(g.num_cells) * np.power(a, mdg.dim_max() - g.dim)
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

        for e, d in mdg.interfaces(return_data=True):
            pp.initialize_data(e, d, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [0, 0, 1], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )
        theta = sparse_solver(U, rhs)

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

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_left_right(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        mdg = pp.meshing.cart_grid([f], [1, 1, 2], **{"physdims": [1, 1, 1]})
        mdg.compute_geometry()


        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = sd.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(left, right)] = ["dir"]

            bc_val = np.zeros(sd.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            bound = pp.BoundaryCondition(sd, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")

        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = sparse_solver(U, rhs)
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

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_1d_0d(self):
        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        f2 = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])

        mdg = pp.meshing.cart_grid([f1, f2, f3], [2, 2, 2], **{"physdims": [1, 1, 1]})
        mdg.compute_geometry()


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

        # for sd, data in mdg.subdomains(return_data=True):
            # if sd.dim == mdg.dim_max():
            #     data["node_number"] = 0
            # if np.allclose(sd.cell_centers[:, 0], cell_centers1[:, 0]):
            #     data["node_number"] = 1
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers2[:, 0]):
            #     data["node_number"] = 2
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers3[:, 0]):
            #     data["node_number"] = 3
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers4[:, 0]):
            #     data["node_number"] = 4
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers5[:, 0]):
            #     data["node_number"] = 5
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers6[:, 0]):
            #     data["node_number"] = 6
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers7[:, 0]):
            #     data["node_number"] = 7
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers8[:, 0]):
            #     data["node_number"] = 8
            # elif np.allclose(sd.cell_centers[:, 0], cell_centers9[:, 0]):
            #     data["node_number"] = 9
            # else:
            #     data["node_number"] = -42


        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:

                bound_face_centers = sd.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)

        assembler.discretize()
        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")
        U, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        theta = sparse_solver(U, rhs)
        #        deltaT = solver.cfl(gb)
        (
            U_known,
            rhs_known,
        ) = (
            _helper_test_upwind_coupling.matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d()
        )

        theta_known = np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.50000000e-01,
                   -2.50000000e-01, -2.50000000e-01, -2.50000000e-01, 2.50000000e-01,
                   2.50000000e-01, 2.50000000e-01, 2.50000000e-01, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, -5.00000000e-03, 5.00000000e-03, 0.00000000e+00,
                   0.00000000e+00, -5.00000000e-03, 5.00000000e-03, -5.55111512e-19,
                   5.55111512e-19, -5.55111512e-19, 5.55111512e-19, -5.55111512e-19,
                   5.55111512e-19, -5.55111512e-19, 5.55111512e-19, 0.00000000e+00,
                   0.00000000e+00, -5.00000000e-03, 5.00000000e-03, 0.00000000e+00,
                   0.00000000e+00, -5.00000000e-03, 5.00000000e-03, 1.00000000e-04,
                   -0.00000000e+00, -0.00000000e+00, -1.00000000e-04, 0.00000000e+00,
                   0.00000000e+00])
        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(U.todense(), U_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(theta, theta_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_beta_positive(self):

        f = np.array([[2, 2], [0, 2]])
        mdg = pp.meshing.cart_grid([f], [4, 2])
        mdg.compute_geometry()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            bf = sd.tags["domain_boundary_faces"].nonzero()[0]
            bc = pp.BoundaryCondition(sd, bf, bf.size * ["neu"])
            specified_parameters = {"aperture": aperture, "bc": bc}
            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [2, 0, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")

        M, rhs = permute_matrix_vector(
            U_tmp, rhs, dof_manager.block_dof, dof_manager.full_dof, grids, variables
        )

        # add generic mass matrix to solve system
        I_diag = np.zeros(M.shape[0])
        I_diag[: mdg.num_subdomain_cells()] = 1
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
        mdg = pp.meshing.cart_grid([f], [4, 2])
        mdg.compute_geometry()

        # define discretization
        key = "transport"
        variable = "T"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        assign_discretization(mdg, upwind, upwind_coupling, variable)

        a = 1e-1
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            labels = np.array(["dir"] * bound_faces.size)
            bc_val = np.zeros(sd.num_faces)
            bc_val[bound_faces] = 3

            bound = pp.BoundaryCondition(sd, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        add_constant_darcy_flux(mdg, upwind, [1, 1, 0], a)

        dof_manager = pp.DofManager(mdg)
        assembler = pp.Assembler(mdg, dof_manager)
        assembler.discretize()

        U_tmp, rhs = assembler.assemble_matrix_rhs()

        grids = list()
        variables = list()
        for sd in mdg.subdomains():
            grids.append(sd)
            variables.append(variable)
        for intf in mdg.interfaces():
            grids.append(intf)
            variables.append("lambda_u")

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


def assign_discretization(mdg, disc, coupling_disc, variable):
    # Identifier of the advection term
    term = "advection"
    for _, data in mdg.subdomains(return_data=True):
        data[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        data[pp.DISCRETIZATION] = {variable: {term: disc}}

    for intf, data in mdg.interfaces(return_data=True):
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
        data[pp.PRIMARY_VARIABLES] = {"lambda_u": {"cells": 1}}
        data[pp.COUPLING_DISCRETIZATION] = {
            variable: {
                sd_primary: (variable, term),
                sd_secondary: (variable, term),
                intf: ("lambda_u", coupling_disc),
            }
        }


def add_constant_darcy_flux(mdg, upwind, flux, a):
    """
    Adds the constant darcy_flux to all internal and mortar faces, the latter
    in the "mortar_solution" field.
    gb - grid bucket
    upwind- upwind discretization class
    flux - 3 x 1 flux at all faces [u, v, w]
    a - cross-sectional area of the fractures.
    """
    for sd, data in mdg.subdomains(return_data=True):
        aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
        data[pp.PARAMETERS]["transport"]["darcy_flux"] = upwind.darcy_flux(
            sd, flux, aperture
        )
    for intf, data in mdg.interfaces(return_data=True):
        sd_primary, _ = mdg.interface_to_subdomain_pair(intf)
        param_primary = mdg.subdomain_data(sd_primary)[pp.PARAMETERS]
        darcy_flux = param_primary["transport"]["darcy_flux"]
        sign = np.zeros(sd_primary.num_faces)
        boundary_faces = sd_primary.get_all_boundary_faces()
        boundary_signs, _ = sd_primary.signs_and_cells_of_boundary_faces(boundary_faces)
        sign[boundary_faces] = boundary_signs

        sign = intf.primary_to_mortar_avg() * sign
        darcy_flux_e = sign * (intf.primary_to_mortar_avg() * darcy_flux)
        if pp.PARAMETERS not in data:
            data[pp.PARAMETERS] = pp.Parameters(
                intf, ["transport"], [{"darcy_flux": darcy_flux_e}]
            )
        else:
            data[pp.PARAMETERS]["transport"]["darcy_flux"] = darcy_flux_e


# #------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
