""" Various tests of the upwind discretization for tranpsort problems on a single grid.

Both within a grid, and upwind coupling on mortar grids.
"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestUpwindDiscretization(unittest.TestCase):
    """Upwind discretization on individual grids."""

    def test_upwind_1d_darcy_flux_positive(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])

        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2, 0, 0], [-2, 2, 0], [0, -2, 0]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_1d_darcy_flux_negative(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_cart_darcy_flux_positive(self):
        g = pp.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 0],
            ]
        )

        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_cart_darcy_flux_negative(self):
        g = pp.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array(
            [
                [0, -1, 0, 0, 0, 0],
                [0, 1, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 1, -1],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_simplex_darcy_flux_positive(self):
        g = pp.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_simplex_darcy_flux_negative(self):
        g = pp.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_3d_cart_darcy_flux_negative(self):
        g = pp.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array(
            [
                [0, -1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        deltaT_known = 1 / 4

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_3d_cart_darcy_flux_positive(self):
        g = pp.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, -1, 0],
            ]
        )

        deltaT_known = 1 / 4

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_1d_surf_darcy_flux_positive(self):
        g = pp.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(-np.pi / 5.0, [0, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 0]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_1d_surf_darcy_flux_negative(self):
        g = pp.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(-np.pi / 8.0, [-1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -1, 0], [0, 1, -1], [0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_cart_surf_darcy_flux_positive(self):
        g = pp.CartGrid([3, 2], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [0, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 0],
            ]
        )

        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_cart_surf_darcy_flux_negative(self):
        g = pp.CartGrid([3, 2], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array(
            [
                [0, -1, 0, 0, 0, 0],
                [0, 1, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 1, -1],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_simplex_surf_darcy_flux_positive(self):
        g = pp.StructuredTriangleGrid([2, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 2.0, [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_2d_simplex_surf_darcy_flux_negative(self):
        g = pp.StructuredTriangleGrid([2, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(-np.pi / 5.0, [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_1d_darcy_flux_negative_bc_dir(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        bc_val = 3 * np.ones(g.num_faces).ravel("F")
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M, rhs = solver.assemble_matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2, -2, 0], [0, 2, -2], [0, 0, 2]])
        rhs_known = np.array([0, 0, 6])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M.todense(), M_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    def test_upwind_1d_darcy_flux_negative_bc_neu(self):
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        solver = pp.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["neu"])
        bc_val = np.array([2, 0, 0, -2]).ravel("F")
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M, rhs = solver.assemble_matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
        rhs_known = np.array([-2, 0, 2])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M.todense(), M_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # Below follows other tests

    def test_upwind_example_1(self, if_export=False):
        #######################
        # Simple 2d upwind problem with explicit Euler scheme in time
        #######################
        T = 1
        Nx, Ny = 4, 1
        g = pp.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        advect = pp.Upwind("transport")
        dis = advect.darcy_flux(g, [1, 0, 0])

        b_faces = g.get_all_boundary_faces()
        bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        time_step = advect.cfl(g, data)
        data[pp.PARAMETERS]["transport"]["time_step"] = time_step

        advect.discretize(g, data)

        U, rhs = advect.assemble_matrix_rhs(g, data)
        rhs = time_step * rhs
        U = time_step * U
        OF = advect.outflow(g, data)
        mass = pp.MassMatrix("transport")
        mass.discretize(g, data)
        M, _ = mass.assemble_matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)

        M_minus_U = M - U
        inv_mass = pp.InvMassMatrix("transport")
        inv_mass.discretize(g, data)
        invM, _ = inv_mass.assemble_matrix_rhs(g, data)

        # Loop over the time
        Nt = int(T / time_step)
        time = np.empty(Nt)
        production = np.zeros(Nt)
        for i in np.arange(Nt):

            # Update the solution
            production[i] = np.sum(OF.dot(conc))
            conc = invM.dot((M_minus_U).dot(conc) + rhs)
            time[i] = time_step * i

        known = 1.09375
        assert np.sum(production) == known

    def test_upwind_example_2(self, if_export=False):
        #######################
        # Simple 2d upwind problem with implicit Euler scheme in time
        #######################
        T = 1
        Nx, Ny = 10, 1
        g = pp.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        advect = pp.Upwind("transport")
        dis = advect.darcy_flux(g, [1, 0, 0])

        b_faces = g.get_all_boundary_faces()
        bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        time_step = advect.cfl(g, data)
        data[pp.PARAMETERS]["transport"]["time_step"] = time_step

        advect.discretize(g, data)
        U, rhs = advect.assemble_matrix_rhs(g, data)
        rhs = time_step * rhs
        U = time_step * U
        mass = pp.MassMatrix("transport")
        mass.discretize(g, data)
        M, _ = mass.assemble_matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)

        # Perform an LU factorization to speedup the solver
        IE_solver = sps.linalg.factorized((M + U).tocsc())

        # Loop over the time
        Nt = int(T / time_step)
        time = np.empty(Nt)
        for i in np.arange(Nt):
            # Update the solution
            # Backward and forward substitution to solve the system
            conc = IE_solver(M.dot(conc) + rhs)
            time[i] = time_step * i

        known = np.array(
            [
                0.99969927,
                0.99769441,
                0.99067741,
                0.97352474,
                0.94064879,
                0.88804726,
                0.81498958,
                0.72453722,
                0.62277832,
                0.51725056,
            ]
        )
        assert np.allclose(conc, known)

    def test_upwind_example_3(self, if_export=False):
        #######################
        # Simple 2d upwind problem with explicit Euler scheme in time coupled with
        # a Darcy problem
        #######################
        T = 2
        Nx, Ny = 10, 10

        def funp_ex(pt):
            return -np.sin(pt[0]) * np.sin(pt[1]) - pt[0]

        g = pp.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        # Permeability
        perm = pp.SecondOrderTensor(kxx=np.ones(g.num_cells))

        # Boundaries
        b_faces = g.get_all_boundary_faces()
        bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.zeros(g.num_faces)
        bc_val[b_faces] = funp_ex(g.face_centers[:, b_faces])
        specified_parameters = {
            "bc": bc,
            "bc_values": bc_val,
            "second_order_tensor": perm,
        }
        data = pp.initialize_default_data(g, {}, "flow", specified_parameters)
        solver = pp.MVEM("flow")
        solver.discretize(g, data)
        D_flow, b_flow = solver.assemble_matrix_rhs(g, data)

        solver_source = pp.DualScalarSource("flow")
        solver_source.discretize(g, data)
        D_source, b_source = solver_source.assemble_matrix_rhs(g, data)

        up = sps.linalg.spsolve(D_flow + D_source, b_flow + b_source)
        _, u = solver.extract_pressure(g, up, data), solver.extract_flux(g, up, data)

        # Darcy_Flux
        dis = u

        # Boundaries
        bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        # Advect solver
        advect = pp.Upwind("transport")
        advect.discretize(g, data)

        U, rhs = advect.assemble_matrix_rhs(g, data)
        time_step = advect.cfl(g, data)
        rhs = time_step * rhs
        U = time_step * U

        data[pp.PARAMETERS]["transport"]["time_step"] = time_step
        mass = pp.MassMatrix("transport")
        mass.discretize(g, data)
        M, _ = mass.assemble_matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)
        M_minus_U = M - U
        inv_mass = pp.InvMassMatrix("transport")
        inv_mass.discretize(g, data)
        invM, _ = inv_mass.assemble_matrix_rhs(g, data)

        # Loop over the time
        Nt = int(T / time_step)
        time = np.empty(Nt)
        for i in np.arange(Nt):

            # Update the solution
            conc = invM.dot((M_minus_U).dot(conc) + rhs)
            time[i] = time_step * i

        known = np.array(
            [
                9.63168200e-01,
                8.64054875e-01,
                7.25390695e-01,
                5.72228235e-01,
                4.25640080e-01,
                2.99387331e-01,
                1.99574336e-01,
                1.26276876e-01,
                7.59011550e-02,
                4.33431230e-02,
                3.30416807e-02,
                1.13058617e-01,
                2.05372538e-01,
                2.78382057e-01,
                3.14035373e-01,
                3.09920132e-01,
                2.75024694e-01,
                2.23163145e-01,
                1.67386939e-01,
                1.16897527e-01,
                1.06111312e-03,
                1.11951850e-02,
                3.87907727e-02,
                8.38516119e-02,
                1.36617802e-01,
                1.82773271e-01,
                2.10446545e-01,
                2.14651936e-01,
                1.97681518e-01,
                1.66549151e-01,
                3.20751341e-05,
                9.85780113e-04,
                6.07062715e-03,
                1.99393042e-02,
                4.53237556e-02,
                8.00799828e-02,
                1.17199623e-01,
                1.47761481e-01,
                1.64729339e-01,
                1.65390555e-01,
                9.18585872e-07,
                8.08267622e-05,
                8.47227168e-04,
                4.08879583e-03,
                1.26336029e-02,
                2.88705048e-02,
                5.27841497e-02,
                8.10459333e-02,
                1.07956484e-01,
                1.27665318e-01,
                2.51295298e-08,
                6.29844122e-06,
                1.09361990e-04,
                7.56743783e-04,
                3.11384414e-03,
                9.04446601e-03,
                2.03443897e-02,
                3.75208816e-02,
                5.89595194e-02,
                8.11457277e-02,
                6.63498510e-10,
                4.73075468e-07,
                1.33728945e-05,
                1.30243418e-04,
                7.01905707e-04,
                2.55272292e-03,
                6.96686157e-03,
                1.52290448e-02,
                2.78607282e-02,
                4.40402650e-02,
                1.71197497e-11,
                3.47118057e-08,
                1.57974045e-06,
                2.13489614e-05,
                1.48634295e-04,
                6.68104990e-04,
                2.18444135e-03,
                5.58646819e-03,
                1.17334966e-02,
                2.09744728e-02,
                4.37822313e-13,
                2.52373622e-09,
                1.83589660e-07,
                3.40553325e-06,
                3.02948532e-05,
                1.66504215e-04,
                6.45119867e-04,
                1.90731440e-03,
                4.53436628e-03,
                8.99977737e-03,
                1.12627412e-14,
                1.84486857e-10,
                2.13562387e-08,
                5.39492977e-07,
                6.08223906e-06,
                4.05535296e-05,
                1.84731221e-04,
                6.25871542e-04,
                1.66459389e-03,
                3.59980231e-03,
            ]
        )

        assert np.allclose(conc, known)


class TestUpwindCoupling(unittest.TestCase):
    """Tests of hyperbolic_interface_laws.UpwindCoupling."""

    def generate_grid(self):
        # Generate cartesian grid with one fracture:
        # ---------
        # |   |   |
        # --------- horizontal fracture
        # |   |   |
        # --------
        gb, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)
        return gb

    def block_matrix(self, gs):
        def ndof(g):
            return g.num_cells

        dof = np.array([ndof(g) for g in gs])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        return cc.reshape((3, 3))

    def test_upwind_2d_1d_positive_flux(self):
        # test coupling between 2D grid and 1D grid with a fluid flux going from
        # 2D grid to 1D grid. The upwind weighting should in this case choose the
        # 2D cell variables as weights

        gb = self.generate_grid()
        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d2 = gb.node_props(g2)
        d1 = gb.node_props(g1)
        de = gb.edge_props((g1, g2))

        zero_mat = self.block_matrix([g2, g1, de["mortar_grid"]])

        lam = np.arange(de["mortar_grid"].num_cells)
        de[pp.PARAMETERS] = {"transport": {"darcy_flux": lam}}
        de[pp.DISCRETIZATION_MATRICES] = {"transport": {}}

        upwind_coupler = pp.UpwindCoupling("transport")
        upwind_coupler.discretize(g2, g1, d2, d1, de)

        matrix, _ = upwind_coupler.assemble_matrix_rhs(g2, g1, d2, d1, de, zero_mat)

        matrix_2 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        matrix_1 = np.array(
            [[0, 0, 0, 0, 0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, -1]]
        )
        matrix_l = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, -1, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
        self.assertTrue(np.allclose(sps.hstack(matrix[0, :]).A, matrix_2))
        self.assertTrue(np.allclose(sps.hstack(matrix[1, :]).A, matrix_1))
        self.assertTrue(np.allclose(sps.hstack(matrix[2, :]).A, matrix_l))

    def test_upwind_2d_1d_negative_flux(self):
        # test coupling between 2D grid and 1D grid with a fluid flux going from
        # 1D grid to 2D grid. The upwind weighting should in this case choose the
        # 1D cell variables as weights

        gb = self.generate_grid()
        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d2 = gb.node_props(g2)
        d1 = gb.node_props(g1)
        de = gb.edge_props((g1, g2))

        zero_mat = self.block_matrix([g2, g1, de["mortar_grid"]])

        lam = np.arange(de["mortar_grid"].num_cells)
        de[pp.PARAMETERS] = {"transport": {"darcy_flux": -lam}}
        de[pp.DISCRETIZATION_MATRICES] = {"transport": {}}

        upwind_coupler = pp.UpwindCoupling("transport")

        upwind_coupler.discretize(g2, g1, d2, d1, de)
        matrix, _ = upwind_coupler.assemble_matrix_rhs(g2, g1, d2, d1, de, zero_mat)

        matrix_2 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        matrix_1 = np.array(
            [[0, 0, 0, 0, 0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, -1]]
        )
        matrix_l = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, -1, 0, 0],
                [0, 0, 0, 0, -2, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, -3, 0, 0, 0, -1],
            ]
        )

        self.assertTrue(np.allclose(sps.hstack(matrix[0, :]).A, matrix_2))
        self.assertTrue(np.allclose(sps.hstack(matrix[1, :]).A, matrix_1))
        self.assertTrue(np.allclose(sps.hstack(matrix[2, :]).A, matrix_l))


if __name__ == "__main__":
    unittest.main()
