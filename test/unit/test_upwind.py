from __future__ import division
import numpy as np
import unittest
import porepy as pp
from porepy.grids import structured, simplex
from porepy.params.bc import BoundaryCondition
from porepy.numerics.fv import upwind

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_darcy_flux_positive(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])

        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2, 0, 0], [-2, 2, 0], [0, -2, 0]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_darcy_flux_negative(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_darcy_flux_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_darcy_flux_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_darcy_flux_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_darcy_flux_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_3d_cart_darcy_flux_negative(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_3d_cart_darcy_flux_positive(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_surf_darcy_flux_positive(self):
        g = structured.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(-np.pi / 5.0, [0, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 0]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_surf_darcy_flux_negative(self):
        g = structured.CartGrid(3, 1)
        R = pp.map_geometry.rotation_matrix(-np.pi / 8.0, [-1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -1, 0], [0, 1, -1], [0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_darcy_flux_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [0, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_darcy_flux_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 6.0, [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_darcy_flux_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 2.0, [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_darcy_flux_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = pp.map_geometry.rotation_matrix(-np.pi / 5.0, [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
        M = solver.assemble_matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M, M_known, rtol, atol))
        self.assertTrue(np.allclose(deltaT, deltaT_known, rtol, atol))

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_darcy_flux_negative_bc_dir(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["dir"])
        bc_val = 3 * np.ones(g.num_faces).ravel("F")
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_darcy_flux_negative_bc_neu(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        dis = solver.darcy_flux(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        bc_val = np.array([2, 0, 0, -2]).ravel("F")
        specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)
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


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
