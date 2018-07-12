from __future__ import division
import numpy as np
import unittest

from porepy.grids import structured, simplex
import porepy.utils.comp_geom as cg
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters
from porepy.numerics.fv.transport import upwind

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_discharge_positive(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2, 0, 0], [-2, 2, 0], [0, -2, 0]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_discharge_negative(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_discharge_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_discharge_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_discharge_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_discharge_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_3d_cart_discharge_negative(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_3d_cart_discharge_positive(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [1, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_surf_discharge_positive(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(-np.pi / 5., [0, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 0]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_surf_discharge_negative(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(-np.pi / 8., [-1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -1, 0], [0, 1, -1], [0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_discharge_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = cg.rot(np.pi / 4., [0, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_discharge_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = cg.rot(np.pi / 6., [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
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
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_discharge_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = cg.rot(np.pi / 2., [1, 1, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_discharge_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = cg.rot(-np.pi / 5., [1, 1, -1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, np.dot(R, [-1, 0, 0]))

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        param.set_bc(solver, bc)

        data = {"param": param, "discharge": dis}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
        deltaT_known = 1 / 6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_discharge_negative_bc_dir(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["dir"])
        bc_val = 3 * np.ones(g.num_faces).ravel("F")
        param.set_bc(solver, bc)
        param.set_bc_val(solver, bc_val)

        data = {"param": param, "discharge": dis}
        M, rhs = solver.matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2, -2, 0], [0, 2, -2], [0, 0, 2]])
        rhs_known = np.array([0, 0, 6])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

    # ------------------------------------------------------------------------------#

    def test_upwind_1d_discharge_negative_bc_neu(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        param = Parameters(g)
        dis = solver.discharge(g, [-2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = BoundaryCondition(g, bf, bf.size * ["neu"])
        bc_val = np.array([2, 0, 0, -2]).ravel("F")
        param.set_bc(solver, bc)
        param.set_bc_val(solver, bc_val)

        data = {"param": param, "discharge": dis}
        M, rhs = solver.matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
        rhs_known = np.array([-2, 0, 2])
        deltaT_known = 1 / 12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
