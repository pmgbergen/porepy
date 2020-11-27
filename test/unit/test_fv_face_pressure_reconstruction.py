import unittest

import numpy as np
import scipy.sparse.linalg as spl

import porepy as pp


class MpfaReconstructPressure(unittest.TestCase):
    def test_cart_2d(self):
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[2, 2])
        g.compute_geometry()
        g = make_true_2d(g)
        sc_top = pp.fvutils.SubcellTopology(g)

        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=0)

        D_g_known = np.array(
            [
                [-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0],
                [0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        CC_known = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

        self.assertTrue(np.all(np.abs(D_g - D_g_known) < 1e-12))
        self.assertTrue(np.all(np.abs(CC - CC_known) < 1e-12))

    def test_simplex_2d(self):
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        g = make_true_2d(g)
        sc_top = pp.fvutils.SubcellTopology(g)

        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=0)
        D_g_known = np.array(
            [
                [-1 / 6, -1 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0],
                [0, 0, -1 / 6, -1 / 3, 0, 0, 0, 0, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 0, 0, -1 / 3, -1 / 6, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1 / 3, -1 / 6, 0, 0.0],
                [-1 / 12, 1 / 12, 0, 0, 0, 0, 1 / 12, -1 / 12, 0, 0, 0, 0],
                [0, 0, 0, 0, -1 / 12, 1 / 12, 0, 0, 0, 0, 1 / 12, -1 / 12],
                [0, 0, 1 / 3, 1 / 6, 0, 0, 0, 0, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 1 / 3, 1 / 6, 0, 0, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1 / 6, 1 / 3, 0, 0.0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 6, 1 / 3],
            ]
        )

        CC_known = np.array(
            [
                [1, 0.0],
                [1, 0.0],
                [0, 1.0],
                [0, 1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [1, 0.0],
                [1, 0.0],
                [0, 1.0],
                [0, 1.0],
            ]
        )

        self.assertTrue(np.all(np.abs(D_g - D_g_known).A < 1e-12))
        self.assertTrue(np.all(np.abs(CC - CC_known) < 1e-12))

    def test_cart_3d(self):
        g = pp.CartGrid([1, 1, 1], physdims=[2, 2, 2])
        g.compute_geometry()
        sc_top = pp.fvutils.SubcellTopology(g)

        D_g, CC = pp.numerics.fv.mpfa.reconstruct_presssure(g, sc_top, eta=1)

        D_g_known = np.array(
            [
                [
                    -1,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
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
                    0,
                ],
                [
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
                    0,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
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
                    -1,
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
                ],
                [
                    0,
                    0,
                    0,
                    1,
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
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
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
                ],
                [
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
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
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    -1,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
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
                    -1,
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
                ],
                [
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
                    -1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
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
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
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
                ],
                [
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
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],
                [
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
                    1,
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
                ],
                [
                    0,
                    0,
                    -1,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
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
                    0,
                    0,
                ],
                [
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
                    -1,
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
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
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
                ],
                [
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
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
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
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                [
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
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                ],
            ]
        )

        CC_known = np.array(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ]
        )

        self.assertTrue(np.all(np.abs(D_g - D_g_known).A < 1e-12))
        self.assertTrue(np.all(np.abs(CC - CC_known) < 1e-12))


class TestTpfaBoundaryPressure(unittest.TestCase):
    def grid(self, nx=[2, 2], physdims=None):
        if physdims is None:
            physdims = nx
        g = pp.CartGrid(nx, physdims)
        g.compute_geometry()
        return g

    def simplex_grid(self, nx=[2, 2]):
        g = pp.StructuredTriangleGrid(nx)
        g.compute_geometry()
        return g

    def pressure(self, fd, g, data):
        fd.discretize(g, data)
        A, b = fd.assemble_matrix_rhs(g, data)
        p = spl.spsolve(A, b)
        return p

    def test_zero_pressure(self):
        g = self.grid()
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        data = make_dictionary(g, bound)
        p = self.pressure(fd, g, data)
        self.assertTrue(np.allclose(p, np.zeros_like(p)))
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = matrix_dictionary["bound_pressure_cell"] * p + matrix_dictionary[
            "bound_pressure_face"
        ] * np.zeros(g.num_faces)
        self.assertTrue(np.allclose(bound_p, np.zeros_like(bound_p)))

    def test_constant_pressure(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.ones(g.num_faces)
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        self.assertTrue(np.allclose(p, np.ones_like(p)))

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.ones(g.num_faces)
        data = make_dictionary(g, bound, bc_val)

        p = self.pressure(fd, g, data)

        self.assertTrue(np.allclose(p, np.ones_like(p)))

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        data = make_dictionary(g, bound, bc_val)

        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = make_dictionary(g, bound, bc_val)

        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))

    def test_linear_pressure_part_neumann_conditions_small_domain(self):
        g = self.grid(physdims=[1, 1])

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -2 * g.face_centers[0, bf]))

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up pressure gradient in x-direction, with value -1
        bc_val[[2, 5]] = -1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], g.face_centers[0, bf]))

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = pp.CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))

    def test_sign_trouble_two_neumann_sides(self):
        g = pp.CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        bc_val[[0, 3]] = 1
        bc_val[[2, 5]] = -1
        t = pp.Tpfa("flow")
        data = make_dictionary(g, pp.BoundaryCondition(g), bc_val)
        t.discretize(g, data)
        t.assemble_matrix_rhs(g, data)
        # The problem is singular, and spsolve does not work well on all systems.
        # Instead, set a consistent solution, and check that the boundary
        # pressure is recovered.
        x = g.cell_centers[0]

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * x
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(bound_p[0] == x[0] - 0.5)
        self.assertTrue(bound_p[2] == x[1] + 0.5)


class TestMpfaBoundaryPressure(unittest.TestCase):
    def grid(self, nx=[2, 2]):
        g = pp.CartGrid(nx)
        g.compute_geometry()
        return g

    def simplex_grid(self, nx=[2, 2]):
        g = pp.StructuredTriangleGrid(nx)
        g.compute_geometry()
        return g

    def pressure(self, fd, g, data):
        fd.discretize(g, data)
        A, b = fd.assemble_matrix_rhs(g, data)
        p = spl.spsolve(A, b)
        return p

    def test_zero_pressure(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")
        data = make_dictionary(g, bound)

        p = self.pressure(fd, g, data)

        self.assertTrue(np.allclose(p, np.zeros_like(p)))

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = matrix_dictionary["bound_pressure_cell"] * p + matrix_dictionary[
            "bound_pressure_face"
        ] * np.zeros(g.num_faces)
        self.assertTrue(np.allclose(bound_p, np.zeros_like(bound_p)))

    def test_constant_pressure(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.ones(g.num_faces)
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        self.assertTrue(np.allclose(p, np.ones_like(p)))

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.ones(g.num_faces)
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        self.assertTrue(np.allclose(p, np.ones_like(p)))

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        # Set Dirichlet conditions. Note that indices are relative to bf,
        # that is, counting only boundary faces.
        bc_type[0] = "dir"
        bc_type[2] = "dir"  # Not [3]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up unit flow in x-direction, thus pressure gradient the other way
        bc_val[[2, 5]] = 1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Tpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up flow in negative x-direction, thus positive gradient of value 1
        bc_val[[2, 5]] = -1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], g.face_centers[0, bf]))

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = pp.CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))

    def test_linear_pressure_dirichlet_conditions_perturbed_grid(self):
        g = self.grid()
        g.nodes[:2] = g.nodes[:2] + np.random.random((2, g.num_nodes))
        g.compute_geometry()

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        data = make_dictionary(g, bound, bc_val)
        p = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], bc_val[bf]))

    def test_sign_trouble_two_neumann_sides(self):
        g = pp.CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        # Flow from right to left
        bc_val[[0, 3]] = 1
        bc_val[[2, 5]] = -1
        t = pp.Mpfa("flow")
        data = make_dictionary(g, pp.BoundaryCondition(g), bc_val)
        t.discretize(g, data)
        _, b = t.assemble_matrix_rhs(g, data)
        # The problem is singular, and spsolve does not work well on all systems.
        # Instead, set a consistent solution, and check that the boundary
        # pressure is recovered.
        x = g.cell_centers[0]

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * x
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(bound_p[0] == x[0] - 0.5)
        self.assertTrue(bound_p[2] == x[1] + 0.5)

    def test_structured_simplex_linear_flow(self):
        g = self.simplex_grid()
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[xright]] = 1
        data = make_dictionary(g, bound, bc_val)

        x = self.pressure(fd, g, data)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * x
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))

    def test_structured_simplex_linear_flow_reverse_sign(self):
        g = self.simplex_grid()
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[xright]] = -1

        data = make_dictionary(g, bound, bc_val)
        x = self.pressure(fd, g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * x
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], g.face_centers[0, bf]))


class TestMpfaSimplexGrid(unittest.TestCase):
    def grid(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        mesh_size = {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}
        network = pp.FractureNetwork2d(domain=domain)
        gb = network.mesh(mesh_size)
        return gb.grids_of_dimension(2)[0]

    def test_linear_flow(self):
        g = self.grid()
        g.compute_geometry()
        bc_val = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = pp.BoundaryCondition(g, bf, bc_type)

        fd = pp.Mpfa("flow")

        bc_val = np.zeros(g.num_faces)
        bc_val[bf[xright]] = 1 * g.face_areas[bf[xright]]

        data = make_dictionary(g, bound, bc_val)

        fd.discretize(g, data)
        A, b = fd.assemble_matrix_rhs(g, data)

        p = spl.spsolve(A, b)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES]["flow"]
        bound_p = (
            matrix_dictionary["bound_pressure_cell"] * p
            + matrix_dictionary["bound_pressure_face"] * bc_val
        )
        self.assertTrue(np.allclose(bound_p[bf], -g.face_centers[0, bf]))


def make_dictionary(g, bc, bc_values=None):
    if bc_values is None:
        bc_values = np.zeros(g.num_faces)
    d = {"bc": bc, "bc_values": bc_values, "mpfa_inverter": "python"}
    return pp.initialize_default_data(g, {}, "flow", d)


def make_true_2d(g):
    if g.dim == 2:
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

    return g


if __name__ == "__main__":
    unittest.main()
