"""
Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs
for handling boundary conditions expressed in a vectorial form
"""

import unittest
import numpy as np

import porepy as pp


class TestMpsaBoundRhs(unittest.TestCase):
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
        g = expand_2d_grid_to_3d(g)

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


def expand_2d_grid_to_3d(g, constit=None):
    g = g.copy()
    g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
    g.face_centers = np.delete(g.face_centers, (2), axis=0)
    g.face_normals = np.delete(g.face_normals, (2), axis=0)
    g.nodes = np.delete(g.nodes, (2), axis=0)
    if constit is None:
        return g
    else:
        constit = constit.copy()
        constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=0)
        constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=1)
        return g, constit


if __name__ == "__main__":
    unittest.main()
