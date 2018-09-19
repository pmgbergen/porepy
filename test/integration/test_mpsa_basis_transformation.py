import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp

"""
Checks the actions done in porepy.numerics.fv.mpsa.create_bound_rhs_nd
for handling boundary conditions expressed in a vectorial form
"""

class TestMpsaBoundRhs(unittest.TestCase):
    def test_neu(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.num_faces, g.dim)
        bc = pp.BoundaryConditionVectorial(g)
        self.run_test(g, basis, bc)

    def test_dir(self):
        g = pp.StructuredTriangleGrid([1, 1])
        basis = np.random.rand(g.dim, g.num_faces, g.dim)
        bc = pp.BoundaryConditionVectorial(g,g.get_all_boundary_faces(), 'dir')
        self.run_test(g, basis, bc)

    def test_mix(self):
        nx = 2
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()
        basis = np.random.rand(g.dim, g.num_faces, g.dim)

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
        be = pp.fvutils.ExcludeBoundaries(st, bc, g.dim)

        bound_rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bc, be, st, g)

        bc.basis = basis
        be = pp.fvutils.ExcludeBoundaries(st, bc, g.dim)
        bound_rhs_b = pp.numerics.fv.mpsa.create_bound_rhs_nd(bc, be, st, g)

        # rhs should not be affected by basis transform
        self.assertTrue(np.allclose(bound_rhs_b.A, bound_rhs.A))

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
