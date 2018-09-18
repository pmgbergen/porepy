import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class MpsaBoundTest(unittest.TestCase):
    def test_dir_rob(self):
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = left + top
        is_neu = np.zeros(left.size, dtype=np.bool)
        is_rob = right + bot

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)

        rhs_known = np.array(
            [
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
            ]
        )
        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_dir_neu_rob(self):
        nx = 2
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = left
        is_neu = top
        is_rob = right + bot

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)
        rhs_known = np.array(
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            ]
        )

        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_dir(self):
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = left + top + right + bot
        is_neu = np.zeros(left.size, dtype=np.bool)
        is_rob = np.zeros(left.size, dtype=np.bool)

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)

        rhs_known = np.array(
            [
                [-1., 0., 0., 0., 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., -1., 0., 0., 0.],
                [0., 0., 0., 0., -1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0.],
                [0., -1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., -1., 0., 0.],
                [0., 0., 0., 0., 0., -1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
            ]
        )
        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_rob(self):
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = np.zeros(left.size, dtype=np.bool)
        is_neu = np.zeros(left.size, dtype=np.bool)
        is_rob = left + top + right + bot

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)

        rhs_known = np.array(
            [
                [0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0., 0., 0., 0., 0.5],
            ]
        )

        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_neu(self):
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = np.zeros(left.size, dtype=np.bool)
        is_neu = left + top + right + bot
        is_rob = np.zeros(left.size, dtype=np.bool)

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)

        rhs_known = np.array(
            [
                [0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0., 0., 0., 0., 0.5],
            ]
        )

        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_mixed_condition(self):
        nx = 2
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[0, left] = True
        bnd.is_neu[1, left] = True

        bnd.is_rob[0, right] = True
        bnd.is_dir[1, right] = True

        bnd.is_neu[0, bot] = True
        bnd.is_rob[1, bot] = True

        bnd.is_rob[0, top] = True
        bnd.is_dir[1, top] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)
        rhs_known = np.array(
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            ]
        )
        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_structured_triang(self):
        nx = 1
        ny = 1
        g = pp.StructuredTriangleGrid([nx, ny], physdims=[1, 1])
        g.compute_geometry()

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        is_dir = left
        is_neu = top
        is_rob = right + bot

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)
        rhs_known = np.array(
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
                [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            ]
        )

        self.assertTrue(np.all(np.abs(rhs_known - rhs) < 1e-12))

    def test_cart_3d(self):
        g = pp.CartGrid([1, 1, 1])
        g.compute_geometry()

        bot = g.face_centers[2] < 1e-10
        top = g.face_centers[2] > 1 - 1e-10
        south = g.face_centers[1] < 1e-10
        north = g.face_centers[1] > 1 - 1e-10
        west = g.face_centers[0] < 1e-10
        east = g.face_centers[0] > 1 - 1e-10

        is_dir = south + top
        is_neu = east + west
        is_rob = north + bot

        bnd = pp.BoundaryConditionVectorial(g)
        bnd.is_neu[:] = False

        bnd.is_dir[:, is_dir] = True
        bnd.is_rob[:, is_rob] = True
        bnd.is_neu[:, is_neu] = True

        sc_top = pp.fvutils.SubcellTopology(g)
        bnd_excl = pp.fvutils.ExcludeBoundaries(sc_top, bnd, g.dim)

        rhs = pp.numerics.fv.mpsa.create_bound_rhs_nd(bnd, bnd_excl, sc_top, g)

        rhs_indptr = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
            ],
            dtype=np.int32,
        )
        rhs_indices = np.array(
            [
                0,
                0,
                0,
                0,
                3,
                3,
                3,
                3,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                2,
                2,
                2,
                2,
                5,
                5,
                5,
                5,
                9,
                9,
                9,
                9,
                12,
                12,
                12,
                12,
                10,
                10,
                10,
                10,
                13,
                13,
                13,
                13,
                11,
                11,
                11,
                11,
                14,
                14,
                14,
                14,
                6,
                6,
                6,
                6,
                15,
                15,
                15,
                15,
                7,
                7,
                7,
                7,
                16,
                16,
                16,
                16,
                8,
                8,
                8,
                8,
                17,
                17,
                17,
                17,
            ],
            dtype=np.int32,
        )
        rhs_data = np.array(
            [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                -1.,
                -1.,
                -1.,
                -1.,
                1.,
                1.,
                1.,
                1.,
                -1.,
                -1.,
                -1.,
                -1.,
                1.,
                1.,
                1.,
                1.,
                -1.,
                -1.,
                -1.,
                -1.,
                1.,
                1.,
                1.,
                1.,
            ]
        )
        rhs_known = sps.csr_matrix((rhs_data, rhs_indices, rhs_indptr), shape=(72, 18))
        self.assertTrue(np.all(np.abs((rhs_known - rhs).data) < 1e-12))
