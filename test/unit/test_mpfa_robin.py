import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp

class RobinBoundTest(unittest.TestCase):
    def test_flow_left_right(self):
        nxs = [1, 3]
        nys = [1, 3]
        for nx in nxs:
            for ny in nys:
                g = pp.CartGrid([nx, ny], physdims=[1, 1])
                g.compute_geometry()
                k = pp.SecondOrderTensor(2, np.ones(g.num_cells))
                robin_weight = 1.5

                left = g.face_centers[0] < 1e-10
                right = g.face_centers[0] > 1 - 1e-10

                dir_ind = np.ravel(np.argwhere(left))
                rob_ind = np.ravel(np.argwhere(right))

                names = ['dir']*len(dir_ind) + ['rob'] * len(rob_ind)
                bnd_ind = np.hstack((dir_ind, rob_ind))
                bnd = pp.BoundaryCondition(g, bnd_ind, names)

                p_bound = 1
                rob_bound = 0
                C = (rob_bound - robin_weight*p_bound) / (robin_weight - p_bound)
                u_ex = - C * g.face_normals[0]
                p_ex = C * g.cell_centers[0] + p_bound
                self.solve_robin(g, k, bnd, robin_weight, p_bound, rob_bound, dir_ind, rob_ind, p_ex, u_ex)

    def test_flow_nz_rhs(self):
        nxs = [1, 3]
        nys = [1, 3]
        for nx in nxs:
            for ny in nys:
                g = pp.CartGrid([nx, ny], physdims=[1, 1])
                g.compute_geometry()
                k = pp.SecondOrderTensor(2, np.ones(g.num_cells))
                robin_weight = 1.5

                left = g.face_centers[0] < 1e-10
                right = g.face_centers[0] > 1 - 1e-10

                dir_ind = np.ravel(np.argwhere(left))
                rob_ind = np.ravel(np.argwhere(right))

                names = ['dir']*len(dir_ind) + ['rob'] * len(rob_ind)
                bnd_ind = np.hstack((dir_ind, rob_ind))
                bnd = pp.BoundaryCondition(g, bnd_ind, names)

                p_bound = 1
                rob_bound = 1

                C = (rob_bound - robin_weight*p_bound) / (robin_weight - p_bound)
                u_ex = - C * g.face_normals[0]
                p_ex = C * g.cell_centers[0] + p_bound
                self.solve_robin(g, k, bnd, robin_weight, p_bound, rob_bound, dir_ind, rob_ind, p_ex, u_ex)
    def test_flow_down(self):
        nxs = [1, 3]
        nys = [1, 3]
        for nx in nxs:
            for ny in nys:
                g = pp.CartGrid([nx, ny], physdims=[1, 1])
                g.compute_geometry()
                k = pp.SecondOrderTensor(2, np.ones(g.num_cells))
                robin_weight = 1.5

                bot = g.face_centers[1] < 1e-10
                top = g.face_centers[1] > 1 - 1e-10

                dir_ind = np.ravel(np.argwhere(top))
                rob_ind = np.ravel(np.argwhere(bot))

                names = ['dir']*len(dir_ind) + ['rob'] * len(rob_ind)
                bnd_ind = np.hstack((dir_ind, rob_ind))
                bnd = pp.BoundaryCondition(g, bnd_ind, names)

                p_bound = 1
                rob_bound = 1
                C = (rob_bound - robin_weight*p_bound) / (robin_weight - p_bound)
                u_ex = C * g.face_normals[1]
                p_ex = C *(1- g.cell_centers[1]) + p_bound
                self.solve_robin(g, k, bnd, robin_weight, p_bound, rob_bound, dir_ind, rob_ind, p_ex, u_ex)

    def test_no_neumann(self):
        g = pp.CartGrid([2, 2], physdims=[1, 1])
        g.compute_geometry()
        k = pp.SecondOrderTensor(2, np.ones(g.num_cells))
        robin_weight = 1.5

        bot = g.face_centers[1] < 1e-10
        top = g.face_centers[1] > 1 - 1e-10
        left = g.face_centers[0] < 1e-10
        right = g.face_centers[0] > 1 - 1e-10

        dir_ind = np.ravel(np.argwhere(top + left))
        rob_ind = np.ravel(np.argwhere(bot + right))

        names = ['dir']*len(dir_ind) + ['rob'] * len(rob_ind)
        bnd_ind = np.hstack((dir_ind, rob_ind))
        bnd = pp.BoundaryCondition(g, bnd_ind, names)

        flux, bound_flux, _, _ = pp.numerics.fv.mpfa._mpfa_local(g, k, bnd, robin_weight=robin_weight)

        div = pp.fvutils.scalar_divergence(g)

        rob_ex = [robin_weight * .25, robin_weight * .75, 1, 1]
        u_bound = np.zeros(g.num_faces)
        u_bound[dir_ind] = g.face_centers[1, dir_ind]
        u_bound[rob_ind] = rob_ex * g.face_areas[rob_ind]


        a = div * flux
        b = -div * bound_flux * u_bound

        p = np.linalg.solve(a.A, b)

        u_ex = [np.dot(g.face_normals[:, f], np.array([0, -1, 0]))
                for f in range(g.num_faces)]
        u_ex = np.array(u_ex).ravel('F')
        p_ex = g.cell_centers[1]

        assert np.allclose(p, p_ex)
        assert np.allclose(flux * p + bound_flux * u_bound, u_ex)

    def solve_robin(self, g, k, bnd, robin_weight, p_bound, rob_bound, dir_ind, rob_ind, p_ex, u_ex):
        flux, bound_flux, _, _ = pp.numerics.fv.mpfa._mpfa_local(g, k, bnd, robin_weight=robin_weight)

        div = pp.fvutils.scalar_divergence(g)

        u_bound = np.zeros(g.num_faces)
        u_bound[dir_ind] = p_bound
        u_bound[rob_ind] = rob_bound * g.face_areas[rob_ind]


        a = div * flux
        b = -div * bound_flux * u_bound

        p = np.linalg.solve(a.A, b)
        assert np.allclose(p, p_ex)
        assert np.allclose(flux * p + bound_flux * u_bound, u_ex)
