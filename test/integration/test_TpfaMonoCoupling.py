import unittest
import numpy as np
import scipy.sparse as sps

from porepy.grids import structured, mortar_grid, grid_bucket
from porepy.numerics.fv import tpfa, fvutils, source
from porepy.params.data import Parameters
from porepy.params.bc import BoundaryCondition

class TestTpfaCouplingDiffGrids(unittest.TestCase):

    def test_two_cart_grids(self):
        """
        We set up the test case -----|---------
                                |    |        |
                                | g1 |    g2  |
                                |    |        |
                                -----|---------
        with a linear pressure increase from left to right
        """
        n = 2
        xmax = 3
        ymax = 1
        split = 2
        gb = self.generate_grids(n, xmax, ymax, split)
        tol = 1e-6
        for g, d in gb:
            d['param'] = Parameters(g)
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            d['param'].set_bc('flow', BoundaryCondition(g, dir_bc, 'dir'))
            bc_val = np.zeros(g.num_faces)
            bc_val[left] = xmax
            bc_val[right] = 0
            d['param'].set_bc_val('flow', bc_val)

        flow_disc = tpfa.TpfaMixedDim()

        A, b = flow_disc.matrix_rhs(gb)

        x = sps.linalg.spsolve(A, b)

        flow_disc.split(gb, 'pressure', x)

        # test pressure
        for g, d in gb:
            assert np.allclose(d['pressure'], xmax - g.cell_centers[0])

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e['mortar_grid']
            g2, g1 = gb.nodes_of_edge(e)
            left_to_m = mg.left_to_mortar_avg()
            right_to_m = mg.right_to_mortar_avg()

            left_area = left_to_m * g1.face_areas
            right_area = right_to_m * g2.face_areas

            assert np.allclose(d_e['mortar_solution']/left_area, 1)
            assert np.allclose(d_e['mortar_solution']/right_area, 1)

    def generate_grids(self, n, xmax, ymax, split):
        g1 = structured.CartGrid([split * n, ymax * n], physdims=[split, ymax])
        g2 = structured.CartGrid([(xmax-split) * n, ymax * n], physdims=[xmax-split, ymax])
        g2.nodes[0] += split

        g1.compute_geometry()
        g2.compute_geometry()
        grids = [g2, g1]

        gb = grid_bucket.GridBucket()

        [gb.add_nodes(g) for g in grids]

        tol = 1e-6
        left_faces = np.argwhere(g1.face_centers[0] > split - tol).ravel()
        right_faces = np.argwhere(g2.face_centers[0] < split + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [g2.num_faces, g1.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        gb.add_edge((g1, g2), face_faces)
        side_g = {mortar_grid.LEFT_SIDE: g1.copy(),
                  mortar_grid.RIGHT_SIDE: g2.copy()}
        d_e = gb.edge_props((g1, g2))
        d_e['mortar_grid'] = mortar_grid.BoundaryMortar(g1.dim -1, side_g, face_faces)
        gb.assign_node_ordering()

        return gb

    if __name__ == '__main__':
        unittest.main()


class TestTpfaCouplingPeriodicBc(unittest.TestCase):

    def test_periodic_bc(self):
        """
        We set up the test case      P
                                |--------|
                                |        |
                              D |    g2  | D
                                |        |
                                |--------|
                                    P

        where D are dirichlet boundaries and P the periodic boundaries.
        We construct periodic solution
               p = sin(pi/xmax * (x - x0)) * cos(2*pi/ymax * (y - y0))
        which gives a source term on the rhs.
        """
        n = 8
        xmax = 1
        ymax = 1
        gb = self.generate_grids(n, xmax, ymax)
        tol = 1e-6

        def analytic_p(x):
            x = x.copy()
            shiftx = xmax / 4
            shifty = ymax / 3
            x[0] = x[0] - shiftx
            x[1] = x[1] - shifty
            p = np.sin(np.pi/xmax * x[0]) * np.cos(2*np.pi/ymax * x[1])
            px = (np.pi / xmax) * np.cos(np.pi/xmax * x[0]) * np.cos(2*np.pi/ymax * x[1])
            py = 2*np.pi / ymax * np.sin(np.pi/xmax * x[0]) * np.sin(2*np.pi/ymax * x[1])
            pxx = -(np.pi / xmax)**2 * p
            pyy = -(2*np.pi / ymax)**2 * p
            return p, np.vstack([px, py]), pxx + pyy

        for g, d in gb:
            d['param'] = Parameters(g)
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            d['param'].set_bc('flow', BoundaryCondition(g, dir_bc, 'dir'))
            bc_val = np.zeros(g.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(g.face_centers[:, dir_bc])

            d['param'].set_bc_val('flow', bc_val)

            pa, _, lpc = analytic_p(g.cell_centers)
            src =  -lpc * g.cell_volumes
            d['param'].set_source('flow', src)

        flow_disc = tpfa.TpfaMixedDim()
        source_disc = source.IntegralMixedDim()

        _, src = source_disc.matrix_rhs(gb)

        A, b = flow_disc.matrix_rhs(gb)

        x = sps.linalg.spsolve(A, b + src)

        flow_disc.split(gb, 'pressure', x)

        # test pressure
        for g, d in gb:
            ap, _, _ = analytic_p(g.cell_centers)
            assert np.max(np.abs(d['pressure'] - ap)) < 5e-2

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e['mortar_grid']
            g2, g1 = gb.nodes_of_edge(e)
            left_to_m = mg.left_to_mortar_avg()
            right_to_m = mg.right_to_mortar_avg()

            _, analytic_flux,_ = analytic_p(g1.face_centers)
            left_flux = left_to_m * np.sum(analytic_flux * g1.face_normals[:2], 0)
            # two the right normals point the wrong way.. need to flip them
            right_flux = -right_to_m * np.sum(analytic_flux * (-g1.face_normals[:2]), 0)
            assert np.max(np.abs(d_e['mortar_solution'] - left_flux)) <5e-2
            assert np.max(np.abs(d_e['mortar_solution'] - right_flux)) < 5e-2

    def generate_grids(self, n, xmax, ymax):
        g1 = structured.CartGrid([xmax * n, ymax * n], physdims=[xmax, ymax])
        g1.compute_geometry()
        gb = grid_bucket.GridBucket()

        gb.add_nodes(g1)

        tol = 1e-6
        left_faces = np.argwhere(g1.face_centers[1] > ymax - tol).ravel()
        right_faces = np.argwhere(g1.face_centers[1] < 0 + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [g1.num_faces, g1.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        gb.add_edge((g1, g1), face_faces)
        side_g = {mortar_grid.LEFT_SIDE: g1.copy()}
        d_e = gb.edge_props((g1, g1))
        d_e['mortar_grid'] = mortar_grid.BoundaryMortar(g1.dim -1, side_g, face_faces)
        gb.assign_node_ordering()
        return gb

if __name__ == '__main__':
    unittest.main()