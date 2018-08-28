import unittest
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist

import porepy as pp


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
            d["param"] = pp.Parameters(g)
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            d["param"].set_bc("flow", pp.BoundaryCondition(g, dir_bc, "dir"))
            bc_val = np.zeros(g.num_faces)
            bc_val[left] = xmax
            bc_val[right] = 0
            d["param"].set_bc_val("flow", bc_val)

        flow_disc = pp.TpfaMixedDim()

        A, b = flow_disc.matrix_rhs(gb)

        x = sps.linalg.spsolve(A, b)

        flow_disc.split(gb, "pressure", x)

        # test pressure
        for g, d in gb:
            assert np.allclose(d["pressure"], xmax - g.cell_centers[0])

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e["mortar_grid"]
            g2, g1 = gb.nodes_of_edge(e)
            left_to_m = mg.left_to_mortar_avg()
            right_to_m = mg.right_to_mortar_avg()

            left_area = left_to_m * g1.face_areas
            right_area = right_to_m * g2.face_areas

            assert np.allclose(d_e["mortar_solution"] / left_area, 1)
            assert np.allclose(d_e["mortar_solution"] / right_area, 1)

    def generate_grids(self, n, xmax, ymax, split):
        g1 = pp.CartGrid([split * n, ymax * n], physdims=[split, ymax])
        g2 = pp.CartGrid([(xmax - split) * n, ymax * n], physdims=[xmax - split, ymax])
        g2.nodes[0] += split

        g1.compute_geometry()
        g2.compute_geometry()
        grids = [g2, g1]

        gb = pp.GridBucket()

        [gb.add_nodes(g) for g in grids]
        [g2, g1] = gb.grids_of_dimension(2)

        tol = 1e-6
        if np.any(g2.cell_centers[0] > split):
            right_grid = g2
            left_grid = g1
        else:
            right_grid = g1
            left_grid = g2

        gb.set_node_prop(left_grid, "node_number", 1)
        gb.set_node_prop(right_grid, "node_number", 0)
        left_faces = np.argwhere(left_grid.face_centers[0] > split - tol).ravel()
        right_faces = np.argwhere(right_grid.face_centers[0] < split + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [right_grid.num_faces, left_grid.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        gb.add_edge((right_grid, left_grid), face_faces)

        mg = pp.TensorGrid(np.array([split] * ((n + 1) * ymax)))
        mg.nodes[1] = np.linspace(0, ymax, (n + 1) * ymax)
        mg.compute_geometry()
        side_g = {pp.grids.mortar_grid.LEFT_SIDE: mg}

        d_e = gb.edge_props((g1, g2))
        d_e["mortar_grid"] = pp.BoundaryMortar(g1.dim - 1, side_g, face_faces)
        d_e["edge_number"] = 0

        return gb

    if __name__ == "__main__":
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
        gb = self.generate_2d_grid(n, xmax, ymax)
        tol = 1e-6

        def analytic_p(x):
            x = x.copy()
            shiftx = xmax / 4
            shifty = ymax / 3
            x[0] = x[0] - shiftx
            x[1] = x[1] - shifty
            p = np.sin(np.pi / xmax * x[0]) * np.cos(2 * np.pi / ymax * x[1])
            px = (
                (np.pi / xmax)
                * np.cos(np.pi / xmax * x[0])
                * np.cos(2 * np.pi / ymax * x[1])
            )
            py = (
                2
                * np.pi
                / ymax
                * np.sin(np.pi / xmax * x[0])
                * np.sin(2 * np.pi / ymax * x[1])
            )
            pxx = -(np.pi / xmax) ** 2 * p
            pyy = -(2 * np.pi / ymax) ** 2 * p
            return p, np.vstack([px, py]), pxx + pyy

        for g, d in gb:
            d["param"] = pp.Parameters(g)
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            d["param"].set_bc("flow", pp.BoundaryCondition(g, dir_bc, "dir"))
            bc_val = np.zeros(g.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(g.face_centers[:, dir_bc])

            d["param"].set_bc_val("flow", bc_val)

            pa, _, lpc = analytic_p(g.cell_centers)
            src = -lpc * g.cell_volumes
            d["param"].set_source("flow", src)

        self.solve(gb, analytic_p)

    def test_periodic_bc_with_fractues(self):
        """
        We set up the test case      P
                                |--------|
                                | g2|    |
                              D | g3x  g0| D
                                | g1|    |
                                |--------|
                                    P

        where D are dirichlet boundaries and P the periodic boundaries.
        We construct periodic solution
               p = sin(pi/xmax * (x - x0)) * cos(2*pi/ymax * (y - y0))
        which gives a source term on the rhs.

        The domain is a 2D domain with two fractures (g1, and g2) connected
        through the periodic bc and 0D intersection (g3)
        """
        n = 8
        xmax = 1
        ymax = 1
        gb = self.generate_grid_with_fractures(n, xmax, ymax)
        tol = 1e-6

        def analytic_p(x):
            x = x.copy()
            shiftx = xmax / 4
            shifty = ymax / 3
            x[0] = x[0] - shiftx
            x[1] = x[1] - shifty
            p = np.sin(np.pi / xmax * x[0]) * np.cos(2 * np.pi / ymax * x[1])
            px = (
                (np.pi / xmax)
                * np.cos(np.pi / xmax * x[0])
                * np.cos(2 * np.pi / ymax * x[1])
            )
            py = (
                2
                * np.pi
                / ymax
                * np.sin(np.pi / xmax * x[0])
                * np.sin(2 * np.pi / ymax * x[1])
            )
            pxx = -(np.pi / xmax) ** 2 * p
            pyy = -(2 * np.pi / ymax) ** 2 * p
            return p, np.vstack([px, py]), pxx + pyy

        for g, d in gb:
            d["param"] = pp.Parameters(g)
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            d["param"].set_bc("flow", pp.BoundaryCondition(g, dir_bc, "dir"))
            bc_val = np.zeros(g.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(g.face_centers[:, dir_bc])

            d["param"].set_bc_val("flow", bc_val)
            aperture = 1e-6 ** (2 - g.dim)
            d["param"].set_aperture(aperture)

            pa, _, lpc = analytic_p(g.cell_centers)
            src = -lpc * g.cell_volumes * aperture
            d["param"].set_source("flow", src)

        for _, d in gb.edges():
            d["kn"] = 1e10

        self.solve(gb, analytic_p)

    def solve(self, gb, analytic_p):
        flow_disc = pp.TpfaMixedDim()
        source_disc = pp.IntegralMixedDim(coupling=[None])

        _, src = source_disc.matrix_rhs(gb)

        A, b = flow_disc.matrix_rhs(gb)
        x = sps.linalg.spsolve(A, b + src)

        flow_disc.split(gb, "pressure", x)
        # test pressure
        for g, d in gb:
            ap, _, _ = analytic_p(g.cell_centers)
            assert np.max(np.abs(d["pressure"] - ap)) < 5e-2

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e["mortar_grid"]
            g2, g1 = gb.nodes_of_edge(e)
            try:
                left_to_m = mg.left_to_mortar_avg()
                right_to_m = mg.right_to_mortar_avg()
            except AttributeError:
                continue
            d1 = gb.node_props(g1)
            d2 = gb.node_props(g2)

            _, analytic_flux, _ = analytic_p(g1.face_centers)
            # the aperture is assumed constant
            a1 = np.max(d1["param"].aperture)
            left_flux = a1 * np.sum(analytic_flux * g1.face_normals[:2], 0)
            left_flux = left_to_m * (d1["bound_flux"] * left_flux)
            # right flux is negative lambda
            a2 = np.max(d2["param"].aperture)
            right_flux = a2 * np.sum(analytic_flux * g2.face_normals[:2], 0)
            right_flux = -right_to_m * (d2["bound_flux"] * right_flux)

            assert np.max(np.abs(d_e["mortar_solution"] - left_flux)) < 5e-2
            assert np.max(np.abs(d_e["mortar_solution"] - right_flux)) < 5e-2

    def generate_2d_grid(self, n, xmax, ymax):
        g1 = pp.CartGrid([xmax * n, ymax * n], physdims=[xmax, ymax])
        g1.compute_geometry()
        gb = pp.GridBucket()

        gb.add_nodes(g1)

        tol = 1e-6
        left_faces = np.argwhere(g1.face_centers[1] > ymax - tol).ravel()
        right_faces = np.argwhere(g1.face_centers[1] < 0 + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [g1.num_faces, g1.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        gb.add_edge((g1, g1), face_faces)

        mg = pp.TensorGrid(np.linspace(0, xmax, n + 1))
        mg.nodes[1] = ymax

        mg.compute_geometry()
        side_g = {pp.grids.mortar_grid.LEFT_SIDE: mg}

        d_e = gb.edge_props((g1, g1))
        d_e["mortar_grid"] = pp.BoundaryMortar(g1.dim - 1, side_g, face_faces)
        gb.assign_node_ordering()
        return gb

    def generate_grid_with_fractures(self, n, xmax, ymax):
        x = [xmax / 3] * 2
        y = [0, ymax / 2]
        fracs = [np.array([x, y])]
        y = [ymax / 2, ymax]
        fracs.append(np.array([x, y]))

        gb = pp.meshing.cart_grid(fracs, [xmax * n, ymax * n], physdims=[xmax, ymax])

        tol = 1e-6
        for g, d in gb:
            # map right faces to left
            left = np.argwhere(g.face_centers[1] < tol).ravel()
            right = np.argwhere(g.face_centers[1] > ymax - tol).ravel()
            g.face_centers[1, right] = 0

            g.left = left
            g.right = right

        # now create mappings between two grids of equal dimension
        for dim in [2, 1]:
            grids = gb.grids_of_dimension(dim)
            for i, gi in enumerate(grids):
                if gi.left.size < 1:
                    continue
                for j, gj in enumerate(grids):
                    if gj.right.size < 1:
                        continue
                    face_faces = self.match_grids(gi, gj)

                    if np.sum(face_faces) == 0:
                        continue

                    gb.add_edge((gi, gj), face_faces)
                    d_e = gb.edge_props((gi, gj))

                    di = gb.node_props(gi)
                    dj = gb.node_props(gj)

                    if di["node_number"] < dj["node_number"]:
                        # gj is left
                        g_m, _, _ = pp.partition.extract_subgrid(
                            gj, gj.right, faces=True
                        )
                        face_faces = face_faces.T
                    else:
                        # gi is left
                        g_m, _, _ = pp.partition.extract_subgrid(
                            gi, gi.left, faces=True
                        )

                    side_g = {pp.grids.mortar_grid.LEFT_SIDE: g_m}
                    d_e["mortar_grid"] = pp.BoundaryMortar(
                        gi.dim - 1, side_g, face_faces
                    )

        gb.compute_geometry()  # basically reset g.face_centers[,right]
        gb.assign_node_ordering()

        return gb

    def match_grids(self, gi, gj):
        tol = 1e-6
        fi = gi.face_centers[:, gi.left]
        fj = gj.face_centers[:, gj.right]

        left_right_faces = np.argwhere(np.abs(cdist(fi[:2].T, fj[:2].T) < tol))
        left = gi.left[left_right_faces[:, 0]]
        right = gj.right[left_right_faces[:, 1]]

        assert left.size == np.unique(left).size
        assert right.size == np.unique(right).size

        val = np.ones(left.shape, dtype=bool)
        shape = [gj.num_faces, gi.num_faces]

        face_faces = sps.coo_matrix((val, (right, left)), shape=shape)
        return face_faces


if __name__ == "__main__":
    unittest.main()
