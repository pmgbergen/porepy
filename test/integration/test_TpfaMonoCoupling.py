import unittest
from test import test_utils

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
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(g, dir_bc, "dir")
            bc_val = np.zeros(g.num_faces)
            bc_val[left] = xmax
            bc_val[right] = 0
            specified_parameters = {"bc": bound, "bc_values": bc_val}
            pp.initialize_default_data(g, d, "flow", specified_parameters)

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{}])
            pp.params.data.add_discretization_matrix_keyword(d, "flow")
        # assign discretization
        data_key = "flow"
        tpfa = pp.Tpfa(data_key)
        coupler = pp.FluxPressureContinuity(data_key, tpfa)
        assembler = test_utils.setup_flow_assembler(gb, tpfa, data_key, coupler=coupler)
        test_utils.solve_and_distribute_pressure(gb, assembler)

        # test pressure
        for g, d in gb:
            self.assertTrue(
                np.allclose(d[pp.STATE]["pressure"], xmax - g.cell_centers[0])
            )

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e["mortar_grid"]
            g2, g1 = gb.nodes_of_edge(e)
            primary_to_m = mg.primary_to_mortar_avg()
            secondary_to_m = mg.secondary_to_mortar_avg()

            primary_area = primary_to_m * g1.face_areas
            secondary_area = secondary_to_m * g2.face_areas

            self.assertTrue(np.allclose(d_e[pp.STATE]["mortar_flux"] / primary_area, 1))
            self.assertTrue(
                np.allclose(d_e[pp.STATE]["mortar_flux"] / secondary_area, 1)
            )

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
        d_e = gb.edge_props((g1, g2))
        d_e["mortar_grid"] = pp.MortarGrid(g1.dim - 1, {"0": mg}, face_faces)
        d_e["edge_number"] = 0

        return gb


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
            pxx = -((np.pi / xmax) ** 2) * p
            pyy = -((2 * np.pi / ymax) ** 2) * p
            return p, np.vstack([px, py]), pxx + pyy

        for g, d in gb:
            left = g.face_centers[0] < tol
            right = g.face_centers[0] > xmax - tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(g, dir_bc, "dir")
            bc_val = np.zeros(g.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(g.face_centers[:, dir_bc])

            pa, _, lpc = analytic_p(g.cell_centers)
            src = -lpc * g.cell_volumes
            specified_parameters = {"bc": bound, "bc_values": bc_val, "source": src}
            pp.initialize_default_data(g, d, "flow", specified_parameters)

        for _, d in gb.edges():
            pp.params.data.add_discretization_matrix_keyword(d, "flow")
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
            pxx = -((np.pi / xmax) ** 2) * p
            pyy = -((2 * np.pi / ymax) ** 2) * p
            return p, np.vstack([px, py]), pxx + pyy

        for g, d in gb:
            right = g.face_centers[0] > xmax - tol
            left = g.face_centers[0] < tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(g, dir_bc, "dir")
            bc_val = np.zeros(g.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(g.face_centers[:, dir_bc])

            aperture = 1e-6 ** (2 - g.dim) * np.ones(g.num_cells)

            pa, _, lpc = analytic_p(g.cell_centers)
            src = -lpc * g.cell_volumes * aperture
            specified_parameters = {"bc": bound, "bc_values": bc_val, "source": src}
            if g.dim == 1:
                specified_parameters["second_order_tensor"] = pp.SecondOrderTensor(
                    1e-10 * np.ones(g.num_cells)
                )
            pp.initialize_default_data(g, d, "flow", specified_parameters)

        for _, d in gb.edges():
            pp.params.data.add_discretization_matrix_keyword(d, "flow")
            d[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries={"normal_diffusivity": 1e10}
            )

        self.solve(gb, analytic_p)

    def solve(self, gb, analytic_p):

        # Parameter-discretization keyword:
        kw = "flow"
        # Terms
        key_flux = "flux"
        key_src = "src"
        # Primary variables
        key_p = "pressure"  # pressure name
        key_m = "mortar"  # mortar name

        tpfa = pp.Tpfa(kw)
        src = pp.ScalarSource(kw)
        for g, d in gb:
            d[pp.DISCRETIZATION] = {key_p: {key_src: src, key_flux: tpfa}}
            d[pp.PRIMARY_VARIABLES] = {key_p: {"cells": 1}}

        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {key_m: {"cells": 1}}
            if g1.dim == g2.dim:
                mortar_disc = pp.FluxPressureContinuity(kw, tpfa)
            else:
                mortar_disc = pp.RobinCoupling(kw, tpfa)
            d[pp.COUPLING_DISCRETIZATION] = {
                key_flux: {
                    g1: (key_p, key_flux),
                    g2: (key_p, key_flux),
                    e: (key_m, mortar_disc),
                }
            }

        assembler = pp.Assembler(gb)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)

        # test pressure
        for g, d in gb:
            ap, _, _ = analytic_p(g.cell_centers)
            self.assertTrue(np.max(np.abs(d[pp.STATE][key_p] - ap)) < 5e-2)

        # test mortar solution
        for e, d_e in gb.edges():
            mg = d_e["mortar_grid"]
            g1, g2 = gb.nodes_of_edge(e)
            if g1 == g2:
                left_to_m = mg.primary_to_mortar_avg()
                right_to_m = mg.secondary_to_mortar_avg()
            else:
                continue
            d1 = gb.node_props(g1)
            d2 = gb.node_props(g2)

            _, analytic_flux, _ = analytic_p(g1.face_centers)
            # the aperture is assumed constant
            left_flux = np.sum(analytic_flux * g1.face_normals[:2], 0)
            left_flux = left_to_m * (
                d1[pp.DISCRETIZATION_MATRICES][kw]["bound_flux"] * left_flux
            )
            # right flux is negative lambda
            right_flux = np.sum(analytic_flux * g2.face_normals[:2], 0)
            right_flux = -right_to_m * (
                d2[pp.DISCRETIZATION_MATRICES][kw]["bound_flux"] * right_flux
            )
            self.assertTrue(np.max(np.abs(d_e[pp.STATE][key_m] - left_flux)) < 5e-2)
            self.assertTrue(np.max(np.abs(d_e[pp.STATE][key_m] - right_flux)) < 5e-2)

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

        d_e = gb.edge_props((g1, g1))
        d_e["mortar_grid"] = pp.MortarGrid(g1.dim - 1, {"0": mg}, face_faces)
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

                    d_e["mortar_grid"] = pp.MortarGrid(
                        gi.dim - 1, {"0": g_m}, face_faces
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

        self.assertTrue(left.size == np.unique(left).size)
        self.assertTrue(right.size == np.unique(right).size)

        val = np.ones(left.shape, dtype=bool)
        shape = [gj.num_faces, gi.num_faces]

        face_faces = sps.coo_matrix((val, (right, left)), shape=shape)
        return face_faces


if __name__ == "__main__":
    unittest.main()
