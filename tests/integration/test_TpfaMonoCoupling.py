import unittest
from tests import test_utils

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
        mdg = self.generate_grids(n, xmax, ymax, split)
        tol = 1e-6
        for sd, data in mdg.subdomains(return_data=True):
            left = sd.face_centers[0] < tol
            right = sd.face_centers[0] > xmax - tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(sd, dir_bc, "dir")
            bc_val = np.zeros(sd.num_faces)
            bc_val[left] = xmax
            bc_val[right] = 0
            specified_parameters = {"bc": bound, "bc_values": bc_val}
            pp.initialize_default_data(sd, data, "flow", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            data[pp.PARAMETERS] = pp.Parameters(intf, ["flow"], [{}])
            pp.params.data.add_discretization_matrix_keyword(data, "flow")

        # assign discretization
        data_key = "flow"
        tpfa = pp.Tpfa(data_key)
        coupler = pp.FluxPressureContinuity(data_key, tpfa)
        assembler = test_utils.setup_flow_assembler(
            mdg, tpfa, data_key, coupler=coupler
        )
        test_utils.solve_and_distribute_pressure(mdg, assembler)

        # test pressure
        for sd, data in mdg.subdomains(return_data=True):
            self.assertTrue(
                np.allclose(data[pp.STATE]["pressure"], xmax - sd.cell_centers[0])
            )

        # test mortar solution
        for intf, data in mdg.interfaces(return_data=True):
            g_primary, g_secondary = mdg.interface_to_subdomain_pair(intf)
            primary_to_m = intf.primary_to_mortar_avg()
            secondary_to_m = intf.secondary_to_mortar_avg()

            primary_area = primary_to_m * g_primary.face_areas
            secondary_area = secondary_to_m * g_secondary.face_areas

            self.assertTrue(
                np.allclose(data[pp.STATE]["mortar_flux"] / primary_area, 1)
            )
            self.assertTrue(
                np.allclose(data[pp.STATE]["mortar_flux"] / secondary_area, 1)
            )

    def generate_grids(self, n, xmax, ymax, split):
        g1 = pp.CartGrid([split * n, ymax * n], physdims=[split, ymax])
        g2 = pp.CartGrid([(xmax - split) * n, ymax * n], physdims=[xmax - split, ymax])
        g2.nodes[0] += split

        g1.compute_geometry()
        g2.compute_geometry()
        grids = [g1, g2]

        mdg = pp.MixedDimensionalGrid()

        mdg.add_subdomains(grids)

        tol = 1e-6
        right_grid = g2
        left_grid = g1

        left_faces = np.argwhere(left_grid.face_centers[0] > split - tol).ravel()
        right_faces = np.argwhere(right_grid.face_centers[0] < split + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [right_grid.num_faces, left_grid.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        side_grid = pp.TensorGrid(np.array([split] * ((n + 1) * ymax)))
        side_grid.nodes[1] = np.linspace(0, ymax, (n + 1) * ymax)
        side_grid.compute_geometry()

        intf = pp.MortarGrid(g1.dim - 1, {"0": side_grid}, face_faces)

        mdg.add_interface(intf, (right_grid, left_grid), face_faces)

        data = mdg.interface_data(intf)
        data["edge_number"] = 0

        return mdg


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
        mdg = self.generate_2d_grid(n, xmax, ymax)
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

        for sd, data in mdg.subdomains(return_data=True):
            left = sd.face_centers[0] < tol
            right = sd.face_centers[0] > xmax - tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(sd, dir_bc, "dir")
            bc_val = np.zeros(sd.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(sd.face_centers[:, dir_bc])

            pa, _, lpc = analytic_p(sd.cell_centers)
            src = -lpc * sd.cell_volumes
            specified_parameters = {"bc": bound, "bc_values": bc_val, "source": src}
            pp.initialize_default_data(sd, data, "flow", specified_parameters)

        for _, data in mdg.interfaces(return_data=True):
            pp.params.data.add_discretization_matrix_keyword(data, "flow")
        self.solve(mdg, analytic_p)

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
        mdg = self.generate_grid_with_fractures(n, xmax, ymax)
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

        for sd, data in mdg.subdomains(return_data=True):
            right = sd.face_centers[0] > xmax - tol
            left = sd.face_centers[0] < tol
            dir_bc = left + right
            bound = pp.BoundaryCondition(sd, dir_bc, "dir")
            bc_val = np.zeros(sd.num_faces)
            bc_val[dir_bc], _, _ = analytic_p(sd.face_centers[:, dir_bc])

            aperture = 1e-6 ** (2 - sd.dim) * np.ones(sd.num_cells)

            pa, _, lpc = analytic_p(sd.cell_centers)
            src = -lpc * sd.cell_volumes * aperture
            specified_parameters = {"bc": bound, "bc_values": bc_val, "source": src}
            if sd.dim == 1:
                specified_parameters["second_order_tensor"] = pp.SecondOrderTensor(
                    1e-10 * np.ones(sd.num_cells)
                )
            pp.initialize_default_data(sd, data, "flow", specified_parameters)

        for _, data in mdg.interfaces(return_data=True):
            pp.params.data.add_discretization_matrix_keyword(data, "flow")
            data[pp.PARAMETERS] = pp.Parameters(
                keywords=["flow"], dictionaries={"normal_diffusivity": 1e10}
            )

        self.solve(mdg, analytic_p)

    def solve(self, mdg, analytic_p):

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
        for _, data in mdg.subdomains(return_data=True):
            data[pp.DISCRETIZATION] = {key_p: {key_src: src, key_flux: tpfa}}
            data[pp.PRIMARY_VARIABLES] = {key_p: {"cells": 1}}

        for intf, data in mdg.interfaces(return_data=True):
            g2, g1 = mdg.interface_to_subdomain_pair(intf)
            data[pp.PRIMARY_VARIABLES] = {key_m: {"cells": 1}}
            if g1.dim == g2.dim:
                mortar_disc = pp.FluxPressureContinuity(kw, tpfa)
            else:
                mortar_disc = pp.RobinCoupling(kw, tpfa)
            data[pp.COUPLING_DISCRETIZATION] = {
                key_flux: {
                    g1: (key_p, key_flux),
                    g2: (key_p, key_flux),
                    intf: (key_m, mortar_disc),
                }
            }

        assembler = pp.Assembler(mdg)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)

        # test pressure
        for sd, data in mdg.subdomains(return_data=True):
            ap, _, _ = analytic_p(sd.cell_centers)
            self.assertTrue(np.max(np.abs(data[pp.STATE][key_p] - ap)) < 5e-2)

        # test mortar solution
        for intf, data in mdg.interfaces(return_data=True):
            g2, g1 = mdg.interface_to_subdomain_pair(intf)
            if g1 == g2:
                left_to_m = intf.primary_to_mortar_avg()
                right_to_m = intf.secondary_to_mortar_avg()
            else:
                continue
            d1 = mdg.subdomain_data(g1)
            d2 = mdg.subdomain_data(g2)

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
            self.assertTrue(np.max(np.abs(data[pp.STATE][key_m] - left_flux)) < 5e-2)
            self.assertTrue(np.max(np.abs(data[pp.STATE][key_m] - right_flux)) < 5e-2)

    def generate_2d_grid(self, n, xmax, ymax):
        g1 = pp.CartGrid([xmax * n, ymax * n], physdims=[xmax, ymax])
        g1.compute_geometry()
        mdg = pp.MixedDimensionalGrid()

        mdg.add_subdomains(g1)
        tol = 1e-6
        left_faces = np.argwhere(g1.face_centers[1] > ymax - tol).ravel()
        right_faces = np.argwhere(g1.face_centers[1] < 0 + tol).ravel()
        val = np.ones(left_faces.size, dtype=np.bool)
        shape = [g1.num_faces, g1.num_faces]

        face_faces = sps.coo_matrix((val, (right_faces, left_faces)), shape=shape)

        side_grid = pp.TensorGrid(np.linspace(0, xmax, n + 1))
        side_grid.nodes[1] = ymax
        side_grid.compute_geometry()

        intf = pp.MortarGrid(g1.dim - 1, {"0": side_grid}, face_faces)
        intf.compute_geometry()

        mdg.add_interface(intf, (g1, g1), face_faces)
        return mdg

    def generate_grid_with_fractures(self, n, xmax, ymax):
        x = [xmax / 3] * 2
        y = [0, ymax / 2]
        fracs = [np.array([x, y])]
        y = [ymax / 2, ymax]
        fracs.append(np.array([x, y]))

        mdg = pp.meshing.cart_grid(fracs, [xmax * n, ymax * n], physdims=[xmax, ymax])

        tol = 1e-6
        for sd, data in mdg.subdomains(return_data=True):
            # map right faces to left
            left = np.argwhere(sd.face_centers[1] < tol).ravel()
            right = np.argwhere(sd.face_centers[1] > ymax - tol).ravel()
            sd.face_centers[1, right] = 0

            sd.left = left
            sd.right = right

        # now create mappings between two grids of equal dimension
        for dim in [2, 1]:
            grids = mdg.subdomains(dim=dim)
            for i, gi in enumerate(grids):
                if gi.left.size < 1:
                    continue
                for j, gj in enumerate(grids):
                    if gj.right.size < 1:
                        continue
                    face_faces = self.match_grids(gi, gj)

                    if np.sum(face_faces) == 0:
                        continue

                    di = mdg.subdomain_data(gi)
                    dj = mdg.subdomain_data(gj)

                    # if mdg.node_id(gi) < dj["node_number"]:
                    if mdg.node_id(gi) < mdg.node_id(gj):
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

                    intf = pp.MortarGrid(gi.dim - 1, {"0": g_m}, face_faces)
                    mdg.add_interface(intf, (gi, gj), face_faces)

        mdg.compute_geometry()  # basically reset sd.face_centers[,right]

        return mdg

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
