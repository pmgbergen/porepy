""" Tests for md flow with gravity. Only finite volumes for now.
    
Much of the code is copied and slightly modified from unit/test_darcy_mortar.py
    
Logic in all tests:
    1) Set up a grid, with the methods set_grids and (submethod) simplex_mdg
    2) Set parameters.
    3) Run test cases.
    4) Check solution for pressure and mortar flux.
    
Rudimentary testing in 3d. The tests are direct analogues of the 2d ones.
Could be expanded.
"""
import unittest
from tests import test_utils
from typing import List

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestMixedDimGravity(unittest.TestCase):
    def mortar_nodes(self) -> List:
        return [3, 4]

    def fracture_nodes(self) -> List:
        return [2, 3, 4]

    def discretizations(self) -> List:
        return ["mpfa", "tpfa"]

    def set_parameters(
        self,
        neu_val_top=None,
        dir_val_top=None,
        kn: float = 1e0,
        method="mpfa",
        aperture: float = 1e-1,
        gravity_angle: float = 0,
    ) -> None:
        """
        Parameters:
            neu_val_top (float): Default None implies Dirichlet on top. If not None,
                the prescribed value will be applied as the Neumann bc value.
            dir_val_top (float): If not None, the prescribed value will be
                applied as the Dirichlet bc value. Note that neu_val_top takes precedent
                over dir_val_top.
            method: Discretization method.
            kn: Normal permeability of the fracture. Will be multiplied by aperture/2
                to yield the normal diffusivity.
            aperture: Fracture aperture.
            gravity_angle: Angle by which to rotate the applied vector source field.
        """
        # Set up flow field with uniform flow in y-direction
        mdg = self.mdg
        for sd, data in mdg.subdomains(return_data=True):
            a = np.power(aperture, mdg.dim_max() - sd.dim)
            perm = pp.SecondOrderTensor(kxx=a * np.ones(sd.num_cells))
            gravity = np.zeros((mdg.dim_max(), sd.num_cells))
            # Angle of zero means force vector of [0, -1]
            gravity[1, :] = -np.cos(gravity_angle)
            gravity[0, :] = np.sin(gravity_angle)

            b_val = np.zeros(sd.num_faces)
            if sd.dim == self.mdg.dim_max():
                if neu_val_top is not None:
                    dir_faces = np.atleast_1d(pp.face_on_side(sd, ["ymin"])[0])
                    neu_faces = np.atleast_1d(pp.face_on_side(sd, ["ymax"])[0])
                    b_val[neu_faces] = neu_val_top
                else:
                    dir_faces = pp.face_on_side(sd, ["ymin", "ymax"])
                    b_val[dir_faces[0]] = 0
                    if dir_val_top is not None:
                        b_val[dir_faces[1]] = dir_val_top
                    dir_faces = np.hstack((dir_faces[0], dir_faces[1]))
                labels = np.array(["dir"] * dir_faces.size)
                bc = pp.BoundaryCondition(sd, dir_faces, labels)

            else:
                bc = pp.BoundaryCondition(sd)
            parameter_dictionary = {
                "bc_values": b_val,
                "bc": bc,
                "ambient_dimension": mdg.dim_max(),
                "mpfa_inverter": "python",
                "second_order_tensor": perm,
                "vector_source": gravity.ravel("F"),
            }
            pp.initialize_data(sd, data, "flow", parameter_dictionary)

        for intf, data in mdg.interfaces(return_data=True):
            a = aperture * np.ones(intf.num_cells)
            gravity = np.zeros((mdg.dim_max(), intf.num_cells))
            # Angle of zero means force vector of [0, -1]
            gravity[1, :] = -np.cos(gravity_angle)
            gravity[0, :] = np.sin(gravity_angle)
            gravity *= a / 2
            parameter_dictionary = {
                "normal_diffusivity": 2 / a * kn,
                "ambient_dimension": mdg.dim_max(),
                "vector_source": gravity.ravel("F"),
            }
            pp.initialize_data(intf, data, "flow", parameter_dictionary)

    def grid_2d(self, pert_node: bool = False) -> pp.Grid:
        """
        Make a 2d unit square simplex grid with six cells:
        __________
        |\      /|
        | \    / |
        |  \  /  |
        |   \/   |
        |--------|
        |   /\   |
        |  /  \  |
        | /    \ |
        |/      \|
        ----------
        pert_node perturbs one node in the grid. Leads to non-matching cells.
        Tests that there is no assumptions on direction of fluxes in the
        mortar coupling.
        """
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0.5, 0],
                [0.5, 0.5, 0],
                [0, 0.5, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
                [1, 0.5, 0],
                [1, 1, 0],
                [0, 1, 0],
            ]
        ).T
        if pert_node:
            nodes[0, 3] = 0.75

        fn = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 3],
                [3, 1],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 5],
                [9, 6],
                [6, 8],
            ]
        ).T
        cf = np.array(
            [[3, 4, 5], [0, 6, 5], [1, 2, 6], [7, 12, 11], [12, 13, 10], [8, 9, 13]]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        data = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1])

        cell_faces = sps.csc_matrix((data, (cf.ravel("F"), cols)))

        g = pp.Grid(2, nodes, face_nodes, cell_faces, "TriangleGrid")
        g.compute_geometry()
        g.tags["fracture_faces"][[2, 3, 7, 8]] = 1

        g.global_point_ind = np.arange(nodes.shape[1])

        return g

    def grid_1d(self, num_pts=3) -> pp.Grid:
        g = pp.TensorGrid(np.arange(num_pts))
        g.nodes = np.vstack(
            (np.linspace(0, 1, num_pts), 0.5 * np.ones(num_pts), np.zeros(num_pts))
        )
        g.compute_geometry()
        g.global_point_ind = np.arange(g.num_nodes)
        return g

    def simplex_mdg(self, num_1d=3, num_nodes_mortar=3, pert_node=False) -> None:
        """Compute a unit square mdg with one throughgoing horizontal fracture and
        a simplex grid consisting of six cells in 2d.
        __________
        |\      /|
        | \    / |
        |  \  /  |
        |   \/   |
        |--------|
        |   /\   |
        |  /  \  |
        | /    \ |
        |/      \|
        ----------

        Options include modifying the number of 1d and mortar cells, perturbing nodes
        and flip face normals.
        """
        sd_2d = self.grid_2d()
        sd_1d = self.grid_1d()

        mdg, _ = pp.meshing._assemble_mdg([[sd_2d], [sd_1d]])

        a = np.zeros((sd_2d.num_faces, sd_1d.num_cells))
        a[2, 1] = 1
        a[3, 0] = 1
        a[7, 0] = 1
        a[8, 1] = 1
        face_cells = sps.csc_matrix(a.T)
        pp.meshing.create_interfaces(mdg, {(sd_2d, sd_1d): face_cells})

        sd_2d_new = self.grid_2d(pert_node=pert_node)
        sd_1d_new = self.grid_1d(num_1d)
        mdg.replace_subdomains_and_interfaces(
            sd_map={sd_2d: sd_2d_new, sd_1d: sd_1d_new}
        )
        intf = mdg.interfaces()[0]

        new_side_grids = {
            s: pp.refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
            for s, g in intf.side_grids.items()
        }
        intf.update_mortar(new_side_grids, tol=1e-4)

        mdg.assign_subdomain_ordering()
        self.mdg = mdg

    def set_grids_2d(
        self,
        N,
        num_nodes_mortar=None,
        num_nodes_1d=None,
        physdims=[1, 1],
        simplex=False,
    ):
        if simplex:
            self.simplex_mdg(num_1d=num_nodes_1d, num_nodes_mortar=num_nodes_mortar)
            return
        f1 = np.array([[0, physdims[0]], [0.5, 0.5]])

        mdg = pp.meshing.cart_grid([f1], N, **{"physdims": physdims})
        mdg.compute_geometry()
        mdg.assign_subdomain_ordering()
        if num_nodes_mortar is None:
            self.mdg = mdg
            return

        for intf in mdg.interfaces():

            new_side_grids = {
                s: pp.refinement.remesh_1d(g, num_nodes=num_nodes_mortar)
                for s, g in intf.side_grids.items()
            }

            intf.update_mortar(new_side_grids, tol=1e-4)
            # refine the 1d-physical grid
            _, old_g = mdg.interface_to_subdomain_pair(intf)
            new_g = pp.refinement.remesh_1d(old_g, num_nodes=num_nodes_1d)
            new_g.compute_geometry()

            mdg.replace_subdomains_and_interfaces({old_g: new_g})
            intf.update_secondary(new_g, tol=1e-4)
        self.mdg = mdg

    def solve(self, method):
        key = "flow"
        mdg = self.mdg
        if method == "tpfa":
            discretization = pp.Tpfa(key)
        elif method == "mpfa":
            discretization = pp.Mpfa(key)
        elif method == "mvem":
            discretization = pp.MVEM(key)
        assembler = test_utils.setup_flow_assembler(mdg, discretization, key)
        assembler.discretize()
        A_flow, b_flow = assembler.assemble_matrix_rhs()
        p = sps.linalg.spsolve(A_flow, b_flow)
        assembler.distribute_variable(p)
        return p

    def verify_pressure(self, p_known: float = 0):
        """Verify that the pressure of all subdomains equals p_known."""
        for _, data in self.mdg.subdomains(return_data=True):
            p = data[pp.STATE]["pressure"]
            self.assertTrue(np.allclose(p, p_known, rtol=1e-3, atol=1e-3))

    def verify_mortar_flux(self, u_known: float):
        """Verify that the mortar flux of all interfaces equals u_known."""
        for _, data in self.mdg.interfaces(return_data=True):
            u = np.abs(data[pp.STATE]["mortar_flux"])
            self.assertTrue(np.allclose(u, u_known, rtol=1e-3, atol=1e-3))

    def verify_hydrostatic(self, angle=0, a=1e-1):
        """Check that the pressure profile is hydrostatic, with the adjustment
        for the fracture.
        Without the fracture, the profile is expected to be linear within each
        subdomain, with a small additional jump of aperture at the fracture.
        The full range is
        0 (bottom) to -1- aperture (top).
        """
        mdg = self.mdg
        sd_primary = mdg.subdomains(dim=mdg.dim_max())[0]
        p_primary = mdg.subdomain_data(sd_primary)[pp.STATE]["pressure"]
        # The cells above the fracture
        h = sd_primary.cell_centers[1]
        ind = h > 0.5
        p_known = -(a * ind + h) * np.cos(angle)
        self.assertTrue(np.allclose(p_primary, p_known, rtol=1e-3, atol=1e-3))
        sd_secondary = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        p_secondary = mdg.subdomain_data(sd_secondary)[pp.STATE]["pressure"]
        # Half the additional jump is added to the fracture pressure
        h = sd_secondary.cell_centers[1]
        p_known = -(a / 2 + h) * np.cos(angle)

        self.assertTrue(np.allclose(p_secondary, p_known, rtol=1e-3, atol=1e-3))
        for _, data in self.mdg.interfaces(return_data=True):
            lmbda = data[pp.STATE]["mortar_flux"]
            self.assertTrue(np.allclose(lmbda, 0, rtol=1e-3, atol=1e-3))

    def test_no_flow_neumann(self):
        """Use homogeneous Neumann boundary conditions on top Dirichlet
        on bottom.

        The pressure distribution should be hydrostatic.

        """
        nx = 3
        for method in self.discretizations():
            for num_nodes_mortar in self.mortar_nodes():
                for num_nodes_1d in self.fracture_nodes():
                    for simplex in [False, True]:
                        if simplex and (
                            num_nodes_mortar != num_nodes_1d or method == "tpfa"
                        ):
                            # Different number of mortar and 1d cells not implemented for simplex
                            # Also, tpfa cannot be expected to produce pressure profiles
                            # with sufficient accuracy for the test to make sense.
                            continue
                        self.set_grids(
                            2,
                            N=[nx, 2],
                            num_nodes_mortar=num_nodes_mortar,
                            num_nodes_1d=num_nodes_1d,
                            simplex=simplex,
                        )
                        self.set_parameters(neu_val_top=0)
                        self.solve(method)

                        self.verify_hydrostatic()
                        self.verify_mortar_flux(0)

    def test_no_flow_rotate_gravity(self):
        """
        Rotate the angle of gravity. Neumann boundaries except Dirichlet on the bottom.

        There should be no flow.

        """

        # The angle pi/2 requires nx = 1 for there not to be flow
        nx = 1
        for method in self.discretizations():
            for num_nodes_mortar in self.mortar_nodes():
                # For the below tests of uniform pressure and no uniform flow to make
                # sense, there can only be a single cell in the fracture grid.
                for num_nodes_1d in [2]:
                    for simplex in [False, True]:
                        if (
                            simplex
                            and num_nodes_mortar != num_nodes_1d
                            or method == "tpfa"
                        ):
                            continue
                        self.set_grids(
                            2,
                            N=[nx, 2],
                            num_nodes_mortar=num_nodes_mortar,
                            num_nodes_1d=num_nodes_1d,
                            simplex=simplex,
                        )
                        for angle in [0, np.pi / 2, np.pi]:
                            self.set_parameters(neu_val_top=0, gravity_angle=angle)
                            x = self.solve(method)
                            if np.isclose(angle, np.pi / 2):
                                if not simplex:
                                    self.verify_pressure()
                            else:
                                self.verify_hydrostatic(angle)
                            self.verify_mortar_flux(0)

    def test_no_flow_dirichlet(self):
        """
        Dirichlet boundary conditions, but set so that the pressure is hydrostatic,
        and there is no flow.

        """

        nx = 3
        for method in self.discretizations():
            for num_nodes_mortar in self.mortar_nodes():
                for num_nodes_1d in self.fracture_nodes():
                    for simplex in [False, True]:
                        if simplex and (
                            num_nodes_mortar != num_nodes_1d or method == "tpfa"
                        ):
                            # Different number of mortar and 1d cells not implemented for simplex
                            continue
                        self.set_grids(
                            2,
                            N=[nx, 2],
                            num_nodes_mortar=num_nodes_mortar,
                            num_nodes_1d=num_nodes_1d,
                            simplex=simplex,
                        )
                        self.set_parameters(dir_val_top=-1.1)
                        x = self.solve(method)
                        self.verify_hydrostatic()
                        self.verify_mortar_flux(0)

    def test_inflow_top(self):
        """
        Prescribed inflow at the top. Strength of the flow counteracts gravity, so that
        the pressure is uniform.
        """
        nx = 2
        a = 1e-2
        for method in self.discretizations():
            for num_nodes_mortar in self.mortar_nodes():
                for num_nodes_1d in self.fracture_nodes():
                    for simplex in [True, False]:
                        if simplex and (
                            num_nodes_mortar != num_nodes_1d or method == "tpfa"
                        ):
                            # Different number of mortar and 1d cells not implemented for simplex
                            continue
                        self.set_grids(
                            2,
                            N=[nx, 2],
                            num_nodes_mortar=num_nodes_mortar,
                            num_nodes_1d=num_nodes_1d,
                            simplex=simplex,
                        )
                        # Flux density is 1/2.
                        val = -1 if simplex else -1 / nx
                        self.set_parameters(neu_val_top=val, aperture=a)
                        x = self.solve(method=method)
                        self.verify_pressure()
                        self.verify_mortar_flux(1 / (num_nodes_mortar - 1))

    def test_uniform_pressure(self):
        """
        Prescribed pressure at the top. Strength of the flow counteracts gravity, so
        that the pressure is uniform.

        The total flow should equal the gravity force (=1).
        """
        a = 3e-3
        for method in self.discretizations():
            for num_nodes_mortar in self.mortar_nodes():
                for num_nodes_1d in self.fracture_nodes():
                    for simplex in [True]:
                        if simplex and (
                            num_nodes_mortar != num_nodes_1d or method == "tpfa"
                        ):
                            # Different number of mortar and 1d cells not implemented for simplex
                            continue
                        self.set_grids(
                            2,
                            N=[1, 2],
                            num_nodes_mortar=num_nodes_mortar,
                            num_nodes_1d=num_nodes_1d,
                            simplex=simplex,
                        )

                        self.set_parameters(dir_val_top=0, aperture=a)
                        x = self.solve(method)
                        self.verify_pressure()

                        self.verify_mortar_flux(1 / (num_nodes_mortar - 1))

    def set_grids(self, dim, **kwargs):
        """Wrapper to facilitate running same tests in 2d and 3d."""
        if dim == 2:
            self.set_grids_2d(**kwargs)
        elif dim == 3:
            self.set_grids_3d(**kwargs)

    # --3d section. See analogous methods/tests above for documentation --#

    def set_grids_3d(self, num_fracs=1):

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        if num_fracs == 0:
            fl = []

        elif num_fracs == 1:
            fl = [
                pp.Fracture3d(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                )
            ]
        elif num_fracs == 2:
            fl = [
                pp.Fracture3d(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.Fracture3d(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
            ]

        elif num_fracs == 3:
            fl = [
                pp.Fracture3d(
                    np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])
                ),
                pp.Fracture3d(
                    np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
                ),
                pp.Fracture3d(
                    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
                ),
            ]

        network = pp.FractureNetwork3d(fl, domain)
        mesh_args = {"mesh_size_frac": 0.5, "mesh_size_min": 0.5}
        mdg = network.mesh(mesh_args)

        self.mdg = mdg

    def test_one_fracture_no_flow_neumann(self):
        for method in self.discretizations():
            self.set_grids(3)
            self.set_parameters(neu_val_top=0)
            x = self.solve(method)
            self.verify_hydrostatic()
            self.verify_mortar_flux(0)

    def test_one_fracture_no_flow_dirichlet(self):
        for method in self.discretizations():
            self.set_grids(3)
            self.set_parameters(dir_val_top=-1.1)
            x = self.solve(method)
            self.verify_hydrostatic()
            self.verify_mortar_flux(0)


if __name__ == "__main__":
    unittest.main()
