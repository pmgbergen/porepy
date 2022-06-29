"""
Module for testing the discrete fracture network (DFN) with continuous pressure and normal flux at the intersection.
"""

import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestDFN(unittest.TestCase):
    def test_mvem_0(self):
        """
        In this test we set up a network with 2 1d fractures that intersect in a point.
        We validate the resulting matrices and right hand side.
        We use the numerical scheme MVEM.
        """
        dfn_dim = 1

        # create the grid bucket
        mdg, _ = pp.grid_buckets_2d.two_intersecting([2, 2], simplex=False)
        create_dfn(mdg, dfn_dim)

        # setup data and assembler
        setup_data(mdg)
        assembler, _ = setup_discr_mvem(mdg)
        dof_manager = assembler._dof_manager
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        A = A.todense()

        A_f1 = np.matrix(
            [
                [0.75, 0.0, 0.0, -0.25, 1, 0],
                [0.0, 0.75, 0.0, 0.0, 0, 0],
                [0.0, -0.25, 0.75, 0.0, 0, -1],
                [0.0, 0.0, 0.0, 0.75, 0, 0],
                [1.0, 0.0, 0.0, -1.0, 0, 0],
                [0.0, 1.0, -1.0, 0.0, 0, 0],
            ]
        )
        b_f1 = np.array([2, 0, 0, 0, 0, 0])

        A_f2 = np.matrix(
            [
                [0.75, 0.0, 0.0, -0.25, 1, 0],
                [0.0, 0.75, 0.0, 0.0, 0, 0],
                [0.0, -0.25, 0.75, 0.0, 0, -1],
                [0.0, 0.0, 0.0, 0.75, 0, 0],
                [1.0, 0.0, 0.0, -1.0, 0, 0],
                [0.0, 1.0, -1.0, 0.0, 0, 0],
            ]
        )
        b_f2 = np.array([1, 0, -1, 0, 0, 0])

        A_0 = np.matrix([[0.0]])
        b_0 = np.array([0])

        global_dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for sd in mdg.subdomains():
            block = dof_manager.block_dof[(sd, "flow")]
            dof = np.arange(global_dof[block], global_dof[block + 1])

            if sd.dim == 1 and np.allclose(sd.nodes[0], 1):  # f1
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f1))
                self.assertTrue(np.allclose(b[dof], b_f1))
            elif sd.dim == 1 and np.allclose(sd.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f2))
                self.assertTrue(np.allclose(b[dof], b_f2))
            elif sd.dim == 0:  # intersection
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_0))
                self.assertTrue(np.allclose(b[dof], b_0))

        # known matrices associate to the edge where f1 is involved
        A_e1_secondary_e = np.matrix(
            [
                [0.0, -0.25],
                [0.0, 0.0],
                [0.25, 0.0],
                [0.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )

        A_e1_e_secondary = np.matrix(
            [[0.0, 0.75, -0.25, 0.0, 0, 1], [0.25, 0.0, 0.0, -0.75, 1, 0]]
        )

        A_e1_secondary_primary = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        A_e1_primary_secondary = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        A_e1_e_e = np.matrix([[-0.75, 0.0], [0.0, -0.75]])

        A_e1_e_primary = np.matrix([[-1.0], [-1.0]])
        A_e1_primary_e = np.matrix([[-1.0, -1.0]])

        b_e1 = np.array([0.0, 0.0])

        A_e2_secondary_e = np.matrix(
            [
                [0.0, -0.25],
                [0.0, 0.0],
                [0.25, 0.0],
                [0.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )

        A_e2_e_secondary = np.matrix(
            [[0.0, 0.75, -0.25, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, -0.75, 1.0, 0.0]]
        )

        A_e2_secondary_primary = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        A_e2_primary_secondary = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        A_e2_e_e = np.matrix([[-0.75, 0.0], [0.0, -0.75]])

        A_e2_e_primary = np.matrix([[-1.0], [-1.0]])
        A_e2_primary_e = np.matrix([[-1.0, -1.0]])

        b_e2 = np.array([0.0, 0.0])

        for intf, data in mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

            block_e = dof_manager.block_dof[(intf, "flow")]
            block_primary = dof_manager.block_dof[(sd_secondary, "flow")]
            block_secondary = dof_manager.block_dof[(sd_primary, "flow")]

            dof_e = np.arange(global_dof[block_e], global_dof[block_e + 1])
            dof_primary = np.arange(
                global_dof[block_primary], global_dof[block_primary + 1]
            )
            dof_secondary = np.arange(
                global_dof[block_secondary], global_dof[block_secondary + 1]
            )

            if np.allclose(sd_primary.nodes[0], 1):  # f1
                self.assertTrue(
                    np.allclose(A[dof_secondary, :][:, dof_e], A_e1_secondary_e)
                )
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_secondary], A_e1_e_secondary)
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_secondary, :][:, dof_primary], A_e1_secondary_primary
                    )
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_primary, :][:, dof_secondary], A_e1_primary_secondary
                    )
                )
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e1_e_e))
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_primary], A_e1_e_primary)
                )
                self.assertTrue(
                    np.allclose(A[dof_primary, :][:, dof_e], A_e1_primary_e)
                )
                self.assertTrue(np.allclose(b[dof_e], b_e1))

            elif np.allclose(sd_primary.nodes[1], 1):  # f2
                self.assertTrue(
                    np.allclose(A[dof_secondary, :][:, dof_e], A_e2_secondary_e)
                )
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_secondary], A_e2_e_secondary)
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_secondary, :][:, dof_primary], A_e2_secondary_primary
                    )
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_primary, :][:, dof_secondary], A_e2_primary_secondary
                    )
                )
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e2_e_e))
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_primary], A_e2_e_primary)
                )
                self.assertTrue(
                    np.allclose(A[dof_primary, :][:, dof_e], A_e2_primary_e)
                )
                self.assertTrue(np.allclose(b[dof_e], b_e2))

    def test_tpfa_0(self):
        """
        In this test we set up a network with 2 1d fractures that intersect in a point.
        We validate the resulting matrices and right hand side.
        We use the numerical scheme Tpfa.
        """
        dfn_dim = 1
        mdg, _ = pp.grid_buckets_2d.two_intersecting([2, 2], simplex=False)
        create_dfn(mdg, dfn_dim)

        # setup data and assembler
        setup_data(mdg)
        assembler, _ = setup_discr_tpfa(mdg)
        dof_manager = assembler._dof_manager

        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        A = A.todense()

        A_f1 = np.matrix([[2, 0], [0, 2]])
        b_f1 = np.array([4, 0])

        A_f2 = np.matrix([[2, 0], [0, 2]])
        b_f2 = np.array([2, 2])

        A_0 = np.matrix([[0.0]])
        b_0 = np.array([0])

        global_dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for sd in mdg.subdomains():
            block = dof_manager.block_dof[(sd, "flow")]
            dof = np.arange(global_dof[block], global_dof[block + 1])

            if sd.dim == 1 and np.allclose(sd.nodes[0], 1):  # f1

                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f1))
                self.assertTrue(np.allclose(b[dof], b_f1))
            elif sd.dim == 1 and np.allclose(sd.nodes[1], 1):  # f2
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_f2))
                self.assertTrue(np.allclose(b[dof], b_f2))
            elif sd.dim == 0:  # intersection
                self.assertTrue(np.allclose(A[dof, :][:, dof], A_0))
                self.assertTrue(np.allclose(b[dof], b_0))

        # known matrices associate to the edge where f1 is involved
        A_e1_secondary_e = np.matrix([[0, 1], [1, 0]])

        A_e1_e_secondary = np.matrix([[0, 1], [1, 0]])

        A_e1_secondary_primary = np.matrix([[0.0], [0.0]])
        A_e1_primary_secondary = np.matrix([[0.0, 0.0]])

        A_e1_e_e = np.matrix([[-0.5, 0.0], [0.0, -0.5]])

        A_e1_e_primary = np.matrix([[-1.0], [-1.0]])
        A_e1_primary_e = np.matrix([[-1.0, -1.0]])

        b_e1 = np.array([0.0, 0.0])

        A_e2_secondary_e = np.matrix([[0, 1], [1, 0]])

        A_e2_e_secondary = np.matrix([[0, 1], [1, 0]])

        A_e2_secondary_primary = np.matrix([[0.0], [0.0]])
        A_e2_primary_secondary = np.matrix([[0.0, 0.0]])

        A_e2_e_e = np.matrix([[-0.5, 0.0], [0.0, -0.5]])

        A_e2_e_primary = np.matrix([[-1.0], [-1.0]])
        A_e2_primary_e = np.matrix([[-1.0, -1.0]])

        b_e2 = np.array([0.0, 0.0])

        for intf in mdg.interfaces():
            g_primary, g_secondary = mdg.interface_to_subdomain_pair(intf)

            block_e = dof_manager.block_dof[(intf, "flow")]
            block_primary = dof_manager.block_dof[(g_secondary, "flow")]
            block_secondary = dof_manager.block_dof[(g_primary, "flow")]

            dof_e = np.arange(global_dof[block_e], global_dof[block_e + 1])
            dof_primary = np.arange(
                global_dof[block_primary], global_dof[block_primary + 1]
            )
            dof_secondary = np.arange(
                global_dof[block_secondary], global_dof[block_secondary + 1]
            )

            if np.allclose(g_primary.nodes[0], 1):  # f1
                self.assertTrue(
                    np.allclose(A[dof_secondary, :][:, dof_e], A_e1_secondary_e)
                )
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_secondary], A_e1_e_secondary)
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_secondary, :][:, dof_primary], A_e1_secondary_primary
                    )
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_primary, :][:, dof_secondary], A_e1_primary_secondary
                    )
                )
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e1_e_e))
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_primary], A_e1_e_primary)
                )
                self.assertTrue(
                    np.allclose(A[dof_primary, :][:, dof_e], A_e1_primary_e)
                )
                self.assertTrue(np.allclose(b[dof_e], b_e1))

            elif np.allclose(g_primary.nodes[1], 1):  # f2
                self.assertTrue(
                    np.allclose(A[dof_secondary, :][:, dof_e], A_e2_secondary_e)
                )
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_secondary], A_e2_e_secondary)
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_secondary, :][:, dof_primary], A_e2_secondary_primary
                    )
                )
                self.assertTrue(
                    np.allclose(
                        A[dof_primary, :][:, dof_secondary], A_e2_primary_secondary
                    )
                )
                self.assertTrue(np.allclose(A[dof_e, :][:, dof_e], A_e2_e_e))
                self.assertTrue(
                    np.allclose(A[dof_e, :][:, dof_primary], A_e2_e_primary)
                )
                self.assertTrue(
                    np.allclose(A[dof_primary, :][:, dof_e], A_e2_primary_e)
                )
                self.assertTrue(np.allclose(b[dof_e], b_e2))

    def test_mvem_1(self):
        """
        In this test we set up a network with 5 1d fractures that intersect in a point.
        We validate the resulting solution.
        We use the numerical scheme MVEM.
        """
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        mdg = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        mdg.compute_geometry()
        create_dfn(mdg, dfn_dim)

        # setup data and assembler
        setup_data(mdg)
        assembler, (discr, _) = setup_discr_mvem(mdg)

        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)
        for sd, data in mdg.subdomains(return_data=True):
            discr = data["discretization"]["flow"]["flux"]
            data["pressure"] = discr.extract_pressure(sd, data[pp.STATE]["flow"], data)

        for sd, data in mdg.subdomains(return_data=True):

            if sd.dim == 1:
                if np.all(sd.cell_centers[1] == 0.5 * N):  # f1
                    known = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
                elif np.all(sd.cell_centers[0] == 0.5 * N):  # f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(sd.cell_centers[0] == 0.625 * N):  # f3
                    known = np.array([4.0, 4.0])
                elif np.all(sd.cell_centers[1] == 0.25 * N):  # f4
                    known = np.array([2.0, 2.0, 2.0, 2.0])
                elif np.all(sd.cell_centers[0] == 0.75 * N):  # f5
                    known = np.array([2.0, 2.0])
                else:
                    raise ValueError

            else:  # sd.dim == 0
                if np.allclose(sd.cell_centers, np.array([[0.5], [0.5], [0.0]]) * N):
                    known = np.array([4.0])
                elif np.allclose(
                    sd.cell_centers, np.array([[0.625], [0.5], [0.0]]) * N
                ):
                    known = np.array([4.0])
                elif np.allclose(sd.cell_centers, np.array([[0.5], [0.25], [0.0]]) * N):
                    known = np.array([2.0])
                elif np.allclose(
                    sd.cell_centers, np.array([[0.75], [0.25], [0.0]]) * N
                ):
                    known = np.array([2.0])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(data["pressure"], known))

    def test_tpfa_1(self):
        """
        In this test we set up a network with 5 1d fractures that intersect in a point.
        We validate the resulting solution.
        We use the numerical scheme Tpfa.
        """
        dfn_dim = 1

        N = 8
        f1 = N * np.array([[0, 1], [0.5, 0.5]])
        f2 = N * np.array([[0.5, 0.5], [0, 1]])
        f3 = N * np.array([[0.625, 0.625], [0.5, 0.75]])
        f4 = N * np.array([[0.25, 0.75], [0.25, 0.25]])
        f5 = N * np.array([[0.75, 0.75], [0.125, 0.375]])

        # create the grid bucket
        mdg = pp.meshing.cart_grid([f1, f2, f3, f4, f5], [N, N])
        mdg.compute_geometry()
        create_dfn(mdg, dfn_dim)

        # setup data and assembler
        setup_data(mdg)
        assembler, _ = setup_discr_tpfa(mdg)
        assembler.discretize()
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        assembler.distribute_variable(x)

        for g, d in mdg.subdomains(return_data=True):

            if g.dim == 1:
                if np.all(g.cell_centers[1] == 0.5 * N):  # f1
                    known = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
                elif np.all(g.cell_centers[0] == 0.5 * N):  # f2
                    known = np.array([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
                elif np.all(g.cell_centers[0] == 0.625 * N):  # f3
                    known = np.array([4, 4])
                elif np.all(g.cell_centers[1] == 0.25 * N):  # f4
                    known = np.array([2, 2, 2, 2])
                elif np.all(g.cell_centers[0] == 0.75 * N):  # f5
                    known = np.array([2, 2])
                else:
                    raise ValueError

            else:  # g.dim == 0
                if np.allclose(g.cell_centers, np.array([[0.5], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.625], [0.5], [0]]) * N):
                    known = np.array([4])
                elif np.allclose(g.cell_centers, np.array([[0.5], [0.25], [0]]) * N):
                    known = np.array([2])
                elif np.allclose(g.cell_centers, np.array([[0.75], [0.25], [0]]) * N):
                    known = np.array([2])
                else:
                    raise ValueError

            self.assertTrue(np.allclose(d[pp.STATE]["flow"], known))


# ------------------------- HELP FUNCTIONS --------------------------------#


def setup_data(mdg, key="flow"):
    """Setup the data"""
    for sd, data in mdg.subdomains(return_data=True):
        param = {}
        kxx = np.ones(sd.num_cells)
        param["second_order_tensor"] = pp.SecondOrderTensor(kxx)

        if sd.dim == mdg.dim_max():
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                sd, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(sd.num_faces)
            bc_val[bound_faces] = sd.face_centers[1, bound_faces]
            param["bc"] = bound
            param["bc_values"] = bc_val
            param["aperture"] = np.ones(sd.num_cells)
            data[pp.PARAMETERS] = pp.Parameters(sd, key, param)
            data[pp.DISCRETIZATION_MATRICES] = {key: {}}

    for _, data in mdg.interfaces(return_data=True):
        data[pp.DISCRETIZATION_MATRICES] = {key: {}}


def setup_discr_mvem(mdg, key="flow"):
    """Setup the discretization MVEM."""
    discr = pp.MVEM(key)
    p_trace = pp.CellDofFaceDofMap(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim == mdg.dim_max():
            data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
            data[pp.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            data[pp.DISCRETIZATION] = {key: {"flux": p_trace}}

    for intf, data in mdg.interfaces(return_data=True):
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
        data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        data[pp.COUPLING_DISCRETIZATION] = {
            "flux": {
                sd_secondary: (key, "flux"),
                sd_primary: (key, "flux"),
                intf: (key, interface),
            }
        }

    return pp.Assembler(mdg), (discr, p_trace)


def setup_discr_tpfa(mdg, key="flow"):
    """Setup the discretization Tpfa."""
    discr = pp.Tpfa(key)
    p_trace = pp.CellDofFaceDofMap(key)
    interface = pp.FluxPressureContinuity(key, discr, p_trace)

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim == mdg.dim_max():
            data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            data[pp.DISCRETIZATION] = {key: {"flux": discr}}
        else:
            data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            data[pp.DISCRETIZATION] = {key: {"flux": p_trace}}

    for intf, data in mdg.interfaces(return_data=True):
        g_primary, g_secondary = mdg.interface_to_subdomain_pair(intf)
        data[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        data[pp.COUPLING_DISCRETIZATION] = {
            "flux": {
                g_secondary: (key, "flux"),
                g_primary: (key, "flux"),
                intf: (key, interface),
            }
        }

    return pp.Assembler(mdg), (discr, p_trace)


def create_dfn(mdg, dim):
    """given a MixedDimensionalGrid remove the higher dimensional node and
    fix the internal mapping."""
    # remove the +1 and -2 dimensional grids with respect to the
    # considered dfn, and re-write the node number
    subdomains = [sd for sd in mdg.subdomains(dim=dim + 1)] + [
        sd for sd in mdg.subdomains(dim=dim - 2)
    ]

    for sd in subdomains:
        node_number = mdg.node_number(sd)
        mdg.remove_subdomain(sd)
        mdg.update_subdomain_ordering(node_number)


if __name__ == "__main__":
    unittest.main()
