from __future__ import division
import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from porepy.numerics.mixed_dim import coupler, condensation
from porepy.numerics.fv.transport import upwind

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):
    """
    Tests for the elimination fluxes.
    """

    # ------------------------------------------------------------------------------#

    def atest_upwind_2d_1d_cross_with_elimination(self):
        """
        Simplest possible elimination scenario, one 0d-grid removed. Check on upwind
        matrix, rhs, solution and time step estimate. Full solution included
        (as comments) for comparison purposes if test breaks.
        """
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [0, 1]])
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        mesh_size = 0.4
        mesh_kwargs = {}
        mesh_kwargs["mesh_size"] = {
            "mode": "constant",
            "value": mesh_size,
            "bound_value": mesh_size,
        }
        gb = pp.meshing.cart_grid([f1, f2], [2, 2], **{"physdims": [1, 1]})
        # gb = pp.meshing.simplex_grid( [f1, f2],domain,**mesh_kwargs)
        gb.compute_geometry()
        gb.assign_node_ordering()

        # Enforce node orderning because of Python 3.5 and 2.7.
        # Don't do it in general.
        cell_centers_1 = np.array(
            [
                [7.50000000e-01, 2.500000000e-01],
                [5.00000000e-01, 5.00000000e-01],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )
        cell_centers_2 = np.array(
            [
                [5.00000000e-01, 5.00000000e-01],
                [7.50000000e-01, 2.500000000e-01],
                [-5.55111512e-17, 5.55111512e-17],
            ]
        )

        for g, d in gb:
            if g.dim == 1:
                if np.allclose(g.cell_centers, cell_centers_1):
                    d["node_number"] = 1
                elif np.allclose(g.cell_centers, cell_centers_2):
                    d["node_number"] = 2
                else:
                    raise ValueError("Grid not found")

        tol = 1e-3
        solver = pp.TpfaMixedDim()
        gb.add_node_props(["param"])
        a = 1e-2
        for g, d in gb:
            param = pp.Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            kxx = np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())
            p = pp.SecondOrderTensor(3, kxx, kyy=kxx, kzz=kxx)
            param.set_tensor("flow", p)

            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                right = bound_face_centers[0, :] > 1 - tol
                left = bound_face_centers[0, :] < tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[right] = ["dir"]

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[right]
                bc_neu = bound_faces[left]
                bc_val[bc_dir] = g.face_centers[0, bc_dir]
                bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

                param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", bc_val)
                # Transport
                bottom = bound_face_centers[1, :] < tol
                top = bound_face_centers[1, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[
                    np.logical_or(
                        np.logical_or(left, right), np.logical_or(top, bottom)
                    )
                ] = ["dir"]

                bc_val = np.zeros(g.num_faces)

                param.set_bc("transport", pp.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("transport", bc_val)
            else:
                param.set_bc(
                    "transport", pp.BoundaryCondition(g, np.empty(0), np.empty(0))
                )
                param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))
            # Transport:
            source = g.cell_volumes * a_dim
            param.set_source("transport", source)

            d["param"] = param

        gb.add_edge_props("param")
        for e, d in gb.edges():
            g_h = gb.nodes_of_edge(e)[1]
            d["param"] = pp.Parameters(g_h)

        A, rhs = solver.matrix_rhs(gb)
        # p = sps.linalg.spsolve(A,rhs)
        _, p_red, _, _ = condensation.solve_static_condensation(A, rhs, gb, dim=0)
        dim_to_remove = 0
        gb_r, elimination_data = gb.duplicate_without_dimension(dim_to_remove)
        condensation.compute_elimination_fluxes(gb, gb_r, elimination_data)

        solver.split(gb_r, "pressure", p_red)

        # pp.fvutils.compute_discharges(gb)
        pp.fvutils.compute_discharges(gb_r)

        # ------Transport------#
        advection_discr = upwind.Upwind(physics="transport")
        advection_coupling_conditions = upwind.UpwindCoupling(advection_discr)
        advection_coupler = coupler.Coupler(
            advection_discr, advection_coupling_conditions
        )
        U_r, rhs_u_r = advection_coupler.matrix_rhs(gb_r)
        _, rhs_src_r = pp.IntegralMixedDim(physics="transport").matrix_rhs(gb_r)
        rhs_u_r = rhs_u_r + rhs_src_r
        deltaT = np.amin(
            gb_r.apply_function(
                advection_discr.cfl, advection_coupling_conditions.cfl
            ).data
        )

        theta_r = sps.linalg.spsolve(U_r, rhs_u_r)

        U_known, rhs_known, theta_known, deltaT_known = known_for_elimination()
        tol = 1e-7
        assert np.isclose(deltaT, deltaT_known, tol, tol)
        assert (np.amax(np.absolute(U_r - U_known))) < tol
        assert (np.amax(np.absolute(rhs_u_r - rhs_known))) < tol
        assert (np.amax(np.absolute(theta_r - theta_known))) < tol


# #------------------------------------------------------------------------------#
def fluxes_2d_1d_left_right_dir_neu():
    d_0 = np.array(
        [
            5.00000000e-01,
            5.04994426e-01,
            5.04994950e-01,
            5.00000000e-01,
            5.04994426e-01,
            5.04994950e-01,
            0.00000000e+00,
            0.00000000e+00,
            4.99442570e-03,
            5.24244319e-07,
            0.00000000e+00,
            0.00000000e+00,
            -4.99442570e-03,
            -5.24244319e-07,
        ]
    )
    d_1 = np.array([-1.01001192e-05, -1.11486078e-05, -1.00000000e-02])
    return d_0, d_1


# ------------------------------------------------------------------------------#


def known_for_elimination():
    U = np.array(
        [
            [
                5.00000000e-01,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                0.00000000e+00,
                5.28888404e-02,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                -5.28888404e-02,
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                5.00000000e-01,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                5.28888404e-02,
                0.00000000e+00,
                0.00000000e+00,
                -5.28888404e-02,
                0.00000000e+00,
            ],
            [
                0.00000000e+00,
                -3.65602442e-03,
                0.00000000e+00,
                -3.65602442e-03,
                9.11534368e-01,
                -3.49849812e-01,
                -2.77186253e-01,
                -2.77186253e-01,
            ],
            [
                -2.42588465e-01,
                0.00000000e+00,
                -2.42588465e-01,
                0.00000000e+00,
                0.00000000e+00,
                4.95176930e-01,
                0.00000000e+00,
                0.00000000e+00,
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                -2.57411535e-01,
                0.00000000e+00,
                0.00000000e+00,
                -7.26635590e-02,
                3.30075094e-01,
                -3.55271368e-15,
            ],
            [
                -2.57411535e-01,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                -7.26635590e-02,
                0.00000000e+00,
                3.30075094e-01,
            ],
        ]
    )

    rhs = np.array([0.25, 0.25, 0.25, 0.25, 0.005, 0.005, 0.005, 0.005])
    t = np.array(
        [0.5, 5.24204316, 0.5, 5.24204316, 0.55273715, 0.5, 0.51514807, 0.51514807]
    )
    dT = 0.00274262835006

    return U, rhs, t, dT
